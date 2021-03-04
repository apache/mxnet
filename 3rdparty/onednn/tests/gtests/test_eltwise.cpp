/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

template <typename T, typename A>
inline T relu_fwd(T s, A alpha) {
    return s > 0 ? s : static_cast<T>(s * alpha);
}
template <typename T, typename A>
inline T relu_bwd(T dd, T s, A alpha) {
    return s > 0 ? dd : static_cast<T>(dd * alpha);
}
template <typename T>
T tanh_fwd(T s) {
    return static_cast<T>(::tanhf((float)s));
}
template <typename T>
T tanh_bwd(T dd, T s) {
    const float th = ::tanhf((float)s);
    return static_cast<T>(dd * (1 - th) * (1 + th));
}

template <typename T, typename A>
T elu_fwd(T s, A alpha) {
    return s > 0 ? s : static_cast<T>(alpha * (::expf(s) - 1));
}
template <typename T, typename A>
T elu_bwd(T dd, T s, A alpha) {
    return static_cast<T>(dd * (s > 0 ? 1 : alpha * ::expf(s)));
}

template <typename T>
T gelu_tanh_fwd(T s) {
    const float a = 0.797884;
    const float b = 0.044715;
    const float g = a * s * (1 + b * s * s);
    return static_cast<T>(0.5 * s * (1 + tanh_fwd(g)));
}

template <typename T>
T gelu_tanh_bwd(T dd, T s) {
    const float a = 0.797884;
    const float b = 0.044715;
    const float g = a * s * (1 + b * s * s);
    const float dg = a * (1 + 3 * b * s * s);
    return static_cast<T>(
            dd * (0.5 * (1 + tanh_fwd(g)) * (1 + s * (1 - tanh_fwd(g)) * dg)));
}

template <typename T>
T square_fwd(T s) {
    return s * s;
}

template <typename T>
T square_bwd(T dd, T s) {
    return dd * 2 * s;
}

template <typename T>
T abs_fwd(T s) {
    return s > 0 ? s : T(-s);
}

template <typename T>
T abs_bwd(T dd, T s) {
    return dd * (s > 0 ? 1 : s < 0 ? -1 : 0);
}

template <typename T, typename A>
T linear_fwd(T s, A alpha, A beta) {
    return alpha * s + beta;
}

template <typename T, typename A>
T linear_bwd(T dd, T s, A alpha, A beta) {
    (void)s;
    (void)beta;
    return dd * alpha;
}

template <typename T, typename A>
T bounded_relu_fwd(T s, A alpha) {
    s = s > 0 ? s : T(0);
    return s > alpha ? T(alpha) : s;
}

template <typename T, typename A>
T bounded_relu_bwd(T dd, T s, A alpha) {
    return dd * ((0 < s && s < alpha) ? 1 : 0);
}

template <typename T>
T soft_relu_fwd(T s) {
    return s < (T)logf(FLT_MAX) ? T(log1pf(::expf(s))) : s;
}

template <typename T>
T soft_relu_bwd(T dd, T s) {
    return dd / (1 + ::expf(-s));
}

template <typename T>
T logistic_fwd(T s) {
    float v = ::expf((float)-s);
    return (T)(1 / (1 + v));
}

template <typename T>
T logistic_bwd(T dd, T s) {
    float v = logistic_fwd<float>(s);
    return (T)(dd * v * (1 - v));
}

template <typename T>
T exp_fwd(T s) {
    return (T)(::expf((float)s));
}

template <typename T>
T exp_bwd(T dd, T s) {
    return dd * (::expf((float)s));
}

template <typename T, typename A>
T swish_fwd(T s, A alpha) {
    return (T)(s / (1.0f + ::expf(-alpha * (float)s)));
}

template <typename T, typename A>
T swish_bwd(T dd, T s, A alpha) {
    float v = logistic_fwd<float>(alpha * s);
    return dd * (v + s * alpha * v * (1 - v));
}

template <typename T>
T gelu_erf_fwd(T s) {
    const float sqrt_2_over_2 = 0.707106;
    float v = s * sqrt_2_over_2;
    return (T)(sqrt_2_over_2 * v * (1.f + ::erff(v)));
}

template <typename T>
T gelu_erf_bwd(T dd, T s) {
    const float two_over_sqrt_pi = 1.128379;
    const float sqrt_2_over_2 = 0.707106;
    float v = s * sqrt_2_over_2;
    return (T)(dd * 0.5f
            * (1.f + ::erff(v) + v * two_over_sqrt_pi * ::expf(-v * v)));
}

struct eltwise_test_params_t {
    algorithm alg_kind;
    memory::format_tag data_format;
    memory::format_tag diff_format;
    float alpha, beta;
    memory::dims dims;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

memory::dim n_elems(const memory::desc &md) {
    memory::dim p = 1;
    const auto *pdims = md.data.padded_dims;
    for (int i = 0; i < md.data.ndims; ++i)
        p *= pdims[i];
    return p;
}

template <typename data_t>
void check_eltwise_fwd(const eltwise_test_params_t &p, const memory::desc &md,
        const memory &src, const memory &dst) {
    auto src_data = map_memory<data_t>(src);
    auto dst_data = map_memory<data_t>(dst);

    memory::dim n = n_elems(md);
    for (memory::dim i = 0; i < n; ++i) {
        data_t s = src_data[i];
        data_t ref_d = 0;
        switch (p.alg_kind) {
            case algorithm::eltwise_relu: ref_d = relu_fwd(s, p.alpha); break;
            case algorithm::eltwise_tanh: ref_d = tanh_fwd(s); break;
            case algorithm::eltwise_elu: ref_d = elu_fwd(s, p.alpha); break;
            case algorithm::eltwise_square: ref_d = square_fwd(s); break;
            case algorithm::eltwise_abs: ref_d = abs_fwd(s); break;
            case algorithm::eltwise_linear:
                ref_d = linear_fwd(s, p.alpha, p.beta);
                break;
            case algorithm::eltwise_bounded_relu:
                ref_d = bounded_relu_fwd(s, p.alpha);
                break;
            case algorithm::eltwise_soft_relu: ref_d = soft_relu_fwd(s); break;
            case algorithm::eltwise_logistic: ref_d = logistic_fwd(s); break;
            case algorithm::eltwise_exp: ref_d = exp_fwd(s); break;
            case algorithm::eltwise_gelu_tanh: ref_d = gelu_tanh_fwd(s); break;
            case algorithm::eltwise_swish: ref_d = swish_fwd(s, p.alpha); break;
            case algorithm::eltwise_gelu_erf: ref_d = gelu_erf_fwd(s); break;
            default: assert(!"unknown alg_kind");
        }
        dst_data[i] = ref_d;
    }
}

template <typename data_t>
void compare_eltwise_fwd(const eltwise_test_params_t &p, const memory::desc &md,
        const memory &dst, const memory &ref_dst) {
    data_t eps;
    if (data_traits<data_t>::data_type == memory::data_type::s8
            || data_traits<data_t>::data_type == memory::data_type::s32)
        eps = 0;
    else
        eps = static_cast<data_t>(
                (data_traits<data_t>::data_type == memory::data_type::f16
                        || data_traits<data_t>::data_type
                                == memory::data_type::bf16)
                        ? 5e-2
                        : (p.alg_kind == algorithm::eltwise_elu
                                  || p.alg_kind == algorithm::eltwise_gelu_tanh
                                  || p.alg_kind == algorithm::eltwise_gelu_erf)
                                ? 2e-5
                                : p.alg_kind == algorithm::eltwise_soft_relu
                                        ? 3e-5
                                        : 1e-6);
    compare_data(ref_dst, dst, eps);
}

template <typename data_t>
void check_eltwise_bwd(const eltwise_test_params_t &p, const memory::desc &md,
        const memory &src, const memory &diff_dst, const memory &diff_src) {
    auto src_data = map_memory<data_t>(src);
    auto diff_dst_data = map_memory<data_t>(diff_dst);
    auto diff_src_data = map_memory<data_t>(diff_src);

    const memory::desc data_d = src.get_desc();
    const memory::desc diff_data_d = diff_src.get_desc();
    const dnnl::impl::memory_desc_wrapper data_mdw(data_d.data);
    const dnnl::impl::memory_desc_wrapper diff_data_mdw(diff_data_d.data);

    float eps_f = 0;
    if (p.alg_kind == algorithm::eltwise_soft_relu) {
        eps_f = 2e-6f;
    } else if (p.alg_kind == algorithm::eltwise_tanh) {
        eps_f = (get_test_engine_kind() == engine::kind::gpu) ? 2e-5f : 2e-6f;
    } else if (p.alg_kind == algorithm::eltwise_gelu_tanh
            || p.alg_kind == algorithm::eltwise_gelu_erf) {
        eps_f = 1e-5f;
    } else {
        eps_f = 1e-6f;
    }
    data_t eps = static_cast<data_t>(eps_f);

    memory::dim n = n_elems(md);
    for (memory::dim i = 0; i < n; ++i) {
        data_t ref_s = src_data[data_mdw.off_l(i)];
        data_t ref_dd = diff_dst_data[diff_data_mdw.off_l(i)];
        data_t ref_ds = 0;
        switch (p.alg_kind) {
            case algorithm::eltwise_relu:
                ref_ds = relu_bwd(ref_dd, ref_s, p.alpha);
                break;
            case algorithm::eltwise_tanh:
                ref_ds = tanh_bwd(ref_dd, ref_s);
                break;
            case algorithm::eltwise_elu:
                ref_ds = elu_bwd(ref_dd, ref_s, p.alpha);
                break;
            case algorithm::eltwise_square:
                ref_ds = square_bwd(ref_dd, ref_s);
                break;
            case algorithm::eltwise_abs: ref_ds = abs_bwd(ref_dd, ref_s); break;
            case algorithm::eltwise_linear:
                ref_ds = linear_bwd(ref_dd, ref_s, p.alpha, p.beta);
                break;
            case algorithm::eltwise_bounded_relu:
                ref_ds = bounded_relu_bwd(ref_dd, ref_s, p.alpha);
                break;
            case algorithm::eltwise_soft_relu:
                ref_ds = soft_relu_bwd(ref_dd, ref_s);
                break;
            case algorithm::eltwise_logistic:
                ref_ds = logistic_bwd(ref_dd, ref_s);
                break;
            case algorithm::eltwise_exp: ref_ds = exp_bwd(ref_dd, ref_s); break;
            case algorithm::eltwise_gelu_tanh:
                ref_ds = gelu_tanh_bwd(ref_dd, ref_s);
                break;
            case algorithm::eltwise_swish:
                ref_ds = swish_bwd(ref_dd, ref_s, p.alpha);
                break;
            case algorithm::eltwise_gelu_erf:
                ref_ds = gelu_erf_bwd(ref_dd, ref_s);
                break;
            default: assert(!"unknown alg_kind");
        }

        data_t tgt = diff_src_data[diff_data_mdw.off_l(i)];
        const data_t diff = tgt == ref_ds ? 0 : tgt - ref_ds;
        data_t error = (std::abs(ref_ds) > eps)
                ? static_cast<data_t>(diff / ref_ds)
                : diff;
        if (p.alg_kind == algorithm::eltwise_logistic
                && (tgt < 1e-3)) { // check for cancellation
            error = diff;
        }
        ASSERT_NEAR(error, 0.0, eps);
    }
}

template <typename data_t>
class eltwise_test_t : public ::testing::TestWithParam<eltwise_test_params_t> {
private:
    memory src;
    std::shared_ptr<memory::desc> data_desc;
    eltwise_forward::primitive_desc eltwise_prim_desc;
    eltwise_test_params_t p;
    engine eng;
    stream strm;
    memory::data_type data_type;

protected:
    void SetUp() override {
        data_type = data_traits<data_t>::data_type;
        SKIP_IF(unsupported_data_type(data_type),
                "Engine does not support this data type.");
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        SKIP_IF((p.alg_kind != algorithm::eltwise_relu
                        || (p.alg_kind == algorithm::eltwise_relu
                                && p.alpha != 0.0))
                        && (data_type == memory::data_type::s32
                                || data_type == memory::data_type::s8),
                "oneDNN only supports relu w/ slope=0 for integers");
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        p = ::testing::TestWithParam<eltwise_test_params_t>::GetParam();

        eng = get_test_engine();
        strm = make_stream(eng);

        Forward();
        if (data_type == memory::data_type::f32
                || data_type == memory::data_type::bf16) {
            Backward();
        }
    }

    void Forward() {
        data_desc.reset(new memory::desc(p.dims, data_type, p.data_format));
        src = test::make_memory(*data_desc, eng);
        auto dst = test::make_memory(*data_desc, eng);
        auto ref_dst = test::make_memory(*data_desc, eng);

        data_t data_median = data_t(0);
        data_t data_deviation = (p.alg_kind == algorithm::eltwise_elu
                                        || p.alg_kind == algorithm::eltwise_exp)
                        || (p.alg_kind == algorithm::eltwise_swish)
                ? data_t(1.0)
                : p.alg_kind == algorithm::eltwise_square ? data_t(6.0)
                                                          : data_t(100.0);
        fill_data<data_t>(
                n_elems(*data_desc), src, data_median, data_deviation);
        check_zero_tail<data_t>(1, src);

        auto eltwise_desc = eltwise_forward::desc(prop_kind::forward_training,
                p.alg_kind, *data_desc, p.alpha, p.beta);
        eltwise_prim_desc = eltwise_forward::primitive_desc(eltwise_desc, eng);
        eltwise_prim_desc = eltwise_forward::primitive_desc(
                eltwise_prim_desc.get()); // test construction from a C pd

        ASSERT_TRUE(eltwise_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == eltwise_prim_desc.src_desc());
        ASSERT_TRUE(eltwise_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == eltwise_prim_desc.dst_desc());

        eltwise_forward(eltwise_prim_desc)
                .execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
        strm.wait();

        check_zero_tail<data_t>(0, dst);
        check_eltwise_fwd<data_t>(p, *data_desc, src, ref_dst);
        check_zero_tail<data_t>(1, ref_dst);
        compare_eltwise_fwd<data_t>(p, *data_desc, dst, ref_dst);
    }

    void Backward() {
        memory::desc diff_data_desc(p.dims, data_type, p.diff_format);
        auto diff_src = test::make_memory(diff_data_desc, eng);
        auto diff_dst = test::make_memory(diff_data_desc, eng);

        data_t data_median = data_t(0);
        data_t data_deviation = p.alg_kind == algorithm::eltwise_elu
                ? data_t(1.0)
                : p.alg_kind == algorithm::eltwise_square ? data_t(6.0)
                                                          : data_t(100.0);
        fill_data<data_t>(
                n_elems(diff_data_desc), diff_dst, data_median, data_deviation);
        check_zero_tail<data_t>(1, diff_dst);

        auto eltwise_bwd_desc = eltwise_backward::desc(
                p.alg_kind, diff_data_desc, *data_desc, p.alpha, p.beta);
        auto eltwise_bwd_prim_desc = eltwise_backward::primitive_desc(
                eltwise_bwd_desc, eng, eltwise_prim_desc);
        eltwise_bwd_prim_desc
                = eltwise_backward::primitive_desc(eltwise_bwd_prim_desc.get());

        ASSERT_TRUE(
                eltwise_bwd_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == eltwise_bwd_prim_desc.src_desc());
        ASSERT_TRUE(
                eltwise_bwd_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == eltwise_bwd_prim_desc.dst_desc());

        eltwise_backward(eltwise_bwd_prim_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, src}, {DNNL_ARG_DIFF_DST, diff_dst},
                                {DNNL_ARG_DIFF_SRC, diff_src}});
        strm.wait();

        check_zero_tail<data_t>(0, diff_src);
        check_eltwise_bwd<data_t>(p, *data_desc, src, diff_dst, diff_src);
    }
};

using eltwise_test_f16 = eltwise_test_t<float16_t>;
using eltwise_test_bf16 = eltwise_test_t<bfloat16_t>;
using eltwise_test_f32 = eltwise_test_t<float>;
using eltwise_test_s32 = eltwise_test_t<int>;
using eltwise_test_s8 = eltwise_test_t<int8_t>;

TEST_P(eltwise_test_f16, TestsEltwise) {}

TEST_P(eltwise_test_bf16, TestsEltwise) {}

TEST_P(eltwise_test_f32, TestsEltwise) {}

TEST_P(eltwise_test_s32, TestsEltwise) {}

TEST_P(eltwise_test_s8, TestsEltwise) {}

#define EXPAND(args) args

#define EXPAND_FORMATS(data) memory::format_tag::data
#define EXPAND_DIMS(...) \
    { __VA_ARGS__ }

#define PARAMS(alg, data, diff_data, alpha, beta, ...) \
    eltwise_test_params_t { \
        algorithm::alg, EXPAND_FORMATS(data), EXPAND_FORMATS(diff_data), \
                alpha, beta, EXPAND_DIMS(__VA_ARGS__) \
    }

#define PARAMS_ALL_ALG(...) \
    EXPAND(PARAMS(eltwise_gelu_tanh, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_relu, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_tanh, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_elu, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_square, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_abs, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_exp, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_swish, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_gelu_erf, __VA_ARGS__))

#define PARAMS_ALL_ALG_SDPART(...) \
    EXPAND(PARAMS(eltwise_linear, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_soft_relu, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_bounded_relu, __VA_ARGS__)), \
            EXPAND(PARAMS(eltwise_logistic, __VA_ARGS__))

#define _CPU_INST_TEST_CASE(str, data_t, ...) \
    CPU_INSTANTIATE_TEST_SUITE_P(str##_##data_t, eltwise_test_##data_t, \
            ::testing::Values(__VA_ARGS__))

#define _INST_TEST_CASE(str, data_t, ...) \
    INSTANTIATE_TEST_SUITE_P_(str##_##data_t, eltwise_test_##data_t, \
            ::testing::Values(__VA_ARGS__))

#define CPU_INST_TEST_CASE_BF16(str, ...) \
    _CPU_INST_TEST_CASE(str, bf16, __VA_ARGS__);
#define INST_TEST_CASE_BF16(str, ...) _INST_TEST_CASE(str, bf16, __VA_ARGS__);
#define GPU_INST_TEST_CASE_F16(str, ...) \
    GPU_INSTANTIATE_TEST_SUITE_P_(TEST_CONCAT(str, _f16), eltwise_test_f16, \
            ::testing::Values(__VA_ARGS__));
#define CPU_INST_TEST_CASE_F32(str, ...) \
    _CPU_INST_TEST_CASE(str, f32, __VA_ARGS__);
#define INST_TEST_CASE_F32(str, ...) _INST_TEST_CASE(str, f32, __VA_ARGS__);
#define CPU_INST_TEST_CASE_S32(str, ...) \
    _CPU_INST_TEST_CASE(str, s32, __VA_ARGS__);
#define INST_TEST_CASE_S32(str, ...) _INST_TEST_CASE(str, s32, __VA_ARGS__);
#define CPU_INST_TEST_CASE_S8(str, ...) \
    _CPU_INST_TEST_CASE(str, s8, __VA_ARGS__);
#define INST_TEST_CASE_S8(str, ...) _INST_TEST_CASE(str, s8, __VA_ARGS__);

#define CPU_INST_TEST_CASE(str, ...) \
    CPU_INST_TEST_CASE_F32(str, __VA_ARGS__) \
    CPU_INST_TEST_CASE_BF16(str, __VA_ARGS__) \
    CPU_INST_TEST_CASE_S32(str, __VA_ARGS__) \
    CPU_INST_TEST_CASE_S8(str, __VA_ARGS__)

#define INST_TEST_CASE(str, ...) \
    GPU_INST_TEST_CASE_F16(str, __VA_ARGS__) \
    INST_TEST_CASE_BF16(str, __VA_ARGS__) \
    INST_TEST_CASE_F32(str, __VA_ARGS__) \
    INST_TEST_CASE_S32(str, __VA_ARGS__) \
    INST_TEST_CASE_S8(str, __VA_ARGS__)

INST_TEST_CASE(SimpleZeroDim,
        PARAMS_ALL_ALG(ncdhw, nCdhw8c, 0.1f, 0.f, 0, 2, 4, 4, 4),
        PARAMS_ALL_ALG(ncdhw, nCdhw8c, 0.1f, 0.f, 2, 0, 4, 4, 4),
        PARAMS_ALL_ALG_SDPART(nCdhw16c, nCdhw16c, 0.1f, 0.2f, 0, 4, 2, 2, 2),
        PARAMS_ALL_ALG_SDPART(nCdhw16c, nCdhw16c, 0.1f, 0.2f, 4, 0, 2, 2, 2));

#define CASE_EF(alg, d0, d1, d2, d3) \
    eltwise_test_params_t { \
        algorithm::eltwise_##alg, EXPAND_FORMATS(nchw), EXPAND_FORMATS(nchw), \
                0.f, 0.f, {d0, d1, d2, d3}, true, dnnl_invalid_arguments \
    }
INST_TEST_CASE(SimpleExpectedFails, CASE_EF(relu, -1, 2, 4, 4),
        CASE_EF(logistic, -1, 2, 4, 4), CASE_EF(relu, 1, -2, 4, 4),
        CASE_EF(logistic, 1, -2, 4, 4));

INST_TEST_CASE(Simple_3D,
        PARAMS_ALL_ALG(ncdhw, nCdhw8c, 0.1f, 0.f, 2, 8, 4, 4, 4),
        PARAMS_ALL_ALG(nCdhw8c, ncdhw, 0.1f, 0.f, 2, 16, 4, 4, 4),
        PARAMS_ALL_ALG(ncdhw, ncdhw, 0.1f, 0.f, 2, 16, 8, 8, 8),
        PARAMS_ALL_ALG(nCdhw8c, nCdhw8c, 0.1f, 0.f, 2, 16, 16, 8, 6),
        PARAMS_ALL_ALG(ndhwc, ncdhw, 0.1f, 0.f, 2, 16, 10, 8, 6),
        PARAMS_ALL_ALG(ncdhw, ndhwc, 0.1f, 0.f, 10, 10, 10, 10, 10));

INST_TEST_CASE(Simple_blocked_3d_padded,
        PARAMS_ALL_ALG(nCdhw16c, nCdhw16c, 0.1f, 0.2f, 4, 15, 2, 2, 2),
        PARAMS_ALL_ALG_SDPART(nCdhw16c, nCdhw16c, 0.1f, 0.2f, 4, 27, 2, 2, 2),
        PARAMS_ALL_ALG(nCdhw16c, nCdhw16c, 0.1f, 0.2f, 4, 23, 2, 2, 2),
        PARAMS_ALL_ALG_SDPART(nCdhw16c, nCdhw16c, 0.1f, 0.2f, 4, 23, 7, 7, 7));

INST_TEST_CASE(Simple_blocked_padded,
        PARAMS_ALL_ALG(nChw16c, nChw16c, 0.1f, 0.2f, 4, 15, 2, 2),
        PARAMS_ALL_ALG_SDPART(nChw16c, nChw16c, 0.1f, 0.2f, 4, 27, 2, 2),
        PARAMS_ALL_ALG(nChw16c, nChw16c, 0.1f, 0.2f, 4, 23, 2, 2),
        PARAMS_ALL_ALG_SDPART(nChw16c, nChw16c, 0.1f, 0.2f, 4, 17, 7, 7),
        PARAMS_ALL_ALG(nChw8c, nChw8c, 0.1f, 0.2f, 4, 15, 2, 2),
        PARAMS_ALL_ALG_SDPART(nChw8c, nChw8c, 0.1f, 0.2f, 4, 27, 2, 2),
        PARAMS_ALL_ALG(nChw8c, nChw8c, 0.1f, 0.2f, 4, 23, 2, 2),
        PARAMS_ALL_ALG_SDPART(nChw8c, nChw8c, 0.1f, 0.2f, 4, 17, 7, 7));

CPU_INST_TEST_CASE(Simple_NCDHW,
        PARAMS_ALL_ALG(ncdhw, ncdhw, 0.f, 0.f, 2, 32, 28, 28, 28),
        PARAMS_ALL_ALG(ncdhw, ncdhw, 1.f, 0.f, 2, 64, 13, 13, 13),
        PARAMS_ALL_ALG(ncdhw, ncdhw, 1.f, 1.f, 1, 64, 27, 27, 27),
        PARAMS_ALL_ALG(ncdhw, ncdhw, 0.f, 1.f, 1, 128, 11, 11, 11));

CPU_INST_TEST_CASE(SimpleZeroNegativeSlope,
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 2, 8, 4, 4),
        PARAMS_ALL_ALG(nChw16c, nChw16c, 0.f, 0.f, 2, 16, 4, 4),
        PARAMS_ALL_ALG(nChw8c, nChw8c, 0.f, 0.f, 2, 16, 8, 8),
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 10, 10, 10, 10),
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 256, 64, 8, 16),
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 1, 1, 1, 1),
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 3, 5, 7, 11));

INST_TEST_CASE(Simple_NCHW, PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 2, 8, 4, 4),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 2, 16, 4, 4),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 2, 16, 8, 8),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 2, 16, 16, 8),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 2, 16, 10, 8),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 10, 10, 10, 10),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 256, 64, 8, 16),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 1, 1, 1, 1),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 3, 5, 7, 11));

INST_TEST_CASE(Simple_NCHW_SDPART,
        PARAMS_ALL_ALG_SDPART(nchw, nchw, 0.1f, 0.f, 256, 64, 8, 16));

CPU_INST_TEST_CASE(Simple, PARAMS_ALL_ALG(nchw, nChw8c, 0.1f, 0.f, 2, 8, 4, 4),
        PARAMS_ALL_ALG(nChw8c, nchw, 0.1f, 0.f, 2, 16, 4, 4),
        PARAMS_ALL_ALG(nchw, nchw, 0.1f, 0.f, 2, 16, 8, 8),
        PARAMS_ALL_ALG(nChw8c, nChw8c, 0.1f, 0.f, 2, 16, 16, 8),
        PARAMS_ALL_ALG(nhwc, nchw, 0.1f, 0.f, 2, 16, 10, 8),
        PARAMS_ALL_ALG(nchw, nhwc, 0.1f, 0.f, 10, 10, 10, 10));

CPU_INST_TEST_CASE(Simple_SDPART,
        PARAMS_ALL_ALG_SDPART(nchw, nChw8c, 0.1f, 0.f, 2, 8, 4, 4),
        PARAMS_ALL_ALG_SDPART(nChw8c, nchw, 0.1f, 0.f, 2, 16, 4, 4),
        PARAMS_ALL_ALG_SDPART(nchw, nchw, 0.1f, 0.f, 2, 16, 8, 8),
        PARAMS_ALL_ALG_SDPART(nChw8c, nChw8c, 0.1f, 0.f, 2, 16, 16, 8),
        PARAMS_ALL_ALG_SDPART(nhwc, nchw, 0.1f, 0.f, 2, 16, 10, 8),
        PARAMS_ALL_ALG_SDPART(nchw, nhwc, 0.1f, 0.f, 10, 10, 10, 10));

INST_TEST_CASE(AlexNet_NCHW,
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 2, 96, 55, 55),
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 2, 256, 27, 27),
        PARAMS_ALL_ALG(nchw, nchw, 0.f, 0.f, 2, 384, 13, 13),
        PARAMS_ALL_ALG_SDPART(nchw, nchw, 0.f, 0.f, 2, 96, 55, 55),
        PARAMS_ALL_ALG_SDPART(nchw, nchw, 0.f, 0.f, 2, 256, 27, 27),
        PARAMS_ALL_ALG_SDPART(nchw, nchw, 0.f, 0.f, 2, 384, 13, 13));

INST_TEST_CASE(Simple_X, PARAMS_ALL_ALG(x, x, 0.f, 0.f, 55));

} // namespace dnnl
