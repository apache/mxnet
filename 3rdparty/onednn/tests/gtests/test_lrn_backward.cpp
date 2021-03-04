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

#include <cmath>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using fmt = memory::format_tag;

struct test_lrn_desc_t {
    memory::dim mb, c;
    memory::dim h, w;
    memory::dim local_size;
    float alpha, beta, k;
};

struct lrn_test_params_t {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    memory::format_tag data_format;
    memory::format_tag diff_data_format;
    test_lrn_desc_t test_ld;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename acc_data_t>
acc_data_t fast_inv_powf(acc_data_t omega, float beta) {
    if (beta == 0.75f) return (acc_data_t)(1.0f / sqrtf(sqrtf(omega) * omega));
    return (acc_data_t)(1.0f / std::pow(omega, beta));
}

template <typename data_t, typename acc_data_t = data_t>
void check_lrn_bwd(const lrn_test_params_t &p, const memory &src,
        const memory &diff_dst, const memory &diff_src) {
    auto src_ptr = map_memory<data_t>(src);
    auto diff_dst_ptr = map_memory<data_t>(diff_dst);
    auto diff_src_ptr = map_memory<data_t>(diff_src);

    const memory::dim N = p.test_ld.mb;
    const memory::dim C = p.test_ld.c;
    const memory::dim H = p.test_ld.h;
    const memory::dim W = p.test_ld.w;
    const auto padded_c = src.get_desc().data.padded_dims[1];

    const float alpha = p.test_ld.alpha;
    const float beta = p.test_ld.beta;
    const memory::dim size = p.test_ld.local_size;
    const memory::dim half_size = (size - 1) / 2;
    const memory::dim summands = p.aalgorithm == algorithm::lrn_across_channels
            ? size
            : size * size;

    data_t *ref_diff_src_ptr = new data_t[N * padded_c * H * W];

    const memory::desc src_d = src.get_desc();
    const memory::desc diff_dst_d = diff_dst.get_desc();
    const memory::desc diff_src_d = diff_src.get_desc();
    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper diff_dst_mdw(diff_dst_d.data);
    const dnnl::impl::memory_desc_wrapper diff_src_mdw(diff_src_d.data);

    auto off = [=](memory::dim n, memory::dim c, memory::dim h, memory::dim w) {
        return ((n * padded_c + c) * H + h) * W + w;
    };

    auto get_omega = [&](memory::dim n, memory::dim oc, memory::dim oh,
                             memory::dim ow) {
        acc_data_t sum = 0;

        if (p.aalgorithm == algorithm::lrn_across_channels) {
            const memory::dim c_st = std::max(oc - half_size, (memory::dim)0);
            const memory::dim c_en = std::min(oc + half_size + 1, C);

            for (memory::dim c = c_st; c < c_en; ++c) {
                acc_data_t s = src_ptr[src_mdw.off_l(off(n, c, oh, ow), true)];
                sum += s * s;
            }
        } else if (p.aalgorithm == algorithm::lrn_within_channel) {
            const memory::dim h_st = std::max(oh - half_size, (memory::dim)0);
            const memory::dim h_en = std::min(oh + half_size + 1, H);
            const memory::dim w_st = std::max(ow - half_size, (memory::dim)0);
            const memory::dim w_en = std::min(ow + half_size + 1, W);

            for (memory::dim h = h_st; h < h_en; ++h) {
                for (memory::dim w = w_st; w < w_en; ++w) {
                    acc_data_t s
                            = src_ptr[src_mdw.off_l(off(n, oc, h, w), true)];
                    sum += s * s;
                }
            }
        }

        return (acc_data_t)(p.test_ld.k + p.test_ld.alpha * sum / summands);
    };

    auto ker = [&](data_t *d, memory::dim n, memory::dim oc, memory::dim oh,
                       memory::dim ow) {
        acc_data_t A = 0, B = 0;

        if (p.aalgorithm == algorithm::lrn_across_channels) {
            const memory::dim c_st = std::max(oc - half_size, (memory::dim)0);
            const memory::dim c_en = std::min(oc + half_size + 1, C);

            for (memory::dim c = c_st; c < c_en; c++) {
                const acc_data_t omega = get_omega(n, c, oh, ow);
                const acc_data_t omega_in_beta
                        = fast_inv_powf<acc_data_t>(omega, p.test_ld.beta);
                const acc_data_t tmp = diff_dst_ptr[diff_dst_mdw.off_l(
                                               off(n, c, oh, ow), true)]
                        * omega_in_beta;

                if (c == oc) A = tmp;

                B += (src_ptr[src_mdw.off_l(off(n, c, oh, ow), true)] * tmp
                        / omega);
            }
        } else if (p.aalgorithm == algorithm::lrn_within_channel) {
            const memory::dim h_st = std::max(oh - half_size, (memory::dim)0);
            const memory::dim h_en = std::min(oh + half_size + 1, H);
            const memory::dim w_st = std::max(ow - half_size, (memory::dim)0);
            const memory::dim w_en = std::min(ow + half_size + 1, W);

            for (memory::dim h = h_st; h < h_en; h++)
                for (memory::dim w = w_st; w < w_en; w++) {
                    const acc_data_t omega = get_omega(n, oc, h, w);
                    const acc_data_t omega_in_beta
                            = fast_inv_powf<acc_data_t>(omega, p.test_ld.beta);
                    const acc_data_t tmp = diff_dst_ptr[diff_dst_mdw.off_l(
                                                   off(n, oc, h, w), true)]
                            * omega_in_beta;

                    if (h == oh && w == ow) A = tmp;

                    B += (src_ptr[src_mdw.off_l(off(n, oc, h, w), true)] * tmp
                            / omega);
                }
        }

        B *= src_ptr[src_mdw.off_l(off(n, oc, oh, ow), true)];
        B *= (2.0f * alpha * beta / summands);
        *d = A - B;
    };

    dnnl::impl::parallel_nd(N, C, H, W,
            [&](memory::dim n, memory::dim c, memory::dim h, memory::dim w) {
                if (is_current_test_failed()) return;

                ker(&ref_diff_src_ptr[diff_src_mdw.off_l(
                            off(n, c, h, w), true)],
                        n, c, h, w);
                auto exp = ref_diff_src_ptr[diff_src_mdw.off_l(
                        off(n, c, h, w), true)];
                auto got = diff_src_ptr[diff_src_mdw.off_l(
                        off(n, c, h, w), true)];
                acc_data_t eps = static_cast<acc_data_t>(2e-6 * size * size);
                memory::data_type data_type = data_traits<data_t>::data_type;
                if (data_type == dnnl::memory::data_type::bf16)
                    eps = static_cast<acc_data_t>(1e-2 * size * size);
                float diff = std::fabs(exp - got);
                if (got > 1e-2) // rel_diff
                    diff /= std::fabs(got);

                ASSERT_NEAR(diff, 0.0, eps);
            });

    delete[] ref_diff_src_ptr;
}

template <typename data_t>
class lrn_test_t : public ::testing::TestWithParam<lrn_test_params_t> {
private:
    std::shared_ptr<test_memory> src;
    std::shared_ptr<test_memory> dst;
    std::shared_ptr<test_memory> diff_src;
    std::shared_ptr<test_memory> diff_dst;
    memory workspace;
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    std::shared_ptr<memory::desc> diff_src_desc;
    std::shared_ptr<memory::desc> diff_dst_desc;
    lrn_forward::primitive_desc lrn_fwd_prim_desc;
    lrn_test_params_t p;
    memory::dims padR;
    engine eng;
    stream strm;
    memory::data_type data_type;

protected:
    void SetUp() override {
        data_type = data_traits<data_t>::data_type;

        SKIP_IF(data_type == memory::data_type::bf16
                        && get_test_engine_kind() == engine::kind::gpu,
                "GPU does not support bf16 data type.");
        SKIP_IF(unsupported_data_type(data_type),
                "Engine does not support this data type.");

        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        ASSERT_TRUE(p.aalgorithm == algorithm::lrn_across_channels
                || p.aalgorithm == algorithm::lrn_within_channel);

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();

        eng = get_test_engine();
        strm = make_stream(eng);
        ASSERT_EQ(true,
                dnnl::impl::utils::one_of(data_type,
                        dnnl::memory::data_type::f32,
                        dnnl::memory::data_type::bf16));

        test_lrn_desc_t ld = p.test_ld;

        src_desc.reset(new memory::desc(
                {ld.mb, ld.c, ld.h, ld.w}, data_type, p.data_format));
        dst_desc.reset(new memory::desc(
                {ld.mb, ld.c, ld.h, ld.w}, data_type, p.data_format));
        diff_src_desc.reset(new memory::desc(
                {ld.mb, ld.c, ld.h, ld.w}, data_type, p.diff_data_format));
        diff_dst_desc.reset(new memory::desc(
                {ld.mb, ld.c, ld.h, ld.w}, data_type, p.diff_data_format));

        Forward();
        Backward();
    }

    void Forward() {
        auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm, *src_desc,
                p.test_ld.local_size, p.test_ld.alpha, p.test_ld.beta,
                p.test_ld.k);
        lrn_fwd_prim_desc = lrn_forward::primitive_desc(lrn_desc, eng);

        src.reset(new test_memory(*src_desc, eng));
        dst.reset(new test_memory(*dst_desc, eng));

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());
        fill_data<data_t>(dst->get_size() / sizeof(data_t), dst->get());
        check_zero_tail<data_t>(1, src->get());
        check_zero_tail<data_t>(1, dst->get());

        // Execute
        auto l = lrn_forward(lrn_fwd_prim_desc);
        std::unordered_map<int, memory> args
                = {{DNNL_ARG_SRC, src->get()}, {DNNL_ARG_DST, dst->get()}};
        auto workspace_md = lrn_fwd_prim_desc.workspace_desc();
        workspace = test::make_memory(workspace_md, eng);
        args.insert({DNNL_ARG_WORKSPACE, workspace});
        l.execute(strm, args);
        strm.wait();
    }

    void Backward() {
        auto lrn_desc = lrn_backward::desc(p.aalgorithm, *src_desc,
                *diff_dst_desc, p.test_ld.local_size, p.test_ld.alpha,
                p.test_ld.beta, p.test_ld.k);

        src.reset(new test_memory(*src_desc, eng));
        diff_src.reset(new test_memory(*diff_src_desc, eng));
        diff_dst.reset(new test_memory(*diff_dst_desc, eng));

        auto lrn_prim_desc = lrn_backward::primitive_desc(
                lrn_desc, eng, lrn_fwd_prim_desc);
        // test construction from a C pd
        lrn_prim_desc = lrn_backward::primitive_desc(lrn_prim_desc.get());

        fill_data<data_t>(src->get_size() / sizeof(data_t), src->get());

        fill_data<data_t>(
                diff_dst->get_size() / sizeof(data_t), diff_dst->get());

        fill_data<data_t>(
                diff_src->get_size() / sizeof(data_t), diff_src->get());
        check_zero_tail<data_t>(1, src->get());
        check_zero_tail<data_t>(1, diff_dst->get());
        check_zero_tail<data_t>(1, diff_src->get());

        ASSERT_TRUE(lrn_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == lrn_prim_desc.src_desc());
        ASSERT_TRUE(
                lrn_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == lrn_prim_desc.diff_src_desc());
        ASSERT_TRUE(
                lrn_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == lrn_prim_desc.diff_dst_desc());
        ASSERT_TRUE(
                lrn_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_WORKSPACE)
                == lrn_prim_desc.workspace_desc());

        // Execute
        lrn_backward(lrn_prim_desc)
                .execute(strm,
                        {{DNNL_ARG_SRC, src->get()},
                                {DNNL_ARG_DIFF_DST, diff_dst->get()},
                                {DNNL_ARG_WORKSPACE, workspace},
                                {DNNL_ARG_DIFF_SRC, diff_src->get()}});
        strm.wait();

        check_zero_tail<data_t>(0, diff_src->get());

        check_lrn_bwd<data_t, float>(
                p, src->get(), diff_dst->get(), diff_src->get());
    }
};

static auto padded_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {0, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 0, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nChw16c, {2, 16, 0, 4, 5, 1.0e-4f, 0.75f, 3.0f}});
};

static auto EF_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {-1, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}, true,
                    dnnl_invalid_arguments},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {2, 10, -4, 4, 5, 1.0e-4f, 0.75f, 3.0f}, true,
                    dnnl_invalid_arguments});
};

static auto nChw16c_padded_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 17, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}});
};

static auto nChw8c_padded_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 7, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}});
};

static auto cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 4.0f}});
};

static auto NHWC_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 3.0f}});
};

static auto nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.0f}});
};

static auto nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.0f}});
};

static auto CaffeNCHW_cases = [](algorithm lk) {
    return ::testing::Values(lrn_test_params_t {prop_kind::forward_training, lk,
            fmt::nchw, fmt::nchw, {2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f}});
};

static auto CaffeNHWC_cases = [](algorithm lk) {
    return ::testing::Values(lrn_test_params_t {prop_kind::forward_training, lk,
            fmt::nhwc, fmt::nhwc, {2, 4, 5, 5, 5, 1.0f, 0.75f, 1.0f}});
};

static auto Caffe_nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(lrn_test_params_t {prop_kind::forward_training, lk,
            fmt::nChw8c, fmt::nChw8c, {2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f}});
};

static auto Caffe_nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(lrn_test_params_t {prop_kind::forward_training, lk,
            fmt::nChw16c, fmt::nChw16c, {2, 96, 55, 55, 3, 1.0f, 0.75f, 1.0f}});
};

static auto AlexnetNCHW_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto AlexnetNHWC_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nhwc,
                    fmt::nhwc, {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto Alexnet_nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto Alexnet_nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto GoogleNetV1NCHW_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nchw,
                    fmt::nchw, {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto GoogleNetV1_nChw8c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw8c,
                    fmt::nChw8c, {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto GoogleNetV1_nChw16c_cases = [](algorithm lk) {
    return ::testing::Values(
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_test_params_t {prop_kind::forward_training, lk, fmt::nChw16c,
                    fmt::nChw16c, {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto RegressionWeightFormat_cases = [](algorithm lk) {
    return ::testing::Values(lrn_test_params_t {prop_kind::forward_training, lk,
            fmt::oihw, fmt::oihw, {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

#define INST_TEST_CASE(test, lk) \
    TEST_P(test, TestsLRN) {} \
    INSTANTIATE_TEST_SUITE_P(Backward_padded, test, padded_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(BackwardEF, test, EF_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P( \
            Backward_nChw16c_padded, test, nChw16c_padded_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P( \
            Backward_nChw8c_padded, test, nChw8c_padded_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(LRN, test, cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(NHWC, test, NHWC_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(nChw8c, test, nChw8c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(nChw16c, test, nChw16c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(CaffeNCHW, test, CaffeNCHW_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(CaffeNHWC, test, CaffeNHWC_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(Caffe_nChw8c, test, Caffe_nChw8c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(Caffe_nChw16c, test, Caffe_nChw16c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(AlexnetNCHW, test, AlexnetNCHW_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(AlexnetNHWC, test, AlexnetNHWC_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(Alexnet_nChw8c, test, Alexnet_nChw8c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P( \
            Alexnet_nChw16c, test, Alexnet_nChw16c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P( \
            GoogleNetV1NCHW, test, GoogleNetV1NCHW_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P( \
            GoogleNetV1_nChw8c, test, GoogleNetV1_nChw8c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P( \
            GoogleNetV1_nChw16c, test, GoogleNetV1_nChw16c_cases(lk)); \
    INSTANTIATE_TEST_SUITE_P(RegressionWeightFormat, test, \
            RegressionWeightFormat_cases( \
                    lk)); // This tests compatibility with Intel MKL-DNN v0.14

using float_across = lrn_test_t<float>;
using float_within = lrn_test_t<float>;
using bfloat16_across = lrn_test_t<bfloat16_t>;
using bfloat16_within = lrn_test_t<bfloat16_t>;

INST_TEST_CASE(float_across, algorithm::lrn_across_channels)
INST_TEST_CASE(bfloat16_across, algorithm::lrn_across_channels)

INST_TEST_CASE(float_within, algorithm::lrn_within_channel)
INST_TEST_CASE(bfloat16_within, algorithm::lrn_within_channel)

} // namespace dnnl
