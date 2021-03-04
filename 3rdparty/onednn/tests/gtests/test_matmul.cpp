/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain src copy of the License at
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

#include <vector>

namespace dnnl {

namespace P {
// Common
unsigned NONE = 0u;

unsigned RUNTIME = 1u << 31;

unsigned SCALES = 1u << 30;
unsigned ZERO_POINTS = 1u << 29;

unsigned LEADING_DIM = 1u << 28;

// matrices indices: 1 .. 7
// bits reserved: 20 .. 22
unsigned MATRIX_MASK = 7u << 20;
unsigned SRC = 1u << 20;
unsigned WEIGHTS = 2u << 20;
unsigned DST = 3u << 20;

// scales and zero points: 1 .. 3
// bits reserved: 0 .. 1
unsigned MASK_MASK = 3u << 0;
unsigned COMMON = 1u << 0;
unsigned PER_N = 1u << 1;
} // namespace P

struct matmul_base_t {
    struct md_t {
        memory::dims dims;
        memory::data_type dt;
        memory::format_tag tag;
        unsigned flags;
    } src, weights, dst;
    memory::data_type bia_dt;
};

// TODO: src way to generalize?
struct matmul_attr_t {
    // ctor {P::SCALE, {P::SRC, P::WEIGHTS, P::DST}, {P::POST_OPS, ...}}

    unsigned scale_flags;

    struct zero_points_t {
        unsigned src, weights, dst;
    } zero_points;

    struct post_op_t {
        primitive::kind kind;
        algorithm alg;
    };

    std::vector<post_op_t> post_ops;
};

struct matmul_test_params_t {
    matmul_base_t base;
    matmul_attr_t attr;

    bool expect_to_fail;
    dnnl_status_t expected_status;
};

using tag = memory::format_tag;

class matmul_iface_test_t
    : public ::testing::TestWithParam<matmul_test_params_t> {
protected:
    void SetUp() override {
        matmul_test_params_t p
                = ::testing::TestWithParam<decltype(p)>::GetParam();

        SKIP_IF(unsupported_data_type(p.base.src.dt),
                "Engine does not support this data type.");

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status, false);
    }

    // use `force_no_rt = true` when create final memory
    static memory::desc init_md(
            const matmul_base_t::md_t &desc, bool force_no_rt = false) {
        const bool runtime = force_no_rt ? false : (desc.flags & P::RUNTIME);
        const bool use_ld = (desc.flags & P::LEADING_DIM);

        memory::dims dims = desc.dims;
        if (runtime)
            dims = memory::dims(desc.dims.size(), DNNL_RUNTIME_DIM_VAL);

        if (runtime || use_ld == false)
            return memory::desc(dims, desc.dt, desc.tag);

        memory::dims strides;
        switch (desc.tag) {
            case tag::ab: strides = {dims[1] + 1, 1}; break;
            case tag::ba: strides = {1, dims[0] + 1}; break;
            case tag::abc:
                strides = {dims[1] * (dims[2] + 1) + 1, dims[2] + 1, 1};
                break;
            case tag::acb:
                strides = {dims[1] * (dims[2] + 1) + 1, dims[2] + 1, 1};
                break;
            default:
                throw std::invalid_argument("tag doesn't support custom ld");
        }

        return memory::desc(dims, desc.dt, strides);
    }

    static void create_attr(const matmul_test_params_t &p, primitive_attr &attr,
            memory &scales_m, memory &zero_points_src_m,
            memory &zero_points_weights_m, memory &zero_points_dst_m,
            engine &eng) {
        const int ndims = (int)p.base.dst.dims.size();

        // output scales
        if (p.attr.scale_flags != P::NONE) {
            ASSERT_TRUE(p.attr.scale_flags & P::SCALES);

            unsigned scales_mask = p.attr.scale_flags & P::MASK_MASK;
            ASSERT_TRUE(scales_mask == P::COMMON || scales_mask == P::PER_N);

            int mask = scales_mask == P::PER_N ? 1 << (ndims - 1) : 0;
            memory::dim scale_size = mask ? p.base.dst.dims[ndims - 1] : 1;

            if (p.attr.scale_flags & P::RUNTIME) {
                attr.set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});

                scales_m = test::make_memory(
                        {{scale_size}, memory::data_type::f32, {1}}, eng);
                auto s = map_memory<float>(scales_m);
                for (memory::dim i = 0; i < scale_size; ++i)
                    s[i] = 2.f;
            } else {
                std::vector<float> scales(scale_size, 2.f);
                attr.set_output_scales(mask, scales);
            }
        }

        // zero points
        auto handle_zero_points = [&](int arg, unsigned flags,
                                          const matmul_base_t::md_t &md,
                                          memory &zero_points_m) {
            if (flags == P::NONE) return;

            ASSERT_TRUE(flags & P::ZERO_POINTS);
            ASSERT_TRUE(flags & P::MATRIX_MASK);

            // sanity check
            switch (arg) {
                case DNNL_ARG_SRC:
                    ASSERT_TRUE((flags & P::MATRIX_MASK) == P::SRC);
                    break;
                case DNNL_ARG_WEIGHTS:
                    ASSERT_TRUE((flags & P::MATRIX_MASK) == P::WEIGHTS);
                    break;
                case DNNL_ARG_DST:
                    ASSERT_TRUE((flags & P::MATRIX_MASK) == P::DST);
                    break;
                default: ASSERT_TRUE(!"unreachable");
            }

            unsigned zero_points_mask = flags & P::MASK_MASK;
            ASSERT_TRUE(zero_points_mask == P::COMMON
                    || zero_points_mask == P::PER_N);
            int mask = zero_points_mask == P::PER_N ? 1 << (ndims - 1) : 0;
            memory::dim zero_points_size = mask ? md.dims[ndims - 1] : 1;

            if (flags & P::RUNTIME) {
                attr.set_zero_points(arg, mask, {DNNL_RUNTIME_S32_VAL});
                zero_points_m = test::make_memory(
                        {{zero_points_size}, memory::data_type::s32, {1}}, eng);
                auto z = map_memory<int32_t>(zero_points_m);
                for (memory::dim i = 0; i < zero_points_size; ++i)
                    z[i] = (arg % 7) - 3;
            } else {
                std::vector<int32_t> zero_points(
                        zero_points_size, (arg % 7) - 3);
                attr.set_zero_points(arg, mask, zero_points);
            }
        };

        handle_zero_points(DNNL_ARG_SRC, p.attr.zero_points.src, p.base.src,
                zero_points_src_m);
        handle_zero_points(DNNL_ARG_WEIGHTS, p.attr.zero_points.weights,
                p.base.weights, zero_points_weights_m);
        handle_zero_points(DNNL_ARG_DST, p.attr.zero_points.dst, p.base.dst,
                zero_points_dst_m);

        // post ops
        post_ops po;
        for (auto post_op : p.attr.post_ops) {
            switch (post_op.kind) {
                case primitive::kind::sum: po.append_sum(); break;
                case primitive::kind::eltwise:
                    po.append_eltwise(1.f, post_op.alg, 0.f, 0.f);
                    break;
                default: ASSERT_TRUE(!"unknown post op kind");
            }
        }
        attr.set_post_ops(po);
    }

    void Test() {
        matmul_test_params_t p
                = ::testing::TestWithParam<matmul_test_params_t>::GetParam();

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto check_matrix_flags = [](unsigned flags, unsigned matrix) {
            if (flags) { ASSERT_EQ(flags & P::MATRIX_MASK, matrix); }
        };
        check_matrix_flags(p.base.src.flags, P::SRC);
        check_matrix_flags(p.base.weights.flags, P::WEIGHTS);
        check_matrix_flags(p.base.dst.flags, P::DST);

        auto src_md = init_md(p.base.src);
        auto weights_md = init_md(p.base.weights);
        auto dst_md = init_md(p.base.dst);

        auto bia_md = memory::desc();
        memory bia_m;
        if (p.base.bia_dt != memory::data_type::undef) {
            memory::dims bia_dims(p.base.dst.dims.size() - 1, 1);
            bia_dims.push_back(p.base.dst.dims.back());
            tag bia_tag = bia_dims.size() == 2 ? tag::ab : tag::abc;
            bia_md = init_md({bia_dims, p.base.bia_dt, bia_tag,
                    p.base.dst.flags & P::RUNTIME});
            bia_m = test::make_memory(
                    init_md({bia_dims, p.base.bia_dt, bia_tag}), eng);
        }

        auto matmul_d = matmul::desc(src_md, weights_md, bia_md, dst_md);

        primitive_attr attr;
        memory scales_m, zero_points_src_m, zero_points_weights_m,
                zero_points_dst_m;
        create_attr(p, attr, scales_m, zero_points_src_m, zero_points_weights_m,
                zero_points_dst_m, eng);

        auto matmul_pd = matmul::primitive_desc(matmul_d, attr, eng);

        ASSERT_TRUE(matmul_pd.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == matmul_pd.src_desc());
        ASSERT_TRUE(matmul_pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS)
                == matmul_pd.weights_desc());
        ASSERT_TRUE(matmul_pd.query_md(query::exec_arg_md, DNNL_ARG_BIAS)
                == matmul_pd.bias_desc());
        ASSERT_TRUE(matmul_pd.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == matmul_pd.dst_desc());

        auto matmul_p = matmul(matmul_pd);

        auto src_m = test::make_memory(init_md(p.base.src, true), eng);
        auto weights_m = test::make_memory(init_md(p.base.weights, true), eng);
        auto dst_m = test::make_memory(init_md(p.base.dst, true), eng);

        // Initialize memory to make sanitizers happy
        auto set_to_zero = [](memory &m) {
            if (m) {
                auto p = map_memory<char>(m);
                memset(p, 0, m.get_desc().get_size());
            }
        };
        set_to_zero(src_m);
        set_to_zero(weights_m);
        set_to_zero(dst_m);
        set_to_zero(bia_m);

        matmul_p.execute(strm,
                {
                        {DNNL_ARG_SRC, src_m},
                        {DNNL_ARG_WEIGHTS, weights_m},
                        {DNNL_ARG_BIAS, bia_m},
                        {DNNL_ARG_DST, dst_m},
                        {DNNL_ARG_ATTR_OUTPUT_SCALES, scales_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                                zero_points_src_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                                zero_points_weights_m},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
                                zero_points_dst_m},
                });
        strm.wait();
    }
};

/********************************* TEST CASES *********************************/

using iface = matmul_iface_test_t;

using data_type = memory::data_type;

TEST_P(iface, TestsMatMul) {}

static auto cases_ef = []() {
    std::vector<matmul_test_params_t> cases;

    // inconsistent dims
    cases.push_back(
            {{{{10, 1}, data_type::f32, tag::ab},
                     {{2, 20}, data_type::f32, tag::ab},
                     {{10, 20}, data_type::f32, tag::ab}, data_type::undef},
                    {}, true, dnnl_invalid_arguments});
    cases.push_back({{{{10, 1}, data_type::f32, tag::ab},
                             {{1, 20}, data_type::f32, tag::ab},
                             {{10, 21}, data_type::f32, tag::ab}},
            {}, true, dnnl_invalid_arguments});
    cases.push_back({{{{10, 1}, data_type::f32, tag::ab},
                             {{1, 1, 20}, data_type::f32, tag::abc},
                             {{10, 20}, data_type::f32, tag::ab}},
            {}, true, dnnl_invalid_arguments});
    cases.push_back({{{{1, 10, 1}, data_type::u8, tag::abc},
                             {{1, 1, 2}, data_type::s8, tag::abc},
                             {{1, 11, 2}, data_type::s8, tag::abc}},
            {}, true, dnnl_invalid_arguments});

    // inconsistent wrt runtime dim vals
    cases.push_back(
            {{{{3, 10, 10}, data_type::f32, tag::abc},
                     {{DNNL_RUNTIME_DIM_VAL, 10, 10}, data_type::f32, tag::abc},
                     {{DNNL_RUNTIME_DIM_VAL, 10, 10}, data_type::f32,
                             tag::abc}},
                    {}, true, dnnl_invalid_arguments});

    // inconsistent wrt broadcasting
    cases.push_back({{{{3, 10, 10}, data_type::f32, tag::abc},
                             {{1, 10, 10}, data_type::f32, tag::abc},
                             {{1, 10, 10}, data_type::f32, tag::abc}},
            {}, true, dnnl_invalid_arguments});

    // no broadcasting on m/k/n dims
    cases.push_back({{{{10, 10}, data_type::f32, tag::ab},
                             {{1, 1}, data_type::f32, tag::ab},
                             {{10, 10}, data_type::f32, tag::ab}},
            {}, true, dnnl_invalid_arguments});

    // f32 data and zero-points
    cases.push_back({{{{10, 1}, data_type::f32, tag::ab},
                             {{1, 20}, data_type::f32, tag::ab},
                             {{10, 20}, data_type::f32, tag::ab}},
            {P::NONE, {P::ZERO_POINTS | P::SRC | P::COMMON}}, true,
            dnnl_unimplemented});

    // unimplemented data types
    cases.push_back(
            {{{{10, 1}, data_type::f32, tag::ab},
                     {{1, 20}, data_type::f32, tag::ab},
                     {{10, 20}, data_type::f32, tag::ab}, data_type::u8},
                    {}, true, dnnl_unimplemented});
    // XXX: disable assert in type_helpers.hpp: default_accum_data_type(...)
    //cases.push_back({{{{10, 1}, data_type::u8, tag::ab}, {{1, 20}, data_type::u8, tag::ab},
    //                           {{10, 20}, data_type::u8, tag::ab}},
    //        {}, true, dnnl_unimplemented});

    return ::testing::ValuesIn(cases);
};
INSTANTIATE_TEST_SUITE_P(EF, iface, cases_ef());

static auto cases_f = [](memory::data_type dt) {
    std::vector<matmul_test_params_t> cases;

    // simple case
    cases.push_back({{{{10, 2}, dt, tag::ab}, {{2, 20}, dt, tag::ab},
                             {{10, 20}, dt, tag::ab}, data_type::undef},
            {}});
    // simple case + leading dimensions
    cases.push_back({{{{10, 1}, dt, tag::ab, P::SRC | P::LEADING_DIM},
                             {{1, 3}, dt, tag::ba},
                             {{10, 3}, dt, tag::ab, P::DST | P::LEADING_DIM},
                             data_type::f32},
            {}});
    // simple case + leading dimensions + runtime dims
    cases.push_back(
            {{{{1, 10}, dt, tag::ab, P::SRC | P::LEADING_DIM | P::RUNTIME},
                     {{10, 2}, dt, tag::ba, P::WEIGHTS | P::RUNTIME},
                     {{1, 2}, dt, tag::ab,
                             P::DST | P::LEADING_DIM | P::RUNTIME},
                     data_type::f32},
                    {}});

    // output scales
    cases.push_back({{{{10, 2}, dt, tag::ab}, {{2, 20}, dt, tag::ab},
                             {{10, 20}, dt, tag::ab}, data_type::undef},
            {P::SCALES | P::COMMON}});
    // output scales + per_n + runtime
    cases.push_back({{{{10, 2}, dt, tag::ab}, {{2, 20}, dt, tag::ab},
                             {{10, 20}, dt, tag::ab}, data_type::undef},
            {P::SCALES | P::PER_N | P::RUNTIME}});

    // post-ops
    cases.push_back({{{{10, 1}, dt, tag::ab}, {{1, 20}, dt, tag::ab},
                             {{10, 20}, dt, tag::ab}},
            {P::NONE, {},
                    {{primitive::kind::eltwise, algorithm::eltwise_relu}}}});
    // multiple post-ops
    cases.push_back({{{{10, 2}, dt, tag::ab}, {{2, 20}, dt, tag::ab},
                             {{10, 20}, dt, tag::ab}},
            {P::SCALES | P::COMMON, {},
                    {{primitive::kind::sum},
                            {primitive::kind::eltwise,
                                    algorithm::eltwise_relu}}}});

    // gemm like: output scale + post-ops(sum)
    cases.push_back({{{{10, 1}, dt, tag::ab}, {{1, 20}, dt, tag::ab},
                             {{10, 20}, dt, tag::ab}, data_type::f32},
            {P::SCALES | P::COMMON, {}, {{primitive::kind::sum}}}});
    // gemm like: output scale + post-ops(sum) + all runtime
    cases.push_back({{{{10, 1}, dt, tag::ab, P::SRC | P::RUNTIME},
                             {{1, 20}, dt, tag::ab, P::WEIGHTS | P::RUNTIME},
                             {{10, 20}, dt, tag::ab, P::DST | P::RUNTIME},
                             data_type::f32},
            {P::SCALES | P::COMMON | P::RUNTIME, {},
                    {{primitive::kind::sum}}}});

    return ::testing::ValuesIn(cases);
};

GPU_INSTANTIATE_TEST_SUITE_P(Generic_f16, iface, cases_f(data_type::f16));
GPU_INSTANTIATE_TEST_SUITE_P(Generic_bf16, iface, cases_f(data_type::bf16));
INSTANTIATE_TEST_SUITE_P(Generic_f32, iface, cases_f(data_type::f32));

static auto cases_x8 = [](memory::data_type src_dt, memory::data_type dst_dt) {
    std::vector<matmul_test_params_t> cases;

    // simple case
    cases.push_back(
            {{{{10, 2}, src_dt, tag::ba}, {{2, 20}, data_type::s8, tag::ab},
                     {{10, 20}, dst_dt, tag::ab}, data_type::undef},
                    {}});
    // simple case + leading dimensions
    cases.push_back(
            {{{{10, 1}, src_dt, tag::ba, P::SRC | P::LEADING_DIM},
                     {{1, 3}, data_type::s8, tag::ba},
                     {{10, 3}, dst_dt, tag::ab, P::DST | P::LEADING_DIM},
                     data_type::s8},
                    {}});
    // simple case + leading dimensions + runtime dims
    cases.push_back(
            {{{{1, 10}, src_dt, tag::ba, P::SRC | P::LEADING_DIM | P::RUNTIME},
                     {{10, 2}, data_type::s8, tag::ba, P::WEIGHTS | P::RUNTIME},
                     {{1, 2}, dst_dt, tag::ab,
                             P::DST | P::LEADING_DIM | P::RUNTIME},
                     data_type::u8},
                    {}});

    // output scales
    cases.push_back(
            {{{{10, 2}, src_dt, tag::ab}, {{2, 20}, data_type::s8, tag::ab},
                     {{10, 20}, dst_dt, tag::ab}, data_type::undef},
                    {P::SCALES | P::COMMON}});
    // output scales + per_n + runtime
    cases.push_back(
            {{{{10, 2}, src_dt, tag::ab}, {{2, 20}, data_type::s8, tag::ab},
                     {{10, 20}, dst_dt, tag::ab}, data_type::undef},
                    {P::SCALES | P::PER_N | P::RUNTIME}});

    // zero points
    cases.push_back(
            {{{{10, 2}, src_dt, tag::ba}, {{2, 20}, data_type::s8, tag::ab},
                     {{10, 20}, dst_dt, tag::ab}, data_type::f32},
                    {P::SCALES | P::COMMON,
                            {P::ZERO_POINTS | P::SRC | P::COMMON,
                                    P::ZERO_POINTS | P::WEIGHTS | P::COMMON,
                                    P::ZERO_POINTS | P::DST | P::COMMON}}});

    // zero points + runtime
    cases.push_back(
            {{{{10, 2}, src_dt, tag::ba}, {{2, 20}, data_type::s8, tag::ab},
                     {{10, 20}, dst_dt, tag::ab}, data_type::f32},
                    {P::SCALES | P::COMMON | P::RUNTIME,
                            {P::ZERO_POINTS | P::SRC | P::COMMON, P::NONE,
                                    P::ZERO_POINTS | P::DST | P::COMMON
                                            | P::RUNTIME}}});

    // post-ops
    cases.push_back({{{{10, 1}, src_dt, tag::ab},
                             {{1, 20}, data_type::s8, tag::ab},
                             {{10, 20}, dst_dt, tag::ab}},
            {P::NONE, {},
                    {{primitive::kind::eltwise, algorithm::eltwise_relu}}}});
    // multiple post-ops
    cases.push_back(
            {{{{10, 2}, src_dt, tag::ab}, {{2, 20}, data_type::s8, tag::ab},
                     {{10, 20}, dst_dt, tag::ab}, data_type::f32},
                    {P::SCALES | P::COMMON, {},
                            {{primitive::kind::sum},
                                    {primitive::kind::eltwise,
                                            algorithm::eltwise_relu}}}});

    // igemm like: output scale + post-ops(sum)
    cases.push_back({{{{10, 1}, src_dt, tag::ab},
                             {{1, 20}, data_type::s8, tag::ab},
                             {{10, 20}, dst_dt, tag::ab}, data_type::s8},
            {P::SCALES | P::COMMON,
                    {P::ZERO_POINTS | P::SRC | P::COMMON, P::NONE,
                            P::ZERO_POINTS | P::DST | P::COMMON | P::RUNTIME},
                    {{primitive::kind::sum}}}});
    // igemm like: output scale + post-ops(sum) + all runtime
    cases.push_back({{{{10, 2}, src_dt, tag::ba},
                             {{2, 20}, data_type::s8, tag::ba},
                             {{10, 20}, dst_dt, tag::ab}, data_type::s8},
            {P::SCALES | P::PER_N | P::RUNTIME,
                    {P::ZERO_POINTS | P::SRC | P::COMMON | P::RUNTIME,
                            P::ZERO_POINTS | P::WEIGHTS | P::COMMON
                                    | P::RUNTIME,
                            P::ZERO_POINTS | P::DST | P::COMMON | P::RUNTIME},
                    {{primitive::kind::sum}}}});

    return ::testing::ValuesIn(cases);
};
INSTANTIATE_TEST_SUITE_P(
        Generic_s8s8s32, iface, cases_x8(data_type::s8, data_type::s32));
INSTANTIATE_TEST_SUITE_P(
        Generic_u8s8u8, iface, cases_x8(data_type::u8, data_type::u8));

} // namespace dnnl
