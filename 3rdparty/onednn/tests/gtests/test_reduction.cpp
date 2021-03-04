/*******************************************************************************
* Copyright 2020 Intel Corporation
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

struct reduction_test_params_t {
    memory::format_tag src_format;
    memory::format_tag dst_format;
    algorithm aalgorithm;
    float p;
    float eps;
    memory::dims src_dims;
    memory::dims dst_dims;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename src_data_t, typename dst_data_t = src_data_t>
class reduction_test_t
    : public ::testing::TestWithParam<reduction_test_params_t> {
private:
    reduction_test_params_t p;
    memory::data_type src_dt, dst_dt;

protected:
    void SetUp() override {
        src_dt = data_traits<src_data_t>::data_type;
        dst_dt = data_traits<dst_data_t>::data_type;

        p = ::testing::TestWithParam<reduction_test_params_t>::GetParam();

        SKIP_IF(unsupported_data_type(src_dt),
                "Engine does not support this data type.");
        SKIP_IF(get_test_engine().get_kind() != engine::kind::cpu,
                "Engine does not support this primitive.");

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        // reduction specific types and values
        using op_desc_t = reduction::desc;
        using pd_t = reduction::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        auto desc_src = memory::desc(p.src_dims, src_dt, p.src_format);
        auto desc_dst = memory::desc(p.dst_dims, dst_dt, p.dst_format);

        // default op desc ctor
        auto op_desc = op_desc_t();
        // regular op desc ctor
        op_desc = op_desc_t(p.aalgorithm, desc_src, desc_dst, p.p, p.eps);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        ASSERT_NO_THROW(pd = pd_t(op_desc, eng));
        // test all pd ctors
        test_fwd_pd_constructors<op_desc_t, pd_t>(op_desc, pd, aa);

        // default primitive ctor
        auto prim = reduction();
        // regular primitive ctor
        prim = reduction(pd);

        const auto src_desc = pd.src_desc();
        const auto dst_desc = pd.dst_desc();

        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == src_desc);
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == dst_desc);

        const auto test_engine = pd.get_engine();

        auto mem_src = memory(src_desc, test_engine);
        auto mem_dst = memory(dst_desc, test_engine);

        fill_data<src_data_t>(
                src_desc.get_size() / sizeof(src_data_t), mem_src);

        prim.execute(strm, {{DNNL_ARG_SRC, mem_src}, {DNNL_ARG_DST, mem_dst}});
        strm.wait();
    }
};

using tag = memory::format_tag;

static auto expected_failures = []() {
    return ::testing::Values(
            // The same src and dst dims
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_sum, 0.0f, 0.0f, {1, 1, 1, 4},
                    {1, 1, 1, 4}, true, dnnl_invalid_arguments},
            // not supported alg_kind
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::eltwise_relu, 0.0f, 0.0f, {1, 1, 1, 4},
                    {1, 1, 1, 4}, true, dnnl_invalid_arguments},
            // negative dim
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_sum, 0.0f, 0.0f, {-1, 1, 1, 4},
                    {-1, 1, 1, 1}, true, dnnl_invalid_arguments},
            // not supported p
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_norm_lp_max, 0.5f, 0.0f, {1, 8, 4, 4},
                    {1, 8, 4, 4}, true, dnnl_invalid_arguments});
};

static auto zero_dim = []() {
    return ::testing::Values(reduction_test_params_t {tag::nchw, tag::nchw,
            algorithm::reduction_sum, 0.0f, 0.0f, {0, 1, 1, 4}, {0, 1, 1, 1}});
};

static auto simple_cases = []() {
    return ::testing::Values(reduction_test_params_t {tag::nchw, tag::nchw,
                                     algorithm::reduction_sum, 0.0f, 0.0f,
                                     {1, 1, 1, 4}, {1, 1, 1, 1}},
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_max, 0.0f, 0.0f, {1, 1, 4, 4},
                    {1, 1, 1, 4}},
            reduction_test_params_t {tag::nChw16c, tag::nChw16c,
                    algorithm::reduction_min, 0.0f, 0.0f, {4, 4, 4, 4},
                    {1, 4, 4, 4}},
            reduction_test_params_t {tag::nChw16c, tag::nchw,
                    algorithm::reduction_sum, 0.0f, 0.0f, {4, 4, 4, 4},
                    {1, 4, 4, 1}},
            reduction_test_params_t {tag::nChw16c, tag::any,
                    algorithm::reduction_min, 0.0f, 0.0f, {4, 4, 4, 4},
                    {1, 1, 1, 1}});
};

static auto f32_cases = []() {
    return ::testing::Values(reduction_test_params_t {tag::nchw, tag::nchw,
                                     algorithm::reduction_norm_lp_max, 1.0f,
                                     0.0f, {1, 1, 1, 4}, {1, 1, 1, 1}},
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_norm_lp_power_p_max, 2.0f, 0.0f,
                    {1, 1, 1, 4}, {1, 1, 1, 1}},
            reduction_test_params_t {tag::nchw, tag::nchw,
                    algorithm::reduction_mean, 0.0f, 0.0f, {1, 4, 4, 4},
                    {1, 1, 4, 4}});
};

#define INST_TEST_CASE(test) \
    TEST_P(test, TestsReduction) {} \
    INSTANTIATE_TEST_SUITE_P(TestReductionEF, test, expected_failures()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionZero, test, zero_dim()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionSimple, test, simple_cases());

#define INST_TEST_CASE_F32(test) \
    TEST_P(test, TestsReduction) {} \
    INSTANTIATE_TEST_SUITE_P(TestReductionEF, test, expected_failures()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionZero, test, zero_dim()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionSimple, test, simple_cases()); \
    INSTANTIATE_TEST_SUITE_P(TestReductionNorm, test, f32_cases());

using reduction_test_f32 = reduction_test_t<float>;
using reduction_test_bf16 = reduction_test_t<bfloat16_t>;
using reduction_test_f16 = reduction_test_t<float16_t>;
using reduction_test_s8 = reduction_test_t<int8_t>;
using reduction_test_u8 = reduction_test_t<uint8_t>;

INST_TEST_CASE_F32(reduction_test_f32)
INST_TEST_CASE(reduction_test_bf16)
INST_TEST_CASE(reduction_test_f16)
INST_TEST_CASE(reduction_test_s8)
INST_TEST_CASE(reduction_test_u8)

} // namespace dnnl
