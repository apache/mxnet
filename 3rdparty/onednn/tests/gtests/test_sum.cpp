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

using tag = memory::format_tag;

/* iface tests */

class iface_sum_test_t : public ::testing::Test {
protected:
    engine eng;
    stream strm;

    void SetUp() override {
        eng = get_test_engine();
        strm = make_stream(eng);
    }
};

TEST_F(iface_sum_test_t, SumTestDstDataTypeCompliance) {
    using dt = memory::data_type;

    const dt src_dt = dt::s8;

    memory::dims shape = {10, 10, 10, 10};
    auto src_md = memory::desc(shape, src_dt, tag::abcd);

    for_(tag dst_tag : {tag::any, tag::abcd, tag::acdb})
    for (dt dst_dt : {dt::undef, dt::s8, dt::s32, dt::f32}) {
        sum::primitive_desc sum_pd;

        if (dst_dt != dt::undef) {
            memory::desc dst_md(shape, dst_dt, dst_tag);
            sum_pd = sum::primitive_desc(
                    dst_md, {2., 2.}, {src_md, src_md}, eng);
        } else {
            sum_pd = sum::primitive_desc({2., 2.}, {src_md, src_md}, eng);
        }

        dt expect_dst_dt = dst_dt == dt::undef ? src_dt : dst_dt;
        ASSERT_EQ(sum_pd.dst_desc().data.data_type, expect_dst_dt);
    }
}

/* correctness tests */

struct sum_test_params {
    std::vector<tag> srcs_format;
    tag dst_format;
    memory::dims dims;
    std::vector<float> scale;
    bool is_output_omitted;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename src_data_t, typename acc_t, typename dst_data_t = src_data_t>
class sum_test_t : public ::testing::TestWithParam<sum_test_params> {
private:
    memory::data_type src_data_type;
    memory::data_type dst_data_type;

    void check_data(const std::vector<memory> &srcs,
            const std::vector<float> &scale, const memory &dst) {
        auto dst_data = map_memory<const dst_data_t>(dst);
        const auto &dst_d = dst.get_desc();
        const auto dst_dims = dst_d.data.dims;
        const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);

        std::vector<mapped_ptr_t<const src_data_t>> mapped_srcs;
        mapped_srcs.reserve(srcs.size());
        for (auto &src : srcs)
            mapped_srcs.emplace_back(map_memory<const src_data_t>(src));

        dnnl::impl::parallel_nd(dst_dims[0], dst_dims[1], dst_dims[2],
                dst_dims[3],
                [&](memory::dim n, memory::dim c, memory::dim h,
                        memory::dim w) {
                    if (is_current_test_failed()) return;

                    acc_t src_sum = 0.0;
                    for (size_t num = 0; num < srcs.size(); num++) {
                        auto &src_data = mapped_srcs[num];
                        const auto &src_d = srcs[num].get_desc();
                        const auto src_dims = src_d.data.dims;
                        const dnnl::impl::memory_desc_wrapper src_mdw(
                                src_d.data);

                        auto src_idx = w + src_dims[3] * h
                                + src_dims[2] * src_dims[3] * c
                                + src_dims[1] * src_dims[2] * src_dims[3] * n;
                        if (num == 0) {
                            src_sum = acc_t(scale[num])
                                    * src_data[src_mdw.off_l(src_idx, false)];
                        } else {
                            src_sum += acc_t(scale[num])
                                    * src_data[src_mdw.off_l(src_idx, false)];
                        }

                        src_sum = (std::max)(
                                (std::min)(src_sum,
                                        (std::numeric_limits<acc_t>::max)()),
                                std::numeric_limits<acc_t>::lowest());
                    }

                    auto dst_idx = w + dst_dims[3] * h
                            + dst_dims[2] * dst_dims[3] * c
                            + dst_dims[1] * dst_dims[2] * dst_dims[3] * n;

                    acc_t dst_val = dst_data[dst_mdw.off_l(dst_idx, false)];
                    ASSERT_EQ(src_sum, dst_val);
                });
    }

protected:
    void SetUp() override {
        src_data_type = data_traits<src_data_t>::data_type;
        dst_data_type = data_traits<dst_data_t>::data_type;
        sum_test_params p
                = ::testing::TestWithParam<sum_test_params>::GetParam();
        SKIP_IF(get_test_engine_kind() == engine::kind::gpu
                        && src_data_type == memory::data_type::bf16,
                "GPU does not support bfloat16 data type.");
        SKIP_IF(unsupported_data_type(src_data_type),
                "Engine does not support this data type.");
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        sum_test_params p
                = ::testing::TestWithParam<sum_test_params>::GetParam();

        const auto num_srcs = p.srcs_format.size();

        auto eng = get_test_engine();
        auto strm = make_stream(eng);

        std::vector<memory::desc> srcs_md;
        std::vector<memory> srcs;

        for (size_t i = 0; i < num_srcs; i++) {
            auto desc = memory::desc(p.dims, src_data_type, p.srcs_format[i]);
            auto src_memory = test::make_memory(desc, eng);
            const size_t sz
                    = src_memory.get_desc().get_size() / sizeof(src_data_t);
            fill_data<src_data_t>(sz, src_memory);

            // Keep few mantissa digits for fp types to avoid round-off errors
            // With proper scalars the computations give exact results
            if (!std::is_integral<src_data_t>::value) {
                using uint_type = typename data_traits<src_data_t>::uint_type;
                int mant_digits
                        = dnnl::impl::nstl::numeric_limits<src_data_t>::digits;
                int want_mant_digits = 3;
                auto src_ptr = map_memory<src_data_t>(src_memory);
                for (size_t i = 0; i < sz; i++) {
                    uint_type mask = (uint_type)-1
                            << (mant_digits - want_mant_digits);
                    *((uint_type *)&src_ptr[i]) &= mask;
                }
            }
            srcs_md.push_back(desc);
            srcs.push_back(src_memory);
        }

        memory dst;
        sum::primitive_desc sum_pd;

        if (p.is_output_omitted) {
            ASSERT_NO_THROW(
                    sum_pd = sum::primitive_desc(p.scale, srcs_md, eng));
        } else {
            auto dst_desc = memory::desc(p.dims, dst_data_type, p.dst_format);
            sum_pd = sum::primitive_desc(dst_desc, p.scale, srcs_md, eng);

            ASSERT_EQ(sum_pd.dst_desc().data.ndims, dst_desc.data.ndims);
        }
        dst = test::make_memory(sum_pd.dst_desc(), eng);
        // test construction from a C pd
        sum_pd = sum::primitive_desc(sum_pd.get());

        ASSERT_TRUE(sum_pd.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == sum_pd.dst_desc());
        for (int i = 0; i < (int)srcs.size(); i++)
            ASSERT_TRUE(sum_pd.query_md(
                                query::exec_arg_md, DNNL_ARG_MULTIPLE_SRC + i)
                    == sum_pd.src_desc(i));

        {
            auto dst_data = map_memory<dst_data_t>(dst);
            const size_t sz = dst.get_desc().get_size() / sizeof(dst_data_t);
            // overwriting dst to prevent false positives for test cases.
            dnnl::impl::parallel_nd(
                    (ptrdiff_t)sz, [&](ptrdiff_t i) { dst_data[i] = -32; });
        }
        sum c(sum_pd);
        std::unordered_map<int, memory> args = {{DNNL_ARG_DST, dst}};
        for (int i = 0; i < (int)num_srcs; i++) {
            args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs[i]});
        }
        c.execute(strm, args);
        strm.wait();

        check_data(srcs, p.scale, dst);
    }
};

static auto simple_test_cases = [](bool omit_output) {
    return ::testing::Values(
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw, {0, 7, 4, 4},
                    {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw, {1, 0, 4, 4},
                    {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw, {1, 8, 0, 4},
                    {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw, {-1, 8, 4, 4},
                    {1.0f, 1.0f}, omit_output, true, dnnl_invalid_arguments},

            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw,
                    {1, 1024, 38, 50}, {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw}, tag::nchw, {2, 8, 2, 2},
                    {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nChw8c, tag::nChw8c}, tag::nChw8c,
                    {2, 16, 3, 4}, {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw}, tag::nChw8c, {2, 16, 2, 2},
                    {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nChw8c, tag::nChw8c}, tag::nchw,
                    {2, 16, 3, 4}, {1.0f, 1.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw}, tag::nchw, {2, 8, 2, 2},
                    {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nChw8c, tag::nChw8c}, tag::nChw8c,
                    {2, 16, 3, 4}, {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw}, tag::nChw8c, {2, 16, 2, 2},
                    {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nChw8c, tag::nChw8c}, tag::nchw,
                    {2, 16, 3, 4}, {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw, {5, 8, 3, 3},
                    {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw,
                    {32, 32, 13, 14}, {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nChw16c, tag::nChw8c}, tag::nChw16c,
                    {2, 16, 3, 3}, {2.0f, 3.0f}, omit_output});
};

static auto simple_test_cases_bf16 = [](bool omit_output) {
    return ::testing::Values(
            sum_test_params {{tag::nChw16c, tag::nChw16c}, tag::nChw16c,
                    {1, 16, 1, 1}, {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw}, tag::nchw, {1, 16, 1, 1},
                    {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw}, tag::nchw, {2, 16, 13, 7},
                    {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw, tag::nchw, tag::nchw},
                    tag::nchw, {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f, 5.0f},
                    omit_output},
            sum_test_params {{tag::nchw, tag::nchw, tag::nchw}, tag::nchw,
                    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f}, omit_output},
            sum_test_params {
                    {tag::nchw, tag::nchw, tag::nchw, tag::nchw, tag::nchw},
                    tag::nchw, {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                    omit_output},
            sum_test_params {{tag::nchw, tag::nchw, tag::nchw}, tag::nchw,
                    {2, 37, 13, 7}, {2.0f, 3.0f, 4.0f}, omit_output},
            sum_test_params {{tag::nchw, tag::nchw, tag::nchw}, tag::nchw,
                    {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f}, omit_output},
            sum_test_params {{tag::nChw16c, tag::nChw16c}, tag::nChw16c,
                    {2, 16, 13, 7}, {2.0f, 3.0f}, omit_output},
            sum_test_params {{tag::nChw16c, tag::nChw16c, tag::nChw16c},
                    tag::nChw16c, {2, 16, 13, 7}, {2.0f, 3.0f, 4.0f},
                    omit_output},
            sum_test_params {{tag::nChw16c, tag::nChw16c, tag::nChw16c,
                                     tag::nChw16c, tag::nChw16c},
                    tag::nChw16c, {2, 16, 13, 7},
                    {2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, omit_output},
            sum_test_params {{tag::nChw16c, tag::nChw16c}, tag::nChw16c,
                    {2, 128, 23, 15}, {2.5f, 0.125f}, omit_output});
};

static auto special_test_cases = []() {
    return ::testing::Values(
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw, {1, 8, 4, 4},
                    {1.0f}, false, true, dnnl_invalid_arguments},
            sum_test_params {{tag::nchw, tag::nChw8c}, tag::nchw, {2, 8, 4, 4},
                    {0.1f}, false, true, dnnl_invalid_arguments});
};

/* corner cases */
#define CASE_CC(itag0, itag1, otag, dims_, ef, st) \
    sum_test_params { \
        {tag::itag0, tag::itag1}, tag::otag, memory::dims dims_, {1.0f, 1.0f}, \
                0, ef, st \
    }
static auto corner_test_cases = []() {
    return ::testing::Values(
            CASE_CC(nchw, nChw8c, nchw, ({0, 7, 4, 4}), false, dnnl_success),
            CASE_CC(nchw, nChw8c, nchw, ({1, 0, 4, 4}), false, dnnl_success),
            CASE_CC(nchw, nChw8c, nchw, ({1, 8, 0, 4}), false, dnnl_success),
            CASE_CC(nchw, nChw8c, nchw, ({-1, 8, 4, 4}), true,
                    dnnl_invalid_arguments));
};
#undef CASE_CC

#define CPU_INST_TEST_CASE(test, omit_output) \
    CPU_TEST_P(test, TestsSum) {} \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            TestSum, test, simple_test_cases(omit_output)); \
    CPU_INSTANTIATE_TEST_SUITE_P(TestSumEF, test, special_test_cases());

#define INST_TEST_CASE_BF16(test, omit_output) \
    CPU_TEST_P(test, TestsSum) {} \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            TestSum, test, simple_test_cases(omit_output)); \
    CPU_INSTANTIATE_TEST_SUITE_P( \
            TestSumBf16, test, simple_test_cases_bf16(omit_output)); \
    CPU_INSTANTIATE_TEST_SUITE_P(TestSumEF, test, special_test_cases());

#define GPU_INST_TEST_CASE(test, omit_output) \
    GPU_TEST_P(test, TestsSum) {} \
    GPU_INSTANTIATE_TEST_SUITE_P( \
            TestSum, test, simple_test_cases(omit_output)); \
    GPU_INSTANTIATE_TEST_SUITE_P(TestSumEF, test, special_test_cases());

#define INST_TEST_CASE(test, omit_output) \
    CPU_INST_TEST_CASE(test, omit_output) \
    GPU_INST_TEST_CASE(test, omit_output)

using sum_test_float_omit_output = sum_test_t<float, float>;
using sum_test_u8_omit_output = sum_test_t<uint8_t, int32_t>;
using sum_test_s8_omit_output = sum_test_t<int8_t, int32_t>;
using sum_test_s32_omit_output = sum_test_t<int32_t, float>;
using sum_test_f16_omit_output = sum_test_t<float16_t, float>;
using sum_test_bf16bf16_omit_output = sum_test_t<bfloat16_t, float>;
using sum_test_bf16f32_omit_output = sum_test_t<bfloat16_t, float, float>;

using sum_test_float = sum_test_t<float, float>;
using sum_test_u8 = sum_test_t<uint8_t, int32_t>;
using sum_test_s8 = sum_test_t<int8_t, int32_t>;
using sum_test_s32 = sum_test_t<int32_t, float>;
using sum_test_f16 = sum_test_t<float16_t, float>;
using sum_test_bf16bf16 = sum_test_t<bfloat16_t, float>;
using sum_test_bf16f32 = sum_test_t<bfloat16_t, float, float>;

using sum_cc_f32 = sum_test_t<float, float>;

TEST_P(sum_cc_f32, TestSumCornerCases) {}
INSTANTIATE_TEST_SUITE_P(TestSumCornerCases, sum_cc_f32, corner_test_cases());

INST_TEST_CASE(sum_test_float_omit_output, 1)
INST_TEST_CASE(sum_test_u8_omit_output, 1)
INST_TEST_CASE(sum_test_s8_omit_output, 1)
INST_TEST_CASE(sum_test_s32_omit_output, 1)
INST_TEST_CASE_BF16(sum_test_bf16bf16_omit_output, 1)
// Automatically created dst descriptor has bf16 data type so this test is not
// valid: INST_TEST_CASE(sum_test_bf16f32_omit_output, 1)
GPU_INST_TEST_CASE(sum_test_f16_omit_output, 1)

INST_TEST_CASE(sum_test_float, 0)
INST_TEST_CASE(sum_test_u8, 0)
INST_TEST_CASE(sum_test_s8, 0)
INST_TEST_CASE(sum_test_s32, 0)
INST_TEST_CASE_BF16(sum_test_bf16bf16, 0)
INST_TEST_CASE_BF16(sum_test_bf16f32, 0)
GPU_INST_TEST_CASE(sum_test_f16, 0)

#undef CPU_INST_TEST_CASE
#undef GPU_INST_TEST_CASE
} // namespace dnnl
