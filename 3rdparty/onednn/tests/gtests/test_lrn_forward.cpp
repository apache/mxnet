/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

struct test_lrn_desc_t {
    memory::dim mb, c;
    memory::dim h, w;
    memory::dim local_size;
    float alpha, beta, k;
};

struct lrn_params_t {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    memory::format_tag format;
    test_lrn_desc_t test_ld;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename data_t, typename acc_data_t = data_t>
void check_lrn_fwd(const lrn_params_t &p, const memory::desc &src_d,
        const memory::desc &dst_d, const memory &src, const memory &dst) {
    auto src_ptr = map_memory<data_t>(src);
    auto dst_ptr = map_memory<data_t>(dst);

    const test_lrn_desc_t &ld = p.test_ld;

    const memory::dim C = ld.c;
    const memory::dim H = ld.h;
    const memory::dim W = ld.w;
    const memory::dim size = ld.local_size;
    const memory::dim CSIZE
            = p.aalgorithm == algorithm::lrn_across_channels ? size : 1;
    const memory::dim HWSIZE = size + 1 - CSIZE;
    const memory::dim summands = p.aalgorithm == algorithm::lrn_across_channels
            ? size
            : size * size;
    const auto padded_c = src.get_desc().data.padded_dims[1];

    const dnnl::impl::memory_desc_wrapper src_mdw(src_d.data);
    const dnnl::impl::memory_desc_wrapper dst_mdw(dst_d.data);

    auto off = [=](memory::dim n, memory::dim c, memory::dim h, memory::dim w) {
        return ((n * padded_c + c) * ld.h + h) * ld.w + w;
    };

    auto ker = [&](data_t *d, memory::dim n, memory::dim oc, memory::dim oh,
                       memory::dim ow) {
        acc_data_t sum = 0.0;
        for (memory::dim c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (memory::dim h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (memory::dim w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    acc_data_t s = src_ptr[src_mdw.off_l(
                            off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2,
                                    w - (HWSIZE - 1) / 2),
                            true)];
                    sum += s * s;
                }
            }
        }
        acc_data_t norm_coef
                = powf(static_cast<float>(ld.k + ld.alpha * sum / summands),
                        static_cast<float>(ld.beta));
        acc_data_t ref_out
                = src_ptr[src_mdw.off_l(off(n, oc, oh, ow), true)] / norm_coef;
        acc_data_t eps = static_cast<acc_data_t>(1.e-7f * (2 * summands + 5));

        memory::data_type data_type = data_traits<data_t>::data_type;
        if (data_type == dnnl::memory::data_type::f16)
            eps = static_cast<acc_data_t>(1.e-4f * 2 * summands);
        else if (data_type == dnnl::memory::data_type::bf16)
            eps = static_cast<acc_data_t>(1.e-3f * 2 * summands);

        acc_data_t out = d[0];
        acc_data_t norm_max = (std::max)(fabs(out), fabs(ref_out));
        if (norm_max < eps) norm_max = 1.;
        ASSERT_NEAR(out, ref_out, eps * norm_max);
    };

    const memory::dim N = ld.mb;
    dnnl::impl::parallel_nd(N, padded_c, H, W,
            [&](memory::dim n, memory::dim c, memory::dim h, memory::dim w) {
                ker(&dst_ptr[dst_mdw.off_l(off(n, c, h, w), true)], n, c, h, w);
            });
}

template <typename data_t>
class lrn_forward_test_t : public ::testing::TestWithParam<lrn_params_t> {
    lrn_params_t p;

protected:
    void SetUp() override {
        memory::data_type data_type = data_traits<data_t>::data_type;
        SKIP_IF(unsupported_data_type(data_type),
                "Engine does not support this data type.");
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Test() {
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        memory::data_type data_type = data_traits<data_t>::data_type;

        test_lrn_desc_t ld = p.test_ld;
        bool with_workspace = p.aprop_kind == prop_kind::forward_training;

        auto l_src_desc
                = create_md({ld.mb, ld.c, ld.h, ld.w}, data_type, p.format);
        auto l_dst_desc
                = create_md({ld.mb, ld.c, ld.h, ld.w}, data_type, p.format);

        auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm,
                l_src_desc, ld.local_size, ld.alpha, ld.beta, ld.k);
        auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, eng);
        // test construction from a C pd
        lrn_prim_desc = lrn_forward::primitive_desc(lrn_prim_desc.get());

        ASSERT_TRUE(lrn_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_SRC)
                == lrn_prim_desc.src_desc());
        ASSERT_TRUE(lrn_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_DST)
                == lrn_prim_desc.dst_desc());
        ASSERT_TRUE(
                lrn_prim_desc.query_md(query::exec_arg_md, DNNL_ARG_WORKSPACE)
                == lrn_prim_desc.workspace_desc());

        auto l_src = test_memory(l_src_desc, eng);
        auto l_dst = test_memory(l_dst_desc, eng);

        // Only true for dense format
        fill_data<data_t>(l_src.get_size() / sizeof(data_t), l_src.get());
        fill_data<data_t>(l_dst.get_size() / sizeof(data_t), l_dst.get());
        check_zero_tail<data_t>(1, l_src.get());
        check_zero_tail<data_t>(1, l_dst.get());

        memory workspace;

        // Execute
        auto l = lrn_forward(lrn_prim_desc);
        std::unordered_map<int, memory> args
                = {{DNNL_ARG_SRC, l_src.get()}, {DNNL_ARG_DST, l_dst.get()}};
        if (with_workspace) {
            auto workspace_md = lrn_prim_desc.workspace_desc();
            workspace = test::make_memory(workspace_md, eng);
            args.insert({DNNL_ARG_WORKSPACE, workspace});
        }
        l.execute(strm, args);
        strm.wait();

        check_zero_tail<data_t>(0, l_dst.get());

        if (data_type == dnnl::memory::data_type::bf16)
            check_lrn_fwd<data_t, float>(
                    p, l_src_desc, l_dst_desc, l_src.get(), l_dst.get());
        else
            check_lrn_fwd<data_t>(
                    p, l_src_desc, l_dst_desc, l_src.get(), l_dst.get());
    }
};

using lrn_f32 = lrn_forward_test_t<float>;
using lrn_bf16 = lrn_forward_test_t<bfloat16_t>;
using lrn_fp16 = lrn_forward_test_t<float16_t>;

using fmt = memory::format_tag;
const prop_kind fwd_training = prop_kind::forward_training;
const prop_kind fwd_scoring = prop_kind::forward_scoring;
const algorithm across = algorithm::lrn_across_channels;
const algorithm within = algorithm::lrn_within_channel;

static auto ForwardZeroDim_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nchw,
                                     {0, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 0, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {2, 16, 0, 4, 5, 1.0e-4f, 0.75f, 3.0f}});
};

static auto ForwardEF_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nchw,
                                     {-1, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f},
                                     true, dnnl_invalid_arguments},
            lrn_params_t {fwd_scoring, across, fmt::nchw,
                    {2, -10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}, true,
                    dnnl_invalid_arguments},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {2, 10, -4, 4, 5, 1.0e-4f, 0.75f, 3.0f}, true,
                    dnnl_invalid_arguments});
};

static auto Forward_nChw16c_padded_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw16c,
                                     {2, 17, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 19, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw16c,
                    {2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 12, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}});
};

static auto Forward_nChw8c_padded_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw8c,
                                     {2, 7, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 9, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw8c,
                    {2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 12, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}});
};

static auto Forward_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nchw,
                                     {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nchw,
                    {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 3.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nchw,
                    {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 3.0f}});
};

static auto ForwardNHWC_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nhwc,
                                     {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nhwc,
                    {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nhwc,
                    {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 4.85f}},
            lrn_params_t {fwd_scoring, across, fmt::nhwc,
                    {2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 4.85f}});
};

static auto Forward_nChw8c_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw8c,
                                     {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw8c,
                    {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}});
};

static auto Forward_nChw16c_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw16c,
                                     {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw16c,
                    {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f}});
};

static auto AlexnetForwardNCHW_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nchw,
                                     {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nchw,
                    {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nchw,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto AlexnetForwardNHWC_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nhwc,
                                     {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nhwc,
                    {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nhwc,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nhwc,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto AlexnetForward_nChw8c_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw8c,
                                     {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw8c,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto AlexnetForward_nChw16c_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw16c,
                                     {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw16c,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto GoogleNetV1ForwardNCHW_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nchw,
                                     {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nchw,
                    {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nchw,
                    {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto GoogleNetV1Forward_nChw8c_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw8c,
                                     {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw8c,
                    {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw8c,
                    {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto GoogleNetV1Forward_nChw16c_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nChw16c,
                                     {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nChw16c,
                    {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, across, fmt::nChw16c,
                    {2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto RCNNForwardBlocked_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, within, fmt::nChw8c,
                                     {2, 96, 55, 55, 3, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, within, fmt::nChw8c,
                    {2, 96, 55, 55, 3, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, within, fmt::nChw8c,
                    {2, 256, 27, 27, 3, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, within, fmt::nChw8c,
                    {2, 256, 27, 27, 3, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, within, fmt::nChw8c,
                    {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, within, fmt::nChw8c,
                    {2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, within, fmt::nChw8c,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_scoring, within, fmt::nChw8c,
                    {2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f}});
};

// This tests compatibility with Intel MKL-DNN v0.14
static auto RegressionWeightFormat_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::oihw,
            {2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f}});
};

static auto ForwardNCHWTail_cases = []() {
    return ::testing::Values(lrn_params_t {fwd_training, across, fmt::nchw,
                                     {1, 64, 1, 9, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {1, 64, 2, 9, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {1, 64, 3, 9, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {1, 64, 4, 9, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {1, 64, 5, 9, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {1, 64, 9, 6, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, across, fmt::nchw,
                    {1, 64, 7, 9, 5, 1.0e-4f, 0.75f, 1.0f}},
            lrn_params_t {fwd_training, algorithm::lrn_across_channels,
                    fmt::nchw, {1, 64, 8, 9, 5, 1.0e-4f, 0.75f, 1.0f}});
};

// ------------- f32 ----------------------
TEST_P(lrn_f32, TestsLRN) {}
INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardZeroDim, lrn_f32, ForwardZeroDim_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForwardEF, lrn_f32, ForwardEF_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw16c_padded, lrn_f32, Forward_nChw16c_padded_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw8c_padded, lrn_f32, Forward_nChw8c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward, lrn_f32, Forward_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForwardNHWC, lrn_f32, ForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw8c, lrn_f32, Forward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw16c, lrn_f32, Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForwardNCHW, lrn_f32, AlexnetForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForwardNHWC, lrn_f32, AlexnetForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForward_nChw8c, lrn_f32, AlexnetForward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForward_nChw16c, lrn_f32, AlexnetForward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNGoogleNetV1ForwardNCHW, lrn_f32, GoogleNetV1ForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw8c, lrn_f32,
        GoogleNetV1Forward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw16c, lrn_f32,
        GoogleNetV1Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNRCNNForwardBlocked, lrn_f32, RCNNForwardBlocked_cases());
// This tests compatibility with Intel MKL-DNN v0.14
INSTANTIATE_TEST_SUITE_P(
        TestLRNRegressionWeightFormat, lrn_f32, RegressionWeightFormat_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardNCHWTail, lrn_f32, ForwardNCHWTail_cases());

// ------------- bf16 ----------------------
TEST_P(lrn_bf16, TestsLRN) {}
INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardZeroDim, lrn_bf16, ForwardZeroDim_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForwardEF, lrn_bf16, ForwardEF_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw16c_padded, lrn_bf16,
        Forward_nChw16c_padded_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw8c_padded, lrn_bf16, Forward_nChw8c_padded_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForward, lrn_bf16, Forward_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNForwardNHWC, lrn_bf16, ForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw8c, lrn_bf16, Forward_nChw8c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw16c, lrn_bf16, Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForwardNCHW, lrn_bf16, AlexnetForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForwardNHWC, lrn_bf16, AlexnetForwardNHWC_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForward_nChw16c, lrn_bf16,
        AlexnetForward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1ForwardNCHW, lrn_bf16,
        GoogleNetV1ForwardNCHW_cases());
INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw16c, lrn_bf16,
        GoogleNetV1Forward_nChw16c_cases());
INSTANTIATE_TEST_SUITE_P(
        TestLRNRCNNForwardBlocked, lrn_bf16, RCNNForwardBlocked_cases());

// ------------- fp16 ----------------------
TEST_P(lrn_fp16, TestsLRN) {}
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardZeroDim, lrn_fp16, ForwardZeroDim_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNForwardEF, lrn_fp16, ForwardEF_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNForward_nChw16c_padded, lrn_fp16,
        Forward_nChw16c_padded_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw8c_padded, lrn_fp16, Forward_nChw8c_padded_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNForward, lrn_fp16, Forward_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNForwardNHWC, lrn_fp16, ForwardNHWC_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw8c, lrn_fp16, Forward_nChw8c_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNForward_nChw16c, lrn_fp16, Forward_nChw16c_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForwardNCHW, lrn_fp16, AlexnetForwardNCHW_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForwardNHWC, lrn_fp16, AlexnetForwardNHWC_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNAlexnetForward_nChw8c, lrn_fp16, AlexnetForward_nChw8c_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNAlexnetForward_nChw16c, lrn_fp16,
        AlexnetForward_nChw16c_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1ForwardNCHW, lrn_fp16,
        GoogleNetV1ForwardNCHW_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw8c, lrn_fp16,
        GoogleNetV1Forward_nChw8c_cases());
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNGoogleNetV1Forward_nChw16c, lrn_fp16,
        GoogleNetV1Forward_nChw16c_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNRCNNForwardBlocked, lrn_fp16, RCNNForwardBlocked_cases());
// This tests compatibility with Intel MKL-DNN v0.14
GPU_INSTANTIATE_TEST_SUITE_P(TestLRNRegressionWeightFormat, lrn_fp16,
        RegressionWeightFormat_cases());
GPU_INSTANTIATE_TEST_SUITE_P(
        TestLRNForwardNCHWTail, lrn_fp16, ForwardNCHWTail_cases());

} // namespace dnnl
