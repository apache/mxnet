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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using tag = memory::format_tag;

template <typename data_t>
struct softmax_test_params_t {
    prop_kind aprop_kind;
    tag memory_format;
    tag diff_memory_format;
    memory::dims dims;
    int axis;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

template <typename data_t>
class softmax_test_t
    : public ::testing::TestWithParam<softmax_test_params_t<data_t>> {
private:
    softmax_test_params_t<data_t> p;
    memory dst, workspace;
    std::shared_ptr<softmax_forward::primitive_desc> pd_fwd_hint;

protected:
    void SetUp() override {
        p = ::testing::TestWithParam<softmax_test_params_t<data_t>>::GetParam();
        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status);
    }

    void Forward() {
        // softmax specific types and values
        using op_desc_t = softmax_forward::desc;
        using pd_t = softmax_forward::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        prop_kind pk = p.aprop_kind == prop_kind::backward_data
                ? prop_kind::forward_training
                : p.aprop_kind;
        auto prec = data_traits<data_t>::data_type;
        auto mem_desc = memory::desc(p.dims, prec, p.memory_format);

        // default op desc ctor
        auto op_desc = op_desc_t();
        // regular op desc ctor
        op_desc = op_desc_t(pk, mem_desc, p.axis);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        ASSERT_NO_THROW(pd = pd_t(op_desc, eng));
        // test all pd ctors
        test_fwd_pd_constructors<op_desc_t, pd_t>(op_desc, pd, aa);
        pd_fwd_hint = std::make_shared<pd_t>(pd);

        // default primitive ctor
        auto softmax = softmax_forward();
        // regular primitive ctor
        softmax = softmax_forward(pd);

        // query for data_desc from pd via src
        const auto data_desc = pd.src_desc();
        // query for data_desc from pd via dst
        ASSERT_TRUE(pd.dst_desc() == data_desc);
        // query for data_desc via exec arg number of src
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC) == data_desc);
        // query for data_desc via exec arg number of dst
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST) == data_desc);

        // query for workspace
        const auto workspace_desc = pd.workspace_desc();

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.diff_src_desc().is_zero());
        ASSERT_TRUE(pd.diff_dst_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto src = test::make_memory(data_desc, eng);
        dst = test::make_memory(data_desc, eng);
        workspace = test::make_memory(workspace_desc, eng);

        auto test_with_given_fill = [&](data_t mean, data_t var) {
            fill_data<data_t>(
                    data_desc.get_size() / sizeof(data_t), src, mean, var);
            check_zero_tail<data_t>(1, src);

            // test out-place mode
            softmax.execute(strm,
                    {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst},
                            {DNNL_ARG_WORKSPACE, workspace}});
            strm.wait();
            check_zero_tail<data_t>(0, dst);

            // test in-place mode
            if (p.aprop_kind != prop_kind::backward_data) {
                softmax.execute(strm,
                        {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, src},
                                {DNNL_ARG_WORKSPACE, workspace}});
                strm.wait();
                check_zero_tail<data_t>(0, src);
            }
        };

        test_with_given_fill(200, 1);
    }

    void Backward() {
        // softmax specific types and values
        using op_desc_t = softmax_backward::desc;
        using pd_t = softmax_backward::primitive_desc;
        using hint_pd_t = softmax_forward::primitive_desc;
        allows_attr_t aa {false}; // doesn't support anything

        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        auto prec = data_traits<data_t>::data_type;
        auto mem_desc = memory::desc(p.dims, prec, p.memory_format);
        auto diff_mem_desc = memory::desc(p.dims, prec, p.diff_memory_format);

        // default op desc ctor
        auto op_desc = op_desc_t();
        // regular op desc ctor
        op_desc = op_desc_t(diff_mem_desc, mem_desc, p.axis);

        // default pd ctor
        auto pd = pd_t();
        // regular pd ctor
        ASSERT_NO_THROW(pd = pd_t(op_desc, eng, *pd_fwd_hint));
        // test all pd ctors
        test_bwd_pd_constructors<op_desc_t, pd_t, hint_pd_t>(
                op_desc, pd, *pd_fwd_hint, aa);

        // default primitive ctor
        auto softmax = softmax_backward();
        // regular primitive ctor
        softmax = softmax_backward(pd);

        // query for diff_data_desc from pd via diff_src
        const auto diff_data_desc = pd.diff_src_desc();
        // query for diff_data_desc from pd via diff_dst
        ASSERT_TRUE(pd.diff_dst_desc() == diff_data_desc);
        // query for diff_data_desc via exec arg number of src
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SRC)
                == diff_data_desc);
        // query for diff_data_desc via exec arg number of dst
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_DST)
                == diff_data_desc);

        // check primitive returns zero_md for all rest md
        ASSERT_TRUE(pd.src_desc().is_zero());
        ASSERT_TRUE(pd.weights_desc().is_zero());
        ASSERT_TRUE(pd.diff_weights_desc().is_zero());

        auto diff_src = test::make_memory(diff_data_desc, eng);
        auto diff_dst = test::make_memory(diff_data_desc, eng);

        auto test_with_given_fill = [&](data_t mean, data_t var) {
            // Fill the softmax backward diffs
            fill_data<data_t>(diff_data_desc.get_size() / sizeof(data_t),
                    diff_dst, data_t(0), data_t(1));
            check_zero_tail<data_t>(1, diff_dst);

            softmax.execute(strm,
                    {{DNNL_ARG_DST, dst}, {DNNL_ARG_DIFF_DST, diff_dst},
                            {DNNL_ARG_DIFF_SRC, diff_src},
                            {DNNL_ARG_WORKSPACE, workspace}});
            strm.wait();

            check_zero_tail<data_t>(0, diff_src);
        };

        test_with_given_fill(0, 1);
    }

    void Test() {
        Forward();
        if (p.aprop_kind == prop_kind::backward_data) Backward();
    }
};

using softmax_forward_test_float = softmax_test_t<float>;
using softmax_forward_test_half = softmax_test_t<float16_t>;
using softmax_forward_test_bfloat16 = softmax_test_t<bfloat16_t>;

using softmax_backward_test_float = softmax_test_t<float>;

template <typename dt>
using test_params = softmax_test_params_t<dt>;

TEST_P(softmax_forward_test_float, TestsSoftmax) {}
INSTANTIATE_TEST_SUITE_P(TestSoftmaxForwardFloat, softmax_forward_test_float,
        ::testing::Values(test_params<float> {prop_kind::forward_training,
                                  tag::nchw, tag::undef, {2, -2, 128, 256}, 0,
                                  true, dnnl_invalid_arguments},
                test_params<float> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 5, true,
                        dnnl_invalid_arguments},
                test_params<float> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 0, 5, 5}, 0},
                test_params<float> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 0, 5, 5}, 1},
                test_params<float> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 19, 16, 64}, 1},
                test_params<float> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {1, 8, 128, 1024}, 3},
                test_params<float> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {2, 1000}, 0},
                test_params<float> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {2, 1000}, 1},
                test_params<float> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {1, 13}, 1},
                test_params<float> {prop_kind::forward_inference, tag::ncw,
                        tag::undef, {16, 257, 32}, 1},
                test_params<float> {prop_kind::forward_inference, tag::ncw,
                        tag::undef, {16, 257, 32}, 2},
                test_params<float> {prop_kind::forward_inference, tag::nChw8c,
                        tag::undef, {64, 1011, 1, 1}, 1},
                test_params<float> {prop_kind::forward_inference, tag::nChw8c,
                        tag::undef, {2, 1011, 32, 1}, 2}));

TEST_P(softmax_forward_test_bfloat16, TestsSoftmax) {}
GPU_INSTANTIATE_TEST_SUITE_P(TestSoftmaxForwardBfloat16,
        softmax_forward_test_bfloat16,
        ::testing::Values(test_params<bfloat16_t> {prop_kind::forward_training,
                                  tag::nchw, tag::undef, {2, -2, 128, 256}, 0,
                                  true, dnnl_invalid_arguments},
                test_params<bfloat16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 5, true,
                        dnnl_invalid_arguments},
                test_params<bfloat16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 0, 5, 5}, 0},
                test_params<bfloat16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 0, 5, 5}, 1},
                test_params<bfloat16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 19, 16, 64}, 1},
                test_params<bfloat16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {1, 8, 128, 1024}, 3},
                test_params<bfloat16_t> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {2, 1000}, 0},
                test_params<bfloat16_t> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {2, 1000}, 1},
                test_params<bfloat16_t> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {1, 13}, 1},
                test_params<bfloat16_t> {prop_kind::forward_inference, tag::ncw,
                        tag::undef, {16, 257, 32}, 1},
                test_params<bfloat16_t> {prop_kind::forward_inference, tag::ncw,
                        tag::undef, {16, 257, 32}, 2},
                test_params<bfloat16_t> {prop_kind::forward_inference,
                        tag::nChw8c, tag::undef, {64, 1011, 1, 1}, 1},
                test_params<bfloat16_t> {prop_kind::forward_inference,
                        tag::nChw8c, tag::undef, {2, 1011, 32, 1}, 2}));

TEST_P(softmax_forward_test_half, TestsSoftmax) {}
GPU_INSTANTIATE_TEST_SUITE_P(TestSoftmaxForwardHalf, softmax_forward_test_half,
        ::testing::Values(test_params<float16_t> {prop_kind::forward_training,
                                  tag::nchw, tag::undef, {2, -2, 128, 256}, 0,
                                  true, dnnl_invalid_arguments},
                test_params<float16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 2, 128, 256}, 5, true,
                        dnnl_invalid_arguments},
                test_params<float16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 0, 5, 5}, 0},
                test_params<float16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 0, 5, 5}, 1},
                test_params<float16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {2, 19, 16, 64}, 1},
                test_params<float16_t> {prop_kind::forward_training, tag::nchw,
                        tag::undef, {1, 8, 128, 1024}, 3},
                test_params<float16_t> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {2, 1000}, 0},
                test_params<float16_t> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {2, 1000}, 1},
                test_params<float16_t> {prop_kind::forward_inference, tag::nc,
                        tag::undef, {1, 13}, 1},
                test_params<float16_t> {prop_kind::forward_inference, tag::ncw,
                        tag::undef, {16, 257, 32}, 1},
                test_params<float16_t> {prop_kind::forward_inference, tag::ncw,
                        tag::undef, {16, 257, 32}, 2},
                test_params<float16_t> {prop_kind::forward_inference,
                        tag::nChw8c, tag::undef, {64, 1011, 1, 1}, 1},
                test_params<float16_t> {prop_kind::forward_inference,
                        tag::nChw8c, tag::undef, {2, 1011, 32, 1}, 2}));

TEST_P(softmax_backward_test_float, TestsSoftmax) {}
INSTANTIATE_TEST_SUITE_P(TestSoftmaxBackward, softmax_backward_test_float,
        ::testing::Values(test_params<float> {prop_kind::backward_data,
                                  tag::nchw, tag::nchw, {2, -2, 128, 256}, 0,
                                  true, dnnl_invalid_arguments},
                test_params<float> {prop_kind::backward_data, tag::nchw,
                        tag::nchw, {2, 19, 128, 256}, 5, true,
                        dnnl_invalid_arguments},
                test_params<float> {prop_kind::backward_data, tag::nchw,
                        tag::nchw, {2, 0, 5, 5}, 0},
                test_params<float> {prop_kind::backward_data, tag::nhwc,
                        tag::nchw, {2, 0, 5, 5}, 1},
                test_params<float> {prop_kind::backward_data, tag::nchw,
                        tag::nchw, {2, 19, 16, 64}, 1},
                test_params<float> {prop_kind::backward_data, tag::nhwc,
                        tag::nchw, {1, 8, 128, 1024}, 3},
                test_params<float> {prop_kind::backward_data, tag::cn, tag::nc,
                        {2, 1000}, 0},
                test_params<float> {prop_kind::backward_data, tag::nc, tag::nc,
                        {2, 1000}, 1},
                test_params<float> {
                        prop_kind::backward_data, tag::nc, tag::cn, {1, 13}, 1},
                test_params<float> {prop_kind::backward_data, tag::ncw,
                        tag::ncw, {16, 257, 32}, 1},
                test_params<float> {prop_kind::backward_data, tag::nCw16c,
                        tag::ncw, {16, 257, 32}, 2},
                test_params<float> {prop_kind::backward_data, tag::nChw8c,
                        tag::nChw8c, {64, 1011, 1, 1}, 1},
                test_params<float> {prop_kind::backward_data, tag::nchw,
                        tag::nChw8c, {2, 1011, 32, 1}, 2}));
} // namespace dnnl
