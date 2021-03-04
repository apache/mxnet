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

// short names for brevity
using data_type = memory::data_type;
using tag = memory::format_tag;

class runtime_dim_test_t : public ::testing::Test {
protected:
    engine eng = get_test_engine();
    void SetUp() override {}

    template <typename F>
    void check_status(const F &f, dnnl_status_t status) {
        catch_expected_failures(f, status != dnnl_success, status, false);
    }
};
#define CHECK_STATUs(status, ...) check_status([&]() { __VA_ARGS__; }, status)
#define CHECK_STATUS(status, ...) CHECK_STATUs(status, __VA_ARGS__)

#define CHECK_OK(...) CHECK_STATUS(dnnl_success, __VA_ARGS__)
#define CHECK_INVALID(...) CHECK_STATUS(dnnl_invalid_arguments, __VA_ARGS__)
#define CHECK_UNIMPL(...) CHECK_STATUS(dnnl_unimplemented, __VA_ARGS__)

TEST_F(runtime_dim_test_t, TestMemory) {
    memory::desc md_tag {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL},
            data_type::f32, tag::ab};
    ASSERT_EQ(md_tag.get_size(), DNNL_RUNTIME_SIZE_VAL);
    CHECK_INVALID(test::make_memory(md_tag, eng));

    memory::desc md_strides {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL},
            data_type::f32, {100, 1}};
    ASSERT_EQ(md_strides.get_size(), DNNL_RUNTIME_SIZE_VAL);
    CHECK_INVALID(test::make_memory(md_strides, eng));
}

TEST_F(runtime_dim_test_t, TestBNorm) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    normalization_flags flags {};
    CHECK_UNIMPL(batch_normalization_forward::desc(
            prop_kind::forward, md, 0.1f, flags));
    CHECK_UNIMPL(batch_normalization_backward::desc(
            prop_kind::backward_data, md, md, 0.1f, flags));
}

TEST_F(runtime_dim_test_t, TestBinary) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(binary::desc(algorithm::binary_add, md, md, md));
}

TEST_F(runtime_dim_test_t, TestConcat) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(concat::primitive_desc(1, {md, md}, eng));
}

TEST_F(runtime_dim_test_t, TestConv) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 32, 7, 7}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}));
    CHECK_UNIMPL(convolution_backward_data::desc(algorithm::convolution_direct,
            src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1}));
    CHECK_UNIMPL(
            convolution_backward_weights::desc(algorithm::convolution_direct,
                    src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1}));
}

TEST_F(runtime_dim_test_t, TestDeconv) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 32, 7, 7}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(deconvolution_forward::desc(prop_kind::forward,
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}));
    CHECK_UNIMPL(
            deconvolution_backward_data::desc(algorithm::deconvolution_direct,
                    src_md, wei_md, dst_md, {1, 1}, {1, 1}, {1, 1}));
    CHECK_UNIMPL(deconvolution_backward_weights::desc(
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}));
}

TEST_F(runtime_dim_test_t, TestEltwise) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_relu, md, 0.1f, 0.f));
    CHECK_UNIMPL(
            eltwise_backward::desc(algorithm::eltwise_relu, md, md, 0.1f, 0.f));
}

TEST_F(runtime_dim_test_t, TestInnerProduct) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc wei_md {{32, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc dst_md {{DNNL_RUNTIME_DIM_VAL, 32}, data_type::f32, tag::ab};
    CHECK_UNIMPL(inner_product_forward::desc(
            prop_kind::forward, src_md, wei_md, dst_md));
    CHECK_UNIMPL(inner_product_backward_data::desc(src_md, wei_md, dst_md));
    CHECK_UNIMPL(inner_product_backward_weights::desc(src_md, wei_md, dst_md));
}

TEST_F(runtime_dim_test_t, TestLNorm) {
    memory::desc md {{DNNL_RUNTIME_DIM_VAL, 16, 16}, data_type::f32, tag::abc};
    memory::desc stat_md {{DNNL_RUNTIME_DIM_VAL, 16}, data_type::f32, tag::ab};
    normalization_flags flags {};
    CHECK_UNIMPL(layer_normalization_forward::desc(
            prop_kind::forward, md, stat_md, 0.1f, flags));
    CHECK_UNIMPL(layer_normalization_backward::desc(
            prop_kind::backward_data, md, md, stat_md, 0.1f, flags));
}

TEST_F(runtime_dim_test_t, TestLRN) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(lrn_forward::desc(prop_kind::forward,
            algorithm::lrn_across_channels, md, 5, 1.f, 0.75f));
    CHECK_UNIMPL(lrn_backward::desc(
            algorithm::lrn_across_channels, md, md, 5, 1.f, 0.75f));
}

CPU_TEST_F(runtime_dim_test_t, TestMatmul) {
    memory::desc a_md {{DNNL_RUNTIME_DIM_VAL, 3}, data_type::f32, tag::ab};
    memory::desc b_md {{3, DNNL_RUNTIME_DIM_VAL}, data_type::f32, tag::ba};
    memory::desc c_md {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL},
            data_type::f32, tag::ab};
    CHECK_OK(matmul::desc(a_md, b_md, c_md));
}

TEST_F(runtime_dim_test_t, TestPool) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 8, 8}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 4, 4}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(
            pooling_forward::desc(prop_kind::forward, algorithm::pooling_max,
                    src_md, dst_md, {2, 2}, {2, 2}, {0, 0}, {0, 0}));
    CHECK_UNIMPL(pooling_backward::desc(algorithm::pooling_max, src_md, dst_md,
            {2, 2}, {2, 2}, {0, 0}, {0, 0}));
}

CPU_TEST_F(runtime_dim_test_t, TestReorder) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 8, 8}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 8, 8}, data_type::f32, tag::acdb};
    CHECK_OK(reorder::primitive_desc(eng, src_md, eng, dst_md));
}

TEST_F(runtime_dim_test_t, TestRNN) {
    memory::dim l = 10, c = 8, g = 1, d = 1;
    memory::desc src_layer_md {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL, c},
            data_type::f32, tag::tnc};
    memory::desc src_iter_md {
            {l, d, DNNL_RUNTIME_DIM_VAL, c}, data_type::f32, tag::ldnc};
    memory::desc wei_layer_md {{l, d, c, g, c}, data_type::f32, tag::ldigo};
    memory::desc wei_iter_md {{l, d, c, g, c}, data_type::f32, tag::ldigo};
    memory::desc bia_md {{l, d, g, c}, data_type::f32, tag::ldgo};
    memory::desc dst_layer_md {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL, c},
            data_type::f32, tag::tnc};
    memory::desc dst_iter_md {
            {l, d, DNNL_RUNTIME_DIM_VAL, c}, data_type::f32, tag::ldnc};
    CHECK_UNIMPL(vanilla_rnn_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, rnn_direction::unidirectional_left2right,
            src_layer_md, src_iter_md, wei_layer_md, wei_iter_md, bia_md,
            dst_layer_md, dst_iter_md));
}

TEST_F(runtime_dim_test_t, TestShuffle) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(shuffle_forward::desc(prop_kind::forward, md, 1, 4));
    CHECK_UNIMPL(shuffle_backward::desc(md, 1, 4));
}

TEST_F(runtime_dim_test_t, TestSoftmax) {
    memory::desc md {{DNNL_RUNTIME_DIM_VAL, 16}, data_type::f32, tag::ab};
    CHECK_UNIMPL(softmax_forward::desc(prop_kind::forward, md, 1));
    CHECK_UNIMPL(softmax_backward::desc(md, md, 1));
}

TEST_F(runtime_dim_test_t, TestSum) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(sum::primitive_desc({1.f, 1.f}, {md, md}, eng));
}

} // namespace dnnl
