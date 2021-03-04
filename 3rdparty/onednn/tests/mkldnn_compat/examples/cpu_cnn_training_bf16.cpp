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

/// @example cpu_cnn_training_bf16.cpp
/// @copybrief cpu_cnn_training_bf16_cpp
///
/// @page cpu_cnn_training_bf16_cpp CNN bf16 training example
/// This C++ API example demonstrates how to build an AlexNet model training
/// using the bfloat16 data type.
///
/// The example implements a few layers from AlexNet model.
///
/// @include cpu_cnn_training_bf16.cpp

#include <assert.h>

#include <iostream>
#include <math.h>
#include <numeric>
#include <string>
#include "mkldnn.hpp"

using namespace mkldnn;

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

void simple_net() {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    auto cpu_engine = engine(engine::kind::cpu, 0);
    stream s(cpu_engine);

    // Vector of primitives and their execute arguments
    std::vector<primitive> net_fwd, net_bwd;
    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;

    const int batch = 1;

    // float data type is used for user data
    std::vector<float> net_src(batch * 3 * 227 * 227);
    std::vector<float> net_dst(batch * 96 * 27 * 27);

    // initializing non-zero values for src
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = sinf((float)i);

    // AlexNet: conv
    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
    // strides: {4, 4}

    memory::dims conv_src_tz = {batch, 3, 227, 227};
    memory::dims conv_weights_tz = {96, 3, 11, 11};
    memory::dims conv_bias_tz = {96};
    memory::dims conv_dst_tz = {batch, 96, 55, 55};
    memory::dims conv_strides = {4, 4};
    memory::dims conv_padding = {0, 0};

    // float data type is used for user data
    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));

    // initializing non-zero values for weights and bias
    for (size_t i = 0; i < conv_weights.size(); ++i)
        conv_weights[i] = sinf((float)i);
    for (size_t i = 0; i < conv_bias.size(); ++i)
        conv_bias[i] = sinf((float)i);

    // create memory for user data
    auto conv_user_src_memory = memory(
            {{conv_src_tz}, dt::f32, tag::nchw}, cpu_engine, net_src.data());
    auto conv_user_weights_memory
            = memory({{conv_weights_tz}, dt::f32, tag::oihw}, cpu_engine,
                    conv_weights.data());
    auto conv_user_bias_memory = memory(
            {{conv_bias_tz}, dt::f32, tag::x}, cpu_engine, conv_bias.data());

    // create memory descriptors for bfloat16 convolution data w/ no specified
    // format tag(`any`)
    // tag `any` lets a primitive(convolution in this case)
    // chose the memory format preferred for best performance.
    auto conv_src_md = memory::desc({conv_src_tz}, dt::bf16, tag::any);
    auto conv_bias_md = memory::desc({conv_bias_tz}, dt::bf16, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::bf16, tag::any);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::bf16, tag::any);

    // create a convolution primitive descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_bias_md, conv_dst_md, conv_strides, conv_padding,
            conv_padding);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);

    // create reorder primitives between user input and conv src if needed
    auto conv_src_memory = conv_user_src_memory;
    if (conv_pd.src_desc() != conv_user_src_memory.get_desc()) {
        conv_src_memory = memory(conv_pd.src_desc(), cpu_engine);
        net_fwd.push_back(reorder(conv_user_src_memory, conv_src_memory));
        net_fwd_args.push_back({{MKLDNN_ARG_FROM, conv_user_src_memory},
                {MKLDNN_ARG_TO, conv_src_memory}});
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (conv_pd.weights_desc() != conv_user_weights_memory.get_desc()) {
        conv_weights_memory = memory(conv_pd.weights_desc(), cpu_engine);
        net_fwd.push_back(
                reorder(conv_user_weights_memory, conv_weights_memory));
        net_fwd_args.push_back({{MKLDNN_ARG_FROM, conv_user_weights_memory},
                {MKLDNN_ARG_TO, conv_weights_memory}});
    }

    // create memory for conv dst
    auto conv_dst_memory = memory(conv_pd.dst_desc(), cpu_engine);

    // finally create a convolution primitive
    net_fwd.push_back(convolution_forward(conv_pd));
    net_fwd_args.push_back({{MKLDNN_ARG_SRC, conv_src_memory},
            {MKLDNN_ARG_WEIGHTS, conv_weights_memory},
            {MKLDNN_ARG_BIAS, conv_user_bias_memory},
            {MKLDNN_ARG_DST, conv_dst_memory}});

    // AlexNet: relu
    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}

    const float negative_slope = 1.0f;

    // create relu primitive desc
    // keep memory format tag of source same as the format tag of convolution
    // output in order to avoid reorder
    auto relu_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, conv_pd.dst_desc(), negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, cpu_engine);

    // create relu dst memory
    auto relu_dst_memory = memory(relu_pd.dst_desc(), cpu_engine);

    // finally create a relu primitive
    net_fwd.push_back(eltwise_forward(relu_pd));
    net_fwd_args.push_back({{MKLDNN_ARG_SRC, conv_dst_memory},
            {MKLDNN_ARG_DST, relu_dst_memory}});

    // AlexNet: lrn
    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
    // local size: 5
    // alpha: 0.0001
    // beta: 0.75
    // k: 1.0
    const uint32_t local_size = 5;
    const float alpha = 0.0001f;
    const float beta = 0.75f;
    const float k = 1.0f;

    // create a lrn primitive descriptor
    auto lrn_desc = lrn_forward::desc(prop_kind::forward,
            algorithm::lrn_across_channels, relu_pd.dst_desc(), local_size,
            alpha, beta, k);
    auto lrn_pd = lrn_forward::primitive_desc(lrn_desc, cpu_engine);

    // create lrn dst memory
    auto lrn_dst_memory = memory(lrn_pd.dst_desc(), cpu_engine);

    // create workspace only in training and only for forward primitive
    // query lrn_pd for workspace, this memory will be shared with forward lrn
    auto lrn_workspace_memory = memory(lrn_pd.workspace_desc(), cpu_engine);

    // finally create a lrn primitive
    net_fwd.push_back(lrn_forward(lrn_pd));
    net_fwd_args.push_back({{MKLDNN_ARG_SRC, relu_dst_memory},
            {MKLDNN_ARG_DST, lrn_dst_memory},
            {MKLDNN_ARG_WORKSPACE, lrn_workspace_memory}});

    // AlexNet: pool
    // {batch, 96, 55, 55} -> {batch, 96, 27, 27}
    // kernel: {3, 3}
    // strides: {2, 2}

    memory::dims pool_dst_tz = {batch, 96, 27, 27};
    memory::dims pool_kernel = {3, 3};
    memory::dims pool_strides = {2, 2};
    memory::dims pool_padding = {0, 0};

    // create memory for pool dst data in user format
    auto pool_user_dst_memory = memory(
            {{pool_dst_tz}, dt::f32, tag::nchw}, cpu_engine, net_dst.data());

    // create pool dst memory descriptor in format any for bfloat16 data type
    auto pool_dst_md = memory::desc({pool_dst_tz}, dt::bf16, tag::any);

    // create a pooling primitive descriptor
    auto pool_desc = pooling_forward::desc(prop_kind::forward,
            algorithm::pooling_max, lrn_dst_memory.get_desc(), pool_dst_md,
            pool_strides, pool_kernel, pool_padding, pool_padding);
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, cpu_engine);

    // create pooling workspace memory if training
    auto pool_workspace_memory = memory(pool_pd.workspace_desc(), cpu_engine);

    // create a pooling primitive
    net_fwd.push_back(pooling_forward(pool_pd));
    // leave DST unknown for now (see the next reorder)
    net_fwd_args.push_back({{MKLDNN_ARG_SRC, lrn_dst_memory},
            // delay putting DST until reorder (if needed)
            {MKLDNN_ARG_WORKSPACE, pool_workspace_memory}});

    // create reorder primitive between pool dst and user dst format
    // if needed
    auto pool_dst_memory = pool_user_dst_memory;
    if (pool_pd.dst_desc() != pool_user_dst_memory.get_desc()) {
        pool_dst_memory = memory(pool_pd.dst_desc(), cpu_engine);
        net_fwd_args.back().insert({MKLDNN_ARG_DST, pool_dst_memory});

        net_fwd.push_back(reorder(pool_dst_memory, pool_user_dst_memory));
        net_fwd_args.push_back({{MKLDNN_ARG_FROM, pool_dst_memory},
                {MKLDNN_ARG_TO, pool_user_dst_memory}});
    } else {
        net_fwd_args.back().insert({MKLDNN_ARG_DST, pool_dst_memory});
    }

    //-----------------------------------------------------------------------
    //----------------- Backward Stream -------------------------------------
    // ... user diff_data in float data type ...
    std::vector<float> net_diff_dst(batch * 96 * 27 * 27);
    for (size_t i = 0; i < net_diff_dst.size(); ++i)
        net_diff_dst[i] = sinf((float)i);

    // create memory for user diff dst data stored in float data type
    auto pool_user_diff_dst_memory = memory({{pool_dst_tz}, dt::f32, tag::nchw},
            cpu_engine, net_diff_dst.data());

    // Backward pooling
    // create memory descriptors for pooling
    auto pool_diff_src_md = lrn_dst_memory.get_desc();
    auto pool_diff_dst_md = pool_dst_memory.get_desc();

    // create backward pooling descriptor
    auto pool_bwd_desc = pooling_backward::desc(algorithm::pooling_max,
            pool_diff_src_md, pool_diff_dst_md, pool_strides, pool_kernel,
            pool_padding, pool_padding);
    // backward primitive descriptor needs to hint forward descriptor
    auto pool_bwd_pd = pooling_backward::primitive_desc(
            pool_bwd_desc, cpu_engine, pool_pd);

    // create reorder primitive between user diff dst and pool diff dst
    // if required
    auto pool_diff_dst_memory = pool_user_diff_dst_memory;
    if (pool_dst_memory.get_desc() != pool_user_diff_dst_memory.get_desc()) {
        pool_diff_dst_memory = memory(pool_dst_memory.get_desc(), cpu_engine);
        net_bwd.push_back(
                reorder(pool_user_diff_dst_memory, pool_diff_dst_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, pool_user_diff_dst_memory},
                {MKLDNN_ARG_TO, pool_diff_dst_memory}});
    }

    // create memory for pool diff src
    auto pool_diff_src_memory = memory(pool_bwd_pd.diff_src_desc(), cpu_engine);

    // finally create backward pooling primitive
    net_bwd.push_back(pooling_backward(pool_bwd_pd));
    net_bwd_args.push_back({{MKLDNN_ARG_DIFF_DST, pool_diff_dst_memory},
            {MKLDNN_ARG_DIFF_SRC, pool_diff_src_memory},
            {MKLDNN_ARG_WORKSPACE, pool_workspace_memory}});

    // Backward lrn
    auto lrn_diff_dst_md = lrn_dst_memory.get_desc();

    // create backward lrn primitive descriptor
    auto lrn_bwd_desc = lrn_backward::desc(algorithm::lrn_across_channels,
            lrn_pd.src_desc(), lrn_diff_dst_md, local_size, alpha, beta, k);
    auto lrn_bwd_pd
            = lrn_backward::primitive_desc(lrn_bwd_desc, cpu_engine, lrn_pd);

    // create memory for lrn diff src
    auto lrn_diff_src_memory = memory(lrn_bwd_pd.diff_src_desc(), cpu_engine);

    // finally create a lrn backward primitive
    // backward lrn needs src: relu dst in this topology
    net_bwd.push_back(lrn_backward(lrn_bwd_pd));
    net_bwd_args.push_back({{MKLDNN_ARG_SRC, relu_dst_memory},
            {MKLDNN_ARG_DIFF_DST, pool_diff_src_memory},
            {MKLDNN_ARG_DIFF_SRC, lrn_diff_src_memory},
            {MKLDNN_ARG_WORKSPACE, lrn_workspace_memory}});

    // Backward relu
    auto relu_diff_dst_md = lrn_diff_src_memory.get_desc();
    auto relu_src_md = conv_pd.dst_desc();

    // create backward relu primitive_descriptor
    auto relu_bwd_desc = eltwise_backward::desc(algorithm::eltwise_relu,
            relu_diff_dst_md, relu_src_md, negative_slope);
    auto relu_bwd_pd = eltwise_backward::primitive_desc(
            relu_bwd_desc, cpu_engine, relu_pd);

    // create memory for relu diff src
    auto relu_diff_src_memory = memory(relu_bwd_pd.diff_src_desc(), cpu_engine);

    // finally create a backward relu primitive
    net_bwd.push_back(eltwise_backward(relu_bwd_pd));
    net_bwd_args.push_back({{MKLDNN_ARG_SRC, conv_dst_memory},
            {MKLDNN_ARG_DIFF_DST, lrn_diff_src_memory},
            {MKLDNN_ARG_DIFF_SRC, relu_diff_src_memory}});

    // Backward convolution with respect to weights
    // create user format diff weights and diff bias memory for float data type
    std::vector<float> conv_user_diff_weights_buffer(product(conv_weights_tz));
    std::vector<float> conv_diff_bias_buffer(product(conv_bias_tz));

    auto conv_user_diff_weights_memory
            = memory({{conv_weights_tz}, dt::f32, tag::nchw}, cpu_engine,
                    conv_user_diff_weights_buffer.data());
    auto conv_diff_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x},
            cpu_engine, conv_diff_bias_buffer.data());

    // create memory descriptors for bfloat16 convolution data
    auto conv_bwd_src_md = memory::desc({conv_src_tz}, dt::bf16, tag::any);
    auto conv_diff_bias_md = memory::desc({conv_bias_tz}, dt::bf16, tag::any);
    auto conv_diff_weights_md
            = memory::desc({conv_weights_tz}, dt::bf16, tag::any);
    auto conv_diff_dst_md = memory::desc({conv_dst_tz}, dt::bf16, tag::any);

    // create backward convolution primitive descriptor
    auto conv_bwd_weights_desc
            = convolution_backward_weights::desc(algorithm::convolution_direct,
                    conv_bwd_src_md, conv_diff_weights_md, conv_diff_bias_md,
                    conv_diff_dst_md, conv_strides, conv_padding, conv_padding);
    auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
            conv_bwd_weights_desc, cpu_engine, conv_pd);

    // for best performance convolution backward might chose
    // different memory format for src and diff_dst
    // than the memory formats preferred by forward convolution
    // for src and dst respectively
    // create reorder primitives for src from forward convolution to the
    // format chosen by backward convolution
    auto conv_bwd_src_memory = conv_src_memory;
    if (conv_bwd_weights_pd.src_desc() != conv_src_memory.get_desc()) {
        conv_bwd_src_memory
                = memory(conv_bwd_weights_pd.src_desc(), cpu_engine);
        net_bwd.push_back(reorder(conv_src_memory, conv_bwd_src_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_src_memory},
                {MKLDNN_ARG_TO, conv_bwd_src_memory}});
    }

    // create reorder primitives for diff_dst between diff_src from relu_bwd
    // and format preferred by conv_diff_weights
    auto conv_diff_dst_memory = relu_diff_src_memory;
    if (conv_bwd_weights_pd.diff_dst_desc()
            != relu_diff_src_memory.get_desc()) {
        conv_diff_dst_memory
                = memory(conv_bwd_weights_pd.diff_dst_desc(), cpu_engine);
        net_bwd.push_back(reorder(relu_diff_src_memory, conv_diff_dst_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, relu_diff_src_memory},
                {MKLDNN_ARG_TO, conv_diff_dst_memory}});
    }

    // create backward convolution primitive
    net_bwd.push_back(convolution_backward_weights(conv_bwd_weights_pd));
    net_bwd_args.push_back({{MKLDNN_ARG_SRC, conv_bwd_src_memory},
            {MKLDNN_ARG_DIFF_DST, conv_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {MKLDNN_ARG_DIFF_BIAS, conv_diff_bias_memory}});

    // create reorder primitives between conv diff weights and user diff weights
    // if needed
    auto conv_diff_weights_memory = conv_user_diff_weights_memory;
    if (conv_bwd_weights_pd.diff_weights_desc()
            != conv_user_diff_weights_memory.get_desc()) {
        conv_diff_weights_memory
                = memory(conv_bwd_weights_pd.diff_weights_desc(), cpu_engine);
        net_bwd_args.back().insert(
                {MKLDNN_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});

        net_bwd.push_back(reorder(
                conv_diff_weights_memory, conv_user_diff_weights_memory));
        net_bwd_args.push_back({{MKLDNN_ARG_FROM, conv_diff_weights_memory},
                {MKLDNN_ARG_TO, conv_user_diff_weights_memory}});
    } else {
        net_bwd_args.back().insert(
                {MKLDNN_ARG_DIFF_WEIGHTS, conv_diff_weights_memory});
    }

    // didn't we forget anything?
    assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
    assert(net_bwd.size() == net_bwd_args.size() && "something is missing");

    int n_iter = 1; // number of iterations for training
    // execute
    while (n_iter) {
        // forward
        for (size_t i = 0; i < net_fwd.size(); ++i)
            net_fwd.at(i).execute(s, net_fwd_args.at(i));

        // update net_diff_dst
        // auto net_output = pool_user_dst_memory.get_data_handle();
        // ..user updates net_diff_dst using net_output...
        // some user defined func update_diff_dst(net_diff_dst.data(),
        // net_output)

        for (size_t i = 0; i < net_bwd.size(); ++i)
            net_bwd.at(i).execute(s, net_bwd_args.at(i));
        // update weights and bias using diff weights and bias
        //
        // auto net_diff_weights
        //     = conv_user_diff_weights_memory.get_data_handle();
        // auto net_diff_bias = conv_diff_bias_memory.get_data_handle();
        //
        // ...user updates weights and bias using diff weights and bias...
        //
        // some user defined func update_weights(conv_weights.data(),
        // conv_bias.data(), net_diff_weights, net_diff_bias);

        --n_iter;
    }

    s.wait();
}

int main(int argc, char **argv) {
    try {
        simple_net();
        std::cout << "passed" << std::endl;
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
