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

/// @example cnn_inference_f32.cpp
/// @copybrief cnn_inference_f32_cpp
/// > Annotated version: @ref cnn_inference_f32_cpp

/// @page cnn_inference_f32_cpp CNN f32 inference example
/// This C++ API example demonstrates how to build an AlexNet neural
/// network topology for forward-pass inference.
///
/// > Example code: @ref cnn_inference_f32.cpp
///
/// Some key take-aways include:
///
/// * How tensors are implemented and submitted to primitives.
/// * How primitives are created.
/// * How primitives are sequentially submitted to the network, where the output
///   from primitives is passed as input to the next primitive. The latter
///   specifies a dependency between the primitive input and output data.
/// * Specific 'inference-only' configurations.
/// * Limiting the number of reorders performed that are detrimental
///   to performance.
///
/// The example implements the AlexNet layers
/// as numbered primitives (for example, conv1, pool1, conv2).

#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

void simple_net(engine::kind engine_kind, int times = 100) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    /// @snippet cnn_inference_f32.cpp Initialize engine and stream
    //[Initialize engine and stream]
    engine eng(engine_kind, 0);
    stream s(eng);
    //[Initialize engine and stream]

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    /// @snippet cnn_inference_f32.cpp Create network
    //[Create network]
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]

    const memory::dim batch = 1;

    // AlexNet: conv1
    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
    // strides: {4, 4}
    memory::dims conv1_src_tz = {batch, 3, 227, 227};
    memory::dims conv1_weights_tz = {96, 3, 11, 11};
    memory::dims conv1_bias_tz = {96};
    memory::dims conv1_dst_tz = {batch, 96, 55, 55};
    memory::dims conv1_strides = {4, 4};
    memory::dims conv1_padding = {0, 0};

    /// Allocate buffers for input and output data, weights, and bias.
    /// @snippet cnn_inference_f32.cpp Allocate buffers
    //[Allocate buffers]
    std::vector<float> user_src(batch * 3 * 227 * 227);
    std::vector<float> user_dst(batch * 1000);
    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));
    //[Allocate buffers]

    /// Create memory that describes data layout in the buffers. This example uses
    /// tag::nchw (batch-channels-height-width) for input data and tag::oihw
    /// for weights.
    /// @snippet cnn_inference_f32.cpp Create user memory
    //[Create user memory]
    auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    auto user_weights_memory
            = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);
    auto conv1_user_bias_memory
            = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);
    //[Create user memory]

    /// Create memory descriptors with layout tag::any. The `any` format enables
    /// the convolution primitive to choose the data format that will result in
    /// best performance based on its input parameters (convolution kernel
    /// sizes, strides, padding, and so on). If the resulting format is different
    /// from `nchw`, the user data must be transformed to the format required for
    /// the convolution (as explained below).
    /// @snippet cnn_inference_f32.cpp Create convolution memory descriptors
    //[Create convolution memory descriptors]
    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);
    //[Create convolution memory descriptors]

    /// Create a convolution descriptor by specifying propagation kind,
    /// [convolution algorithm](@ref dev_guide_convolution), shapes of input,
    /// weights, bias, output, convolution strides, padding, and kind of padding.
    /// Propagation kind is set to prop_kind::forward_inference to optimize for
    /// inference execution and omit computations that are necessary only for
    /// backward propagation.
    /// @snippet cnn_inference_f32.cpp Create convolution descriptor
    //[Create convolution descriptor]
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
            conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
            conv1_padding);
    //[Create convolution descriptor]

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the `any` format specified
    /// in the convolution descriptor.
    /// @snippet cnn_inference_f32.cpp Create convolution primitive descriptor
    //[Create convolution primitive descriptor]
    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);
    //[Create convolution primitive descriptor]

    /// Check whether data and weights formats required by convolution is different
    /// from the user format. In case it is different change the layout using
    /// reorder primitive.
    /// @snippet cnn_inference_f32.cpp Reorder data and weights
    //[Reorder data and weights]
    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                {DNNL_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
                .execute(s, user_weights_memory, conv1_weights_memory);
    }
    //[Reorder data and weights]

    /// Create a memory primitive for output.
    /// @snippet cnn_inference_f32.cpp Create memory for output
    //[Create memory for output]
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
    //[Create memory for output]

    /// Create a convolution primitive and add it to the net.
    /// @snippet cnn_inference_f32.cpp Create memory for output
    //[Create convolution primitive]
    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
            {DNNL_ARG_BIAS, conv1_user_bias_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});
    //[Create convolution primitive]

    // AlexNet: relu1
    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
    const float negative1_slope = 0.0f;

    /// Create the relu primitive. For better performance, keep the input data
    /// format for ReLU (as well as for other operation primitives until another
    /// convolution or inner product is encountered) the same as the one chosen
    /// for convolution. Also note that ReLU is done in-place by using conv1 memory.
    /// @snippet cnn_inference_f32.cpp Create relu primitive
    //[Create relu primitive]
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv1_dst_memory.get_desc(),
            negative1_slope);
    auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);

    net.push_back(eltwise_forward(relu1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});
    //[Create relu primitive]

    // AlexNet: lrn1
    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
    // local size: 5
    // alpha1: 0.0001
    // beta1: 0.75
    const memory::dim local1_size = 5;
    const float alpha1 = 0.0001f;
    const float beta1 = 0.75f;
    const float k1 = 1.0f;

    // create lrn primitive and add it to net
    auto lrn1_desc = lrn_forward::desc(prop_kind::forward_inference,
            algorithm::lrn_across_channels, conv1_dst_memory.get_desc(),
            local1_size, alpha1, beta1, k1);
    auto lrn1_prim_desc = lrn_forward::primitive_desc(lrn1_desc, eng);
    auto lrn1_dst_memory = memory(lrn1_prim_desc.dst_desc(), eng);

    net.push_back(lrn_forward(lrn1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
            {DNNL_ARG_DST, lrn1_dst_memory}});

    // AlexNet: pool1
    // {batch, 96, 55, 55} -> {batch, 96, 27, 27}
    // kernel: {3, 3}
    // strides: {2, 2}
    memory::dims pool1_dst_tz = {batch, 96, 27, 27};
    memory::dims pool1_kernel = {3, 3};
    memory::dims pool1_strides = {2, 2};
    memory::dims pool_padding = {0, 0};

    auto pool1_dst_md = memory::desc({pool1_dst_tz}, dt::f32, tag::any);

    /// For training execution, pooling requires a private workspace memory
    /// to perform the backward pass. However, pooling should not use 'workspace'
    /// for inference, because this is detrimental to performance.
    /// @snippet cnn_inference_f32.cpp Create pooling primitive
    ///
    /// The example continues to create more layers according
    /// to the AlexNet topology.
    //[Create pooling primitive]
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, lrn1_dst_memory.get_desc(), pool1_dst_md,
            pool1_strides, pool1_kernel, pool_padding, pool_padding);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, eng);
    auto pool1_dst_memory = memory(pool1_pd.dst_desc(), eng);

    net.push_back(pooling_forward(pool1_pd));
    net_args.push_back({{DNNL_ARG_SRC, lrn1_dst_memory},
            {DNNL_ARG_DST, pool1_dst_memory}});
    //[Create pooling primitive]

    // AlexNet: conv2
    // {batch, 96, 27, 27} (x) {2, 128, 48, 5, 5} -> {batch, 256, 27, 27}
    // strides: {1, 1}
    memory::dims conv2_src_tz = {batch, 96, 27, 27};
    memory::dims conv2_weights_tz = {2, 128, 48, 5, 5};
    memory::dims conv2_bias_tz = {256};
    memory::dims conv2_dst_tz = {batch, 256, 27, 27};
    memory::dims conv2_strides = {1, 1};
    memory::dims conv2_padding = {2, 2};

    std::vector<float> conv2_weights(product(conv2_weights_tz));
    std::vector<float> conv2_bias(product(conv2_bias_tz));

    // create memory for user data
    auto conv2_user_weights_memory
            = memory({{conv2_weights_tz}, dt::f32, tag::goihw}, eng);
    write_to_dnnl_memory(conv2_weights.data(), conv2_user_weights_memory);
    auto conv2_user_bias_memory
            = memory({{conv2_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv2_bias.data(), conv2_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto conv2_src_md = memory::desc({conv2_src_tz}, dt::f32, tag::any);
    auto conv2_bias_md = memory::desc({conv2_bias_tz}, dt::f32, tag::any);
    auto conv2_weights_md = memory::desc({conv2_weights_tz}, dt::f32, tag::any);
    auto conv2_dst_md = memory::desc({conv2_dst_tz}, dt::f32, tag::any);

    // create a convolution
    auto conv2_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv2_src_md, conv2_weights_md,
            conv2_bias_md, conv2_dst_md, conv2_strides, conv2_padding,
            conv2_padding);
    auto conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, eng);

    auto conv2_src_memory = pool1_dst_memory;
    if (conv2_prim_desc.src_desc() != conv2_src_memory.get_desc()) {
        conv2_src_memory = memory(conv2_prim_desc.src_desc(), eng);
        net.push_back(reorder(pool1_dst_memory, conv2_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, pool1_dst_memory},
                {DNNL_ARG_TO, conv2_src_memory}});
    }

    auto conv2_weights_memory = conv2_user_weights_memory;
    if (conv2_prim_desc.weights_desc()
            != conv2_user_weights_memory.get_desc()) {
        conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), eng);
        reorder(conv2_user_weights_memory, conv2_weights_memory)
                .execute(s, conv2_user_weights_memory, conv2_weights_memory);
    }

    auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(convolution_forward(conv2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_src_memory},
            {DNNL_ARG_WEIGHTS, conv2_weights_memory},
            {DNNL_ARG_BIAS, conv2_user_bias_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});

    // AlexNet: relu2
    // {batch, 256, 27, 27} -> {batch, 256, 27, 27}
    const float negative2_slope = 0.0f;

    // create relu primitive and add it to net
    auto relu2_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv2_dst_memory.get_desc(),
            negative2_slope);
    auto relu2_prim_desc = eltwise_forward::primitive_desc(relu2_desc, eng);

    net.push_back(eltwise_forward(relu2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_dst_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});

    // AlexNet: lrn2
    // {batch, 256, 27, 27} -> {batch, 256, 27, 27}
    // local size: 5
    // alpha2: 0.0001
    // beta2: 0.75
    const memory::dim local2_size = 5;
    const float alpha2 = 0.0001f;
    const float beta2 = 0.75f;
    const float k2 = 1.0f;

    // create lrn primitive and add it to net
    auto lrn2_desc = lrn_forward::desc(prop_kind::forward_inference,
            algorithm::lrn_across_channels, conv2_prim_desc.dst_desc(),
            local2_size, alpha2, beta2, k2);
    auto lrn2_prim_desc = lrn_forward::primitive_desc(lrn2_desc, eng);
    auto lrn2_dst_memory = memory(lrn2_prim_desc.dst_desc(), eng);

    net.push_back(lrn_forward(lrn2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_dst_memory},
            {DNNL_ARG_DST, lrn2_dst_memory}});

    // AlexNet: pool2
    // {batch, 256, 27, 27} -> {batch, 256, 13, 13}
    // kernel: {3, 3}
    // strides: {2, 2}
    memory::dims pool2_dst_tz = {batch, 256, 13, 13};
    memory::dims pool2_kernel = {3, 3};
    memory::dims pool2_strides = {2, 2};
    memory::dims pool2_padding = {0, 0};

    auto pool2_dst_md = memory::desc({pool2_dst_tz}, dt::f32, tag::any);

    // create a pooling
    auto pool2_desc = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, lrn2_dst_memory.get_desc(), pool2_dst_md,
            pool2_strides, pool2_kernel, pool2_padding, pool2_padding);
    auto pool2_pd = pooling_forward::primitive_desc(pool2_desc, eng);
    auto pool2_dst_memory = memory(pool2_pd.dst_desc(), eng);

    // create pooling primitive an add it to net
    net.push_back(pooling_forward(pool2_pd));
    net_args.push_back({{DNNL_ARG_SRC, lrn2_dst_memory},
            {DNNL_ARG_DST, pool2_dst_memory}});

    // AlexNet: conv3
    // {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13};
    // strides: {1, 1}
    memory::dims conv3_src_tz = {batch, 256, 13, 13};
    memory::dims conv3_weights_tz = {384, 256, 3, 3};
    memory::dims conv3_bias_tz = {384};
    memory::dims conv3_dst_tz = {batch, 384, 13, 13};
    memory::dims conv3_strides = {1, 1};
    memory::dims conv3_padding = {1, 1};

    std::vector<float> conv3_weights(product(conv3_weights_tz));
    std::vector<float> conv3_bias(product(conv3_bias_tz));

    // create memory for user data
    auto conv3_user_weights_memory
            = memory({{conv3_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv3_weights.data(), conv3_user_weights_memory);
    auto conv3_user_bias_memory
            = memory({{conv3_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv3_bias.data(), conv3_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto conv3_src_md = memory::desc({conv3_src_tz}, dt::f32, tag::any);
    auto conv3_bias_md = memory::desc({conv3_bias_tz}, dt::f32, tag::any);
    auto conv3_weights_md = memory::desc({conv3_weights_tz}, dt::f32, tag::any);
    auto conv3_dst_md = memory::desc({conv3_dst_tz}, dt::f32, tag::any);

    // create a convolution
    auto conv3_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv3_src_md, conv3_weights_md,
            conv3_bias_md, conv3_dst_md, conv3_strides, conv3_padding,
            conv3_padding);
    auto conv3_prim_desc = convolution_forward::primitive_desc(conv3_desc, eng);

    auto conv3_src_memory = pool2_dst_memory;
    if (conv3_prim_desc.src_desc() != conv3_src_memory.get_desc()) {
        conv3_src_memory = memory(conv3_prim_desc.src_desc(), eng);
        net.push_back(reorder(pool2_dst_memory, conv3_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, pool2_dst_memory},
                {DNNL_ARG_TO, conv3_src_memory}});
    }

    auto conv3_weights_memory = conv3_user_weights_memory;
    if (conv3_prim_desc.weights_desc()
            != conv3_user_weights_memory.get_desc()) {
        conv3_weights_memory = memory(conv3_prim_desc.weights_desc(), eng);
        reorder(conv3_user_weights_memory, conv3_weights_memory)
                .execute(s, conv3_user_weights_memory, conv3_weights_memory);
    }

    auto conv3_dst_memory = memory(conv3_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(convolution_forward(conv3_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv3_src_memory},
            {DNNL_ARG_WEIGHTS, conv3_weights_memory},
            {DNNL_ARG_BIAS, conv3_user_bias_memory},
            {DNNL_ARG_DST, conv3_dst_memory}});

    // AlexNet: relu3
    // {batch, 384, 13, 13} -> {batch, 384, 13, 13}
    const float negative3_slope = 0.0f;

    // create relu primitive and add it to net
    auto relu3_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv3_dst_memory.get_desc(),
            negative3_slope);
    auto relu3_prim_desc = eltwise_forward::primitive_desc(relu3_desc, eng);

    net.push_back(eltwise_forward(relu3_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv3_dst_memory},
            {DNNL_ARG_DST, conv3_dst_memory}});

    // AlexNet: conv4
    // {batch, 384, 13, 13} (x)  {2, 192, 192, 3, 3}; ->
    // {batch, 384, 13, 13};
    // strides: {1, 1}
    memory::dims conv4_src_tz = {batch, 384, 13, 13};
    memory::dims conv4_weights_tz = {2, 192, 192, 3, 3};
    memory::dims conv4_bias_tz = {384};
    memory::dims conv4_dst_tz = {batch, 384, 13, 13};
    memory::dims conv4_strides = {1, 1};
    memory::dims conv4_padding = {1, 1};

    std::vector<float> conv4_weights(product(conv4_weights_tz));
    std::vector<float> conv4_bias(product(conv4_bias_tz));

    // create memory for user data
    auto conv4_user_weights_memory
            = memory({{conv4_weights_tz}, dt::f32, tag::goihw}, eng);
    write_to_dnnl_memory(conv4_weights.data(), conv4_user_weights_memory);
    auto conv4_user_bias_memory
            = memory({{conv4_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv4_bias.data(), conv4_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto conv4_src_md = memory::desc({conv4_src_tz}, dt::f32, tag::any);
    auto conv4_bias_md = memory::desc({conv4_bias_tz}, dt::f32, tag::any);
    auto conv4_weights_md = memory::desc({conv4_weights_tz}, dt::f32, tag::any);
    auto conv4_dst_md = memory::desc({conv4_dst_tz}, dt::f32, tag::any);

    // create a convolution
    auto conv4_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv4_src_md, conv4_weights_md,
            conv4_bias_md, conv4_dst_md, conv4_strides, conv4_padding,
            conv4_padding);
    auto conv4_prim_desc = convolution_forward::primitive_desc(conv4_desc, eng);

    auto conv4_src_memory = conv3_dst_memory;
    if (conv4_prim_desc.src_desc() != conv4_src_memory.get_desc()) {
        conv4_src_memory = memory(conv4_prim_desc.src_desc(), eng);
        net.push_back(reorder(conv3_dst_memory, conv4_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv3_dst_memory},
                {DNNL_ARG_TO, conv4_src_memory}});
    }

    auto conv4_weights_memory = conv4_user_weights_memory;
    if (conv4_prim_desc.weights_desc()
            != conv4_user_weights_memory.get_desc()) {
        conv4_weights_memory = memory(conv4_prim_desc.weights_desc(), eng);
        reorder(conv4_user_weights_memory, conv4_weights_memory)
                .execute(s, conv4_user_weights_memory, conv4_weights_memory);
    }

    auto conv4_dst_memory = memory(conv4_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(convolution_forward(conv4_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv4_src_memory},
            {DNNL_ARG_WEIGHTS, conv4_weights_memory},
            {DNNL_ARG_BIAS, conv4_user_bias_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});

    // AlexNet: relu4
    // {batch, 384, 13, 13} -> {batch, 384, 13, 13}
    const float negative4_slope = 0.0f;

    // create relu primitive and add it to net
    auto relu4_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv4_dst_memory.get_desc(),
            negative4_slope);
    auto relu4_prim_desc = eltwise_forward::primitive_desc(relu4_desc, eng);

    net.push_back(eltwise_forward(relu4_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv4_dst_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});

    // AlexNet: conv5
    // {batch, 384, 13, 13} (x)  {2, 128, 192, 3, 3}; -> {batch, 256, 13, 13};
    // strides: {1, 1}
    memory::dims conv5_src_tz = {batch, 384, 13, 13};
    memory::dims conv5_weights_tz = {2, 128, 192, 3, 3};
    memory::dims conv5_bias_tz = {256};
    memory::dims conv5_dst_tz = {batch, 256, 13, 13};
    memory::dims conv5_strides = {1, 1};
    memory::dims conv5_padding = {1, 1};

    std::vector<float> conv5_weights(product(conv5_weights_tz));
    std::vector<float> conv5_bias(product(conv5_bias_tz));

    // create memory for user data
    auto conv5_user_weights_memory
            = memory({{conv5_weights_tz}, dt::f32, tag::goihw}, eng);
    write_to_dnnl_memory(conv5_weights.data(), conv5_user_weights_memory);
    auto conv5_user_bias_memory
            = memory({{conv5_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv5_bias.data(), conv5_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto conv5_src_md = memory::desc({conv5_src_tz}, dt::f32, tag::any);
    auto conv5_weights_md = memory::desc({conv5_weights_tz}, dt::f32, tag::any);
    auto conv5_bias_md = memory::desc({conv5_bias_tz}, dt::f32, tag::any);
    auto conv5_dst_md = memory::desc({conv5_dst_tz}, dt::f32, tag::any);

    // create a convolution
    auto conv5_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv5_src_md, conv5_weights_md,
            conv5_bias_md, conv5_dst_md, conv5_strides, conv5_padding,
            conv5_padding);
    auto conv5_prim_desc = convolution_forward::primitive_desc(conv5_desc, eng);

    auto conv5_src_memory = conv4_dst_memory;
    if (conv5_prim_desc.src_desc() != conv5_src_memory.get_desc()) {
        conv5_src_memory = memory(conv5_prim_desc.src_desc(), eng);
        net.push_back(reorder(conv4_dst_memory, conv5_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv4_dst_memory},
                {DNNL_ARG_TO, conv5_src_memory}});
    }

    auto conv5_weights_memory = conv5_user_weights_memory;
    if (conv5_prim_desc.weights_desc()
            != conv5_user_weights_memory.get_desc()) {
        conv5_weights_memory = memory(conv5_prim_desc.weights_desc(), eng);
        reorder(conv5_user_weights_memory, conv5_weights_memory)
                .execute(s, conv5_user_weights_memory, conv5_weights_memory);
    }

    auto conv5_dst_memory = memory(conv5_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(convolution_forward(conv5_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv5_src_memory},
            {DNNL_ARG_WEIGHTS, conv5_weights_memory},
            {DNNL_ARG_BIAS, conv5_user_bias_memory},
            {DNNL_ARG_DST, conv5_dst_memory}});

    // AlexNet: relu5
    // {batch, 256, 13, 13} -> {batch, 256, 13, 13}
    const float negative5_slope = 0.0f;

    // create relu primitive and add it to net
    auto relu5_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv5_dst_memory.get_desc(),
            negative5_slope);
    auto relu5_prim_desc = eltwise_forward::primitive_desc(relu5_desc, eng);

    net.push_back(eltwise_forward(relu5_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv5_dst_memory},
            {DNNL_ARG_DST, conv5_dst_memory}});

    // AlexNet: pool5
    // {batch, 256, 13, 13} -> {batch, 256, 6, 6}
    // kernel: {3, 3}
    // strides: {2, 2}
    memory::dims pool5_dst_tz = {batch, 256, 6, 6};
    memory::dims pool5_kernel = {3, 3};
    memory::dims pool5_strides = {2, 2};
    memory::dims pool5_padding = {0, 0};

    std::vector<float> pool5_dst(product(pool5_dst_tz));

    auto pool5_dst_md = memory::desc({pool5_dst_tz}, dt::f32, tag::any);

    // create a pooling
    auto pool5_desc = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, conv5_dst_memory.get_desc(), pool5_dst_md,
            pool5_strides, pool5_kernel, pool5_padding, pool5_padding);
    auto pool5_pd = pooling_forward::primitive_desc(pool5_desc, eng);

    auto pool5_dst_memory = memory(pool5_pd.dst_desc(), eng);

    // create pooling primitive an add it to net
    net.push_back(pooling_forward(pool5_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv5_dst_memory},
            {DNNL_ARG_DST, pool5_dst_memory}});

    // fc6 inner product {batch, 256, 6, 6} (x) {4096, 256, 6, 6}-> {batch,
    // 4096}
    memory::dims fc6_src_tz = {batch, 256, 6, 6};
    memory::dims fc6_weights_tz = {4096, 256, 6, 6};
    memory::dims fc6_bias_tz = {4096};
    memory::dims fc6_dst_tz = {batch, 4096};

    std::vector<float> fc6_weights(product(fc6_weights_tz));
    std::vector<float> fc6_bias(product(fc6_bias_tz));

    // create memory for user data
    auto fc6_user_weights_memory
            = memory({{fc6_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(fc6_weights.data(), fc6_user_weights_memory);
    auto fc6_user_bias_memory = memory({{fc6_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc6_bias.data(), fc6_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto fc6_src_md = memory::desc({fc6_src_tz}, dt::f32, tag::any);
    auto fc6_bias_md = memory::desc({fc6_bias_tz}, dt::f32, tag::any);
    auto fc6_weights_md = memory::desc({fc6_weights_tz}, dt::f32, tag::any);
    auto fc6_dst_md = memory::desc({fc6_dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto fc6_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc6_src_md, fc6_weights_md, fc6_bias_md, fc6_dst_md);
    auto fc6_prim_desc = inner_product_forward::primitive_desc(fc6_desc, eng);

    auto fc6_src_memory = pool5_dst_memory;
    if (fc6_prim_desc.src_desc() != fc6_src_memory.get_desc()) {
        fc6_src_memory = memory(fc6_prim_desc.src_desc(), eng);
        net.push_back(reorder(pool5_dst_memory, fc6_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, pool5_dst_memory},
                {DNNL_ARG_TO, fc6_src_memory}});
    }

    auto fc6_weights_memory = fc6_user_weights_memory;
    if (fc6_prim_desc.weights_desc() != fc6_user_weights_memory.get_desc()) {
        fc6_weights_memory = memory(fc6_prim_desc.weights_desc(), eng);
        reorder(fc6_user_weights_memory, fc6_weights_memory)
                .execute(s, fc6_user_weights_memory, fc6_weights_memory);
    }

    auto fc6_dst_memory = memory(fc6_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(inner_product_forward(fc6_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, fc6_src_memory},
            {DNNL_ARG_WEIGHTS, fc6_weights_memory},
            {DNNL_ARG_BIAS, fc6_user_bias_memory},
            {DNNL_ARG_DST, fc6_dst_memory}});

    // fc7 inner product {batch, 4096} (x) {4096, 4096}-> {batch, 4096}
    memory::dims fc7_weights_tz = {4096, 4096};
    memory::dims fc7_bias_tz = {4096};
    memory::dims fc7_dst_tz = {batch, 4096};

    std::vector<float> fc7_weights(product(fc7_weights_tz));
    std::vector<float> fc7_bias(product(fc7_bias_tz));

    // create memory for user data
    auto fc7_user_weights_memory
            = memory({{fc7_weights_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(fc7_weights.data(), fc7_user_weights_memory);

    auto fc7_user_bias_memory = memory({{fc7_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc7_bias.data(), fc7_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto fc7_bias_md = memory::desc({fc7_bias_tz}, dt::f32, tag::any);
    auto fc7_weights_md = memory::desc({fc7_weights_tz}, dt::f32, tag::any);
    auto fc7_dst_md = memory::desc({fc7_dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto fc7_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc6_dst_memory.get_desc(), fc7_weights_md, fc7_bias_md, fc7_dst_md);
    auto fc7_prim_desc = inner_product_forward::primitive_desc(fc7_desc, eng);

    auto fc7_weights_memory = fc7_user_weights_memory;
    if (fc7_prim_desc.weights_desc() != fc7_user_weights_memory.get_desc()) {
        fc7_weights_memory = memory(fc7_prim_desc.weights_desc(), eng);
        reorder(fc7_user_weights_memory, fc7_weights_memory)
                .execute(s, fc7_user_weights_memory, fc7_weights_memory);
    }

    auto fc7_dst_memory = memory(fc7_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(inner_product_forward(fc7_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, fc6_dst_memory},
            {DNNL_ARG_WEIGHTS, fc7_weights_memory},
            {DNNL_ARG_BIAS, fc7_user_bias_memory},
            {DNNL_ARG_DST, fc7_dst_memory}});

    // fc8 inner product {batch, 4096} (x) {1000, 4096}-> {batch, 1000}
    memory::dims fc8_weights_tz = {1000, 4096};
    memory::dims fc8_bias_tz = {1000};
    memory::dims fc8_dst_tz = {batch, 1000};

    std::vector<float> fc8_weights(product(fc8_weights_tz));
    std::vector<float> fc8_bias(product(fc8_bias_tz));

    // create memory for user data
    auto fc8_user_weights_memory
            = memory({{fc8_weights_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(fc8_weights.data(), fc8_user_weights_memory);
    auto fc8_user_bias_memory = memory({{fc8_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc8_bias.data(), fc8_user_bias_memory);
    auto user_dst_memory = memory({{fc8_dst_tz}, dt::f32, tag::nc}, eng);
    write_to_dnnl_memory(user_dst.data(), user_dst_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto fc8_bias_md = memory::desc({fc8_bias_tz}, dt::f32, tag::any);
    auto fc8_weights_md = memory::desc({fc8_weights_tz}, dt::f32, tag::any);
    auto fc8_dst_md = memory::desc({fc8_dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto fc8_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc7_dst_memory.get_desc(), fc8_weights_md, fc8_bias_md, fc8_dst_md);
    auto fc8_prim_desc = inner_product_forward::primitive_desc(fc8_desc, eng);

    auto fc8_weights_memory = fc8_user_weights_memory;
    if (fc8_prim_desc.weights_desc() != fc8_user_weights_memory.get_desc()) {
        fc8_weights_memory = memory(fc8_prim_desc.weights_desc(), eng);
        reorder(fc8_user_weights_memory, fc8_weights_memory)
                .execute(s, fc8_user_weights_memory, fc8_weights_memory);
    }

    auto fc8_dst_memory = memory(fc8_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(inner_product_forward(fc8_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, fc7_dst_memory},
            {DNNL_ARG_WEIGHTS, fc8_weights_memory},
            {DNNL_ARG_BIAS, fc8_user_bias_memory},
            {DNNL_ARG_DST, fc8_dst_memory}});

    // create reorder between internal and user data if it is needed and
    // add it to net after pooling
    if (fc8_dst_memory != user_dst_memory) {
        net.push_back(reorder(fc8_dst_memory, user_dst_memory));
        net_args.push_back({{DNNL_ARG_FROM, fc8_dst_memory},
                {DNNL_ARG_TO, user_dst_memory}});
    }

    /// @page cnn_inference_f32_cpp
    /// Finally, execute the primitives. For this example, the net is executed
    /// multiple times and each execution is timed individually.
    /// @snippet cnn_inference_f32.cpp Execute model
    //[Execute model]
    for (int j = 0; j < times; ++j) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
    }
    //[Execute model]

    s.wait();
}

void cnn_inference_f32(engine::kind engine_kind) {
    auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    int times = 100;
    simple_net(engine_kind, times);
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                       .count();
    std::cout << "Use time: " << (end - begin) / (times + 0.0)
              << " ms per iteration." << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            cnn_inference_f32, parse_engine_kind(argc, argv));
}
