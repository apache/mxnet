/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

/// @example rnn_training_f32.cpp
/// @copybrief rnn_training_f32_cpp
///
/// @page rnn_training_f32_cpp RNN f32 training example
/// This C++ API example demonstrates how to build GNMT model training.
///
/// @include rnn_training_f32.cpp

#include <cstring>
#include <math.h>
#include <numeric>
#include <utility>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

// User input is:
//     N0 sequences of length T0
const int N0 = 1 + rand() % 31;
//     N1 sequences of length T1
const int N1 = 1 + rand() % 31;
// Assume T0 > T1
const int T0 = 31 + 1 + rand() % 31;
const int T1 = 1 + rand() % 31;

// Memory required to hold it: N0 * T0 + N1 * T1
// However it is possible to have these coming
// as padded chunks in larger memory:
//      e.g. (N0 + N1) * T0
// We don't need to compact the data before processing,
// we can address the chunks via sub-memory and
// process the data via two RNN primitives:
//     of time lengths T1 and T0 - T1.
// The leftmost primitive will process N0 + N1 subsequences of length T1
// The rightmost primitive will process remaining N0 subsequences
// of T0 - T1 length
const int leftmost_batch = N0 + N1;
const int rightmost_batch = N0;

const int leftmost_seq_length = T1;
const int rightmost_seq_length = T0 - T1;

// Number of channels
const int common_feature_size = 1024;

// RNN primitive characteristics
const int common_n_layers = 1;
const int lstm_n_gates = 4;

void simple_net(engine::kind engine_kind) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    auto eng = engine(engine_kind, 0);
    stream s(eng);

    bool is_training = true;
    auto fwd_inf_train = is_training ? prop_kind::forward_training
                                     : prop_kind::forward_inference;

    std::vector<primitive> fwd_net;
    std::vector<primitive> bwd_net;

    // Input tensor holds two batches with different sequence lengths.
    // Shorter sequences are padded
    memory::dims net_src_dims = {
            T0, // time, maximum sequence length
            N0 + N1, // n, total batch size
            common_feature_size // c, common number of channels
    };

    // Two RNN primitives for different sequence lengths,
    // one unidirectional layer, LSTM-based
    memory::dims leftmost_src_layer_dims = {
            leftmost_seq_length, // time
            leftmost_batch, // n
            common_feature_size // c
    };
    memory::dims rightmost_src_layer_dims = {
            rightmost_seq_length, // time
            rightmost_batch, // n
            common_feature_size // c
    };
    memory::dims common_weights_layer_dims = {
            common_n_layers, // layers
            1, // directions
            common_feature_size, // input feature size
            lstm_n_gates, // gates number
            common_feature_size // output feature size
    };
    memory::dims common_weights_iter_dims = {
            common_n_layers, // layers
            1, // directions
            common_feature_size, // input feature size
            lstm_n_gates, // gates number
            common_feature_size // output feature size
    };
    memory::dims common_bias_dims = {
            common_n_layers, // layers
            1, // directions
            lstm_n_gates, // gates number
            common_feature_size // output feature size
    };
    memory::dims leftmost_dst_layer_dims = {
            leftmost_seq_length, // time
            leftmost_batch, // n
            common_feature_size // c
    };
    memory::dims rightmost_dst_layer_dims = {
            rightmost_seq_length, // time
            rightmost_batch, // n
            common_feature_size // c
    };

    // leftmost primitive passes its states to the next RNN iteration
    // so it needs dst_iter parameter.
    //
    // rightmost primitive will consume these as src_iter and will access the
    // memory via a sub-memory because it will have different batch dimension.
    // We have arranged our primitives so that
    // leftmost_batch >= rightmost_batch, and so the rightmost data will fit
    // into the memory allocated for the leftmost.
    memory::dims leftmost_dst_iter_dims = {
            common_n_layers, // layers
            1, // directions
            leftmost_batch, // n
            common_feature_size // c
    };
    memory::dims leftmost_dst_iter_c_dims = {
            common_n_layers, // layers
            1, // directions
            leftmost_batch, // n
            common_feature_size // c
    };
    memory::dims rightmost_src_iter_dims = {
            common_n_layers, // layers
            1, // directions
            rightmost_batch, // n
            common_feature_size // c
    };
    memory::dims rightmost_src_iter_c_dims = {
            common_n_layers, // layers
            1, // directions
            rightmost_batch, // n
            common_feature_size // c
    };

    // multiplication of tensor dimensions
    auto tz_volume = [=](memory::dims tz_dims) {
        return std::accumulate(tz_dims.begin(), tz_dims.end(), (memory::dim)1,
                std::multiplies<memory::dim>());
    };

    // Create auxillary f32 memory descriptor
    // based on user- supplied dimensions and layout.
    auto formatted_md
            = [=](const memory::dims &dimensions, memory::format_tag layout) {
                  return memory::desc {{dimensions}, dt::f32, layout};
              };
    // Create auxillary generic f32 memory descriptor
    // based on supplied dimensions, with format_tag::any.
    auto generic_md = [=](const memory::dims &dimensions) {
        return formatted_md(dimensions, tag::any);
    };

    //
    // I/O memory, coming from user
    //

    // Net input
    std::vector<float> net_src(tz_volume(net_src_dims), 1.0f);
    // NOTE: in this example we study input sequences with variable batch
    // dimension, which get processed by two separate RNN primitives, thus
    // the destination memory for the two will have different shapes: batch
    // is the second dimension currently: see format_tag::tnc.
    // We are not copying the output to some common user provided memory as we
    // suggest that the user should rather keep the two output memories separate
    // throughout the whole topology and only reorder to something else as
    // needed.
    // So there's no common net_dst, but there are two destinations instead:
    //    leftmost_dst_layer_memory
    //    rightmost_dst_layer_memory

    // Memory for the user allocated memory
    // Suppose user data is in tnc format.
    auto net_src_memory
            = dnnl::memory({{net_src_dims}, dt::f32, tag::tnc}, eng);
    write_to_dnnl_memory(net_src.data(), net_src_memory);
    // src_layer memory of the leftmost and rightmost RNN primitives
    // are accessed through the respective sub-memories in larger memory.
    // View primitives compute the strides to accommodate for padding.
    auto user_leftmost_src_layer_md = net_src_memory.get_desc().submemory_desc(
            leftmost_src_layer_dims, {0, 0, 0}); // t, n, c offsets
    auto user_rightmost_src_layer_md
            = net_src_memory.get_desc().submemory_desc(rightmost_src_layer_dims,
                    {leftmost_seq_length, 0, 0}); // t, n, c offsets
    auto leftmost_src_layer_memory = net_src_memory;
    auto rightmost_src_layer_memory = net_src_memory;

    // Other user provided memory arrays, descriptors and primitives with the
    // data layouts chosen by user. We'll have to reorder if RNN
    // primitive prefers it in a different format.
    std::vector<float> user_common_weights_layer(
            tz_volume(common_weights_layer_dims), 1.0f);
    auto user_common_weights_layer_memory = dnnl::memory(
            {common_weights_layer_dims, dt::f32, tag::ldigo}, eng);
    write_to_dnnl_memory(
            user_common_weights_layer.data(), user_common_weights_layer_memory);

    std::vector<float> user_common_weights_iter(
            tz_volume(common_weights_iter_dims), 1.0f);
    auto user_common_weights_iter_memory = dnnl::memory(
            {{common_weights_iter_dims}, dt::f32, tag::ldigo}, eng);
    write_to_dnnl_memory(
            user_common_weights_layer.data(), user_common_weights_iter_memory);

    std::vector<float> user_common_bias(tz_volume(common_bias_dims), 1.0f);
    auto user_common_bias_memory
            = dnnl::memory({{common_bias_dims}, dt::f32, tag::ldgo}, eng);
    write_to_dnnl_memory(user_common_bias.data(), user_common_bias_memory);

    std::vector<float> user_leftmost_dst_layer(
            tz_volume(leftmost_dst_layer_dims), 1.0f);
    auto user_leftmost_dst_layer_memory
            = dnnl::memory({{leftmost_dst_layer_dims}, dt::f32, tag::tnc}, eng);
    write_to_dnnl_memory(
            user_leftmost_dst_layer.data(), user_leftmost_dst_layer_memory);

    std::vector<float> user_rightmost_dst_layer(
            tz_volume(rightmost_dst_layer_dims), 1.0f);
    auto user_rightmost_dst_layer_memory = dnnl::memory(
            {{rightmost_dst_layer_dims}, dt::f32, tag::tnc}, eng);
    write_to_dnnl_memory(
            user_rightmost_dst_layer.data(), user_rightmost_dst_layer_memory);

    // Describe layer, forward pass, leftmost primitive.
    // There are no primitives to the left from here,
    // so src_iter_desc needs to be zero memory desc
    lstm_forward::desc leftmost_layer_desc(fwd_inf_train, // aprop_kind
            rnn_direction::unidirectional_left2right, // direction
            user_leftmost_src_layer_md, // src_layer_desc
            memory::desc(), // src_iter_desc
            memory::desc(), // src_iter_c_desc
            generic_md(common_weights_layer_dims), // weights_layer_desc
            generic_md(common_weights_iter_dims), // weights_iter_desc
            generic_md(common_bias_dims), // bias_desc
            formatted_md(leftmost_dst_layer_dims, tag::tnc), // dst_layer_desc
            generic_md(leftmost_dst_iter_dims), // dst_iter_desc
            generic_md(leftmost_dst_iter_c_dims) // dst_iter_c_desc
    );
    // Describe primitive
    auto leftmost_prim_desc
            = dnnl::lstm_forward::primitive_desc(leftmost_layer_desc, eng);

    //
    // Need to connect leftmost and rightmost via "iter" parameters.
    // We allocate memory here based on the shapes provided by RNN primitive.
    //
    auto leftmost_dst_iter_memory
            = dnnl::memory(leftmost_prim_desc.dst_iter_desc(), eng);
    auto leftmost_dst_iter_c_memory
            = dnnl::memory(leftmost_prim_desc.dst_iter_c_desc(), eng);

    // rightmost src_iter will be a sub-memory of dst_iter of leftmost
    auto rightmost_src_iter_md
            = leftmost_dst_iter_memory.get_desc().submemory_desc(
                    rightmost_src_iter_dims,
                    {0, 0, 0, 0}); // l, d, n, c offsets
    auto rightmost_src_iter_memory = leftmost_dst_iter_memory;

    auto rightmost_src_iter_c_md
            = leftmost_dst_iter_c_memory.get_desc().submemory_desc(
                    rightmost_src_iter_c_dims,
                    {0, 0, 0, 0}); // l, d, n, c offsets
    auto rightmost_src_iter_c_memory = leftmost_dst_iter_c_memory;

    // Now rightmost primitive
    // There are no primitives to the right from here,
    // so dst_iter_desc is explicit zero memory desc
    lstm_forward::desc rightmost_layer_desc(fwd_inf_train, // aprop_kind
            rnn_direction::unidirectional_left2right, // direction
            user_rightmost_src_layer_md, // src_layer_desc
            rightmost_src_iter_md, // src_iter_desc
            rightmost_src_iter_c_md, // src_iter_c_desc
            generic_md(common_weights_layer_dims), // weights_layer_desc
            generic_md(common_weights_iter_dims), // weights_iter_desc
            generic_md(common_bias_dims), // bias_desc
            formatted_md(rightmost_dst_layer_dims, tag::tnc), // dst_layer_desc
            memory::desc(), // dst_iter_desc
            memory::desc() // dst_iter_c_desc
    );
    auto rightmost_prim_desc
            = lstm_forward::primitive_desc(rightmost_layer_desc, eng);

    //
    // Weights and biases, layer memory
    // Same layout should work across the layer, no reorders
    // needed between leftmost and rigthmost, only reordering
    // user memory to the RNN-friendly shapes.
    //

    auto common_weights_layer_memory = user_common_weights_layer_memory;
    if (leftmost_prim_desc.weights_layer_desc()
            != common_weights_layer_memory.get_desc()) {
        common_weights_layer_memory
                = dnnl::memory(leftmost_prim_desc.weights_layer_desc(), eng);
        reorder(user_common_weights_layer_memory, common_weights_layer_memory)
                .execute(s, user_common_weights_layer_memory,
                        common_weights_layer_memory);
    }

    auto common_weights_iter_memory = user_common_weights_iter_memory;
    if (leftmost_prim_desc.weights_iter_desc()
            != common_weights_iter_memory.get_desc()) {
        common_weights_iter_memory
                = dnnl::memory(leftmost_prim_desc.weights_iter_desc(), eng);
        reorder(user_common_weights_iter_memory, common_weights_iter_memory)
                .execute(s, user_common_weights_iter_memory,
                        common_weights_iter_memory);
    }

    auto common_bias_memory = user_common_bias_memory;
    if (leftmost_prim_desc.bias_desc() != common_bias_memory.get_desc()) {
        common_bias_memory = dnnl::memory(leftmost_prim_desc.bias_desc(), eng);
        reorder(user_common_bias_memory, common_bias_memory)
                .execute(s, user_common_bias_memory, common_bias_memory);
    }

    //
    // Destination layer memory
    //

    auto leftmost_dst_layer_memory = user_leftmost_dst_layer_memory;
    if (leftmost_prim_desc.dst_layer_desc()
            != leftmost_dst_layer_memory.get_desc()) {
        leftmost_dst_layer_memory
                = dnnl::memory(leftmost_prim_desc.dst_layer_desc(), eng);
        reorder(user_leftmost_dst_layer_memory, leftmost_dst_layer_memory)
                .execute(s, user_leftmost_dst_layer_memory,
                        leftmost_dst_layer_memory);
    }

    auto rightmost_dst_layer_memory = user_rightmost_dst_layer_memory;
    if (rightmost_prim_desc.dst_layer_desc()
            != rightmost_dst_layer_memory.get_desc()) {
        rightmost_dst_layer_memory
                = dnnl::memory(rightmost_prim_desc.dst_layer_desc(), eng);
        reorder(user_rightmost_dst_layer_memory, rightmost_dst_layer_memory)
                .execute(s, user_rightmost_dst_layer_memory,
                        rightmost_dst_layer_memory);
    }

    // We also create workspace memory based on the information from
    // the workspace_primitive_desc(). This is needed for internal
    // communication between forward and backward primitives during
    // training.
    auto create_ws = [=](dnnl::lstm_forward::primitive_desc &pd) {
        return dnnl::memory(pd.workspace_desc(), eng);
    };
    auto leftmost_workspace_memory = create_ws(leftmost_prim_desc);
    auto rightmost_workspace_memory = create_ws(rightmost_prim_desc);

    // Construct the RNN primitive objects
    lstm_forward leftmost_layer(leftmost_prim_desc);
    leftmost_layer.execute(s,
            {{DNNL_ARG_SRC_LAYER, leftmost_src_layer_memory},
                    {DNNL_ARG_WEIGHTS_LAYER, common_weights_layer_memory},
                    {DNNL_ARG_WEIGHTS_ITER, common_weights_iter_memory},
                    {DNNL_ARG_BIAS, common_bias_memory},
                    {DNNL_ARG_DST_LAYER, leftmost_dst_layer_memory},
                    {DNNL_ARG_DST_ITER, leftmost_dst_iter_memory},
                    {DNNL_ARG_DST_ITER_C, leftmost_dst_iter_c_memory},
                    {DNNL_ARG_WORKSPACE, leftmost_workspace_memory}});

    lstm_forward rightmost_layer(rightmost_prim_desc);
    rightmost_layer.execute(s,
            {{DNNL_ARG_SRC_LAYER, rightmost_src_layer_memory},
                    {DNNL_ARG_SRC_ITER, rightmost_src_iter_memory},
                    {DNNL_ARG_SRC_ITER_C, rightmost_src_iter_c_memory},
                    {DNNL_ARG_WEIGHTS_LAYER, common_weights_layer_memory},
                    {DNNL_ARG_WEIGHTS_ITER, common_weights_iter_memory},
                    {DNNL_ARG_BIAS, common_bias_memory},
                    {DNNL_ARG_DST_LAYER, rightmost_dst_layer_memory},
                    {DNNL_ARG_WORKSPACE, rightmost_workspace_memory}});

    // No backward pass for inference
    if (!is_training) return;

    //
    // Backward primitives will reuse memory from forward
    // and allocate/describe specifics here. Only relevant for training.
    //

    // User-provided memory for backward by data output
    std::vector<float> net_diff_src(tz_volume(net_src_dims), 1.0f);
    auto net_diff_src_memory
            = dnnl::memory(formatted_md(net_src_dims, tag::tnc), eng);
    write_to_dnnl_memory(net_diff_src.data(), net_diff_src_memory);

    // diff_src follows the same layout we have for net_src
    auto user_leftmost_diff_src_layer_md
            = net_diff_src_memory.get_desc().submemory_desc(
                    leftmost_src_layer_dims, {0, 0, 0}); // t, n, c offsets
    auto user_rightmost_diff_src_layer_md
            = net_diff_src_memory.get_desc().submemory_desc(
                    rightmost_src_layer_dims,
                    {leftmost_seq_length, 0, 0}); // t, n, c offsets
    auto leftmost_diff_src_layer_memory = net_diff_src_memory;
    auto rightmost_diff_src_layer_memory = net_diff_src_memory;

    // User-provided memory for backpropagation by weights
    std::vector<float> user_common_diff_weights_layer(
            tz_volume(common_weights_layer_dims), 1.0f);
    auto user_common_diff_weights_layer_memory = dnnl::memory(
            formatted_md(common_weights_layer_dims, tag::ldigo), eng);
    write_to_dnnl_memory(user_common_diff_weights_layer.data(),
            user_common_diff_weights_layer_memory);

    std::vector<float> user_common_diff_bias(tz_volume(common_bias_dims), 1.0f);
    auto user_common_diff_bias_memory
            = dnnl::memory(formatted_md(common_bias_dims, tag::ldgo), eng);
    write_to_dnnl_memory(
            user_common_diff_bias.data(), user_common_diff_bias_memory);

    // User-provided input to the backward primitive.
    // To be updated by the user after forward pass using some cost function.
    memory::dims net_diff_dst_dims = {
            T0, // time
            N0 + N1, // n
            common_feature_size // c
    };
    // Suppose user data is in tnc format.
    std::vector<float> net_diff_dst(tz_volume(net_diff_dst_dims), 1.0f);
    auto net_diff_dst_memory
            = dnnl::memory(formatted_md(net_diff_dst_dims, tag::tnc), eng);
    write_to_dnnl_memory(net_diff_dst.data(), net_diff_dst_memory);
    // diff_dst_layer memory of the leftmost and rightmost RNN primitives
    // are accessed through the respective sub-memory in larger memory.
    // View primitives compute the strides to accommodate for padding.
    auto user_leftmost_diff_dst_layer_md
            = net_diff_dst_memory.get_desc().submemory_desc(
                    leftmost_dst_layer_dims, {0, 0, 0}); // t, n, c offsets
    auto user_rightmost_diff_dst_layer_md
            = net_diff_dst_memory.get_desc().submemory_desc(
                    rightmost_dst_layer_dims,
                    {leftmost_seq_length, 0, 0}); // t, n, c offsets
    auto leftmost_diff_dst_layer_memory = net_diff_dst_memory;
    auto rightmost_diff_dst_layer_memory = net_diff_dst_memory;

    // Backward leftmost primitive descriptor
    lstm_backward::desc leftmost_layer_bwd_desc(
            prop_kind::backward, // aprop_kind
            rnn_direction::unidirectional_left2right, // direction
            user_leftmost_src_layer_md, // src_layer_desc
            memory::desc(), // src_iter_desc
            memory::desc(), // src_iter_c_desc
            generic_md(common_weights_layer_dims), // weights_layer_desc
            generic_md(common_weights_iter_dims), // weights_iter_desc
            generic_md(common_bias_dims), // bias_desc
            formatted_md(leftmost_dst_layer_dims, tag::tnc), // dst_layer_desc
            generic_md(leftmost_dst_iter_dims), // dst_iter_desc
            generic_md(leftmost_dst_iter_c_dims), // dst_iter_c_desc
            user_leftmost_diff_src_layer_md, // diff_src_layer_desc
            memory::desc(), // diff_src_iter_desc
            memory::desc(), // diff_src_iter_c_desc
            generic_md(common_weights_layer_dims), // diff_weights_layer_desc
            generic_md(common_weights_iter_dims), // diff_weights_iter_desc
            generic_md(common_bias_dims), // diff_bias_desc
            user_leftmost_diff_dst_layer_md, // diff_dst_layer_desc
            generic_md(leftmost_dst_iter_dims), // diff_dst_iter_desc
            generic_md(leftmost_dst_iter_c_dims) // diff_dst_iter_c_desc
    );
    auto leftmost_bwd_prim_desc = lstm_backward::primitive_desc(
            leftmost_layer_bwd_desc, eng, leftmost_prim_desc);

    // As the batch dimensions are different between leftmost and rightmost
    // we need to use a sub-memory. rightmost needs less memory, so it will
    // be a sub-memory of leftmost.
    auto leftmost_diff_dst_iter_memory
            = dnnl::memory(leftmost_bwd_prim_desc.diff_dst_iter_desc(), eng);
    auto leftmost_diff_dst_iter_c_memory
            = dnnl::memory(leftmost_bwd_prim_desc.diff_dst_iter_c_desc(), eng);

    auto rightmost_diff_src_iter_md
            = leftmost_diff_dst_iter_memory.get_desc().submemory_desc(
                    rightmost_src_iter_dims,
                    {0, 0, 0, 0}); // l, d, n, c offsets
    auto rightmost_diff_src_iter_memory = leftmost_diff_dst_iter_memory;

    auto rightmost_diff_src_iter_c_md
            = leftmost_diff_dst_iter_c_memory.get_desc().submemory_desc(
                    rightmost_src_iter_c_dims,
                    {0, 0, 0, 0}); // l, d, n, c offsets
    auto rightmost_diff_src_iter_c_memory = leftmost_diff_dst_iter_c_memory;

    // Backward rightmost primitive descriptor
    lstm_backward::desc rightmost_layer_bwd_desc(
            prop_kind::backward, // aprop_kind
            rnn_direction::unidirectional_left2right, // direction
            user_rightmost_src_layer_md, // src_layer_desc
            generic_md(rightmost_src_iter_dims), // src_iter_desc
            generic_md(rightmost_src_iter_c_dims), // src_iter_c_desc
            generic_md(common_weights_layer_dims), // weights_layer_desc
            generic_md(common_weights_iter_dims), // weights_iter_desc
            generic_md(common_bias_dims), // bias_desc
            formatted_md(rightmost_dst_layer_dims, tag::tnc), // dst_layer_desc
            memory::desc(), // dst_iter_desc
            memory::desc(), // dst_iter_c_desc
            user_rightmost_diff_src_layer_md, // diff_src_layer_desc
            rightmost_diff_src_iter_md, // diff_src_iter_desc
            rightmost_diff_src_iter_c_md, // diff_src_iter_c_desc
            generic_md(common_weights_layer_dims), // diff_weights_layer_desc
            generic_md(common_weights_iter_dims), // diff_weights_iter_desc
            generic_md(common_bias_dims), // diff_bias_desc
            user_rightmost_diff_dst_layer_md, // diff_dst_layer_desc
            memory::desc(), // diff_dst_iter_desc
            memory::desc() // diff_dst_iter_c_desc
    );
    auto rightmost_bwd_prim_desc = lstm_backward::primitive_desc(
            rightmost_layer_bwd_desc, eng, rightmost_prim_desc);

    //
    // Memory for backward pass
    //

    // src layer uses the same memory as forward
    auto leftmost_src_layer_bwd_memory = leftmost_src_layer_memory;
    auto rightmost_src_layer_bwd_memory = rightmost_src_layer_memory;

    // Memory for weights and biases for backward pass
    // Try to use the same memory between forward and backward, but
    // sometimes reorders are needed.
    auto common_weights_layer_bwd_memory = common_weights_layer_memory;
    if (leftmost_bwd_prim_desc.weights_layer_desc()
            != leftmost_prim_desc.weights_layer_desc()) {
        common_weights_layer_bwd_memory
                = memory(leftmost_bwd_prim_desc.weights_layer_desc(), eng);
        reorder(common_weights_layer_memory, common_weights_layer_bwd_memory)
                .execute(s, common_weights_layer_memory,
                        common_weights_layer_bwd_memory);
    }

    auto common_weights_iter_bwd_memory = common_weights_iter_memory;
    if (leftmost_bwd_prim_desc.weights_iter_desc()
            != leftmost_prim_desc.weights_iter_desc()) {
        common_weights_iter_bwd_memory
                = memory(leftmost_bwd_prim_desc.weights_iter_desc(), eng);
        reorder(common_weights_iter_memory, common_weights_iter_bwd_memory)
                .execute(s, common_weights_iter_memory,
                        common_weights_iter_bwd_memory);
    }

    auto common_bias_bwd_memory = common_bias_memory;
    if (leftmost_bwd_prim_desc.bias_desc() != common_bias_memory.get_desc()) {
        common_bias_bwd_memory
                = dnnl::memory(leftmost_bwd_prim_desc.bias_desc(), eng);
        reorder(common_bias_memory, common_bias_bwd_memory)
                .execute(s, common_bias_memory, common_bias_bwd_memory);
    }

    // diff_weights and biases
    auto common_diff_weights_layer_memory
            = user_common_diff_weights_layer_memory;
    auto reorder_common_diff_weights_layer = false;
    if (leftmost_bwd_prim_desc.diff_weights_layer_desc()
            != common_diff_weights_layer_memory.get_desc()) {
        common_diff_weights_layer_memory = dnnl::memory(
                leftmost_bwd_prim_desc.diff_weights_layer_desc(), eng);
        reorder_common_diff_weights_layer = true;
    }

    auto common_diff_bias_memory = user_common_diff_bias_memory;
    auto reorder_common_diff_bias = false;
    if (leftmost_bwd_prim_desc.diff_bias_desc()
            != common_diff_bias_memory.get_desc()) {
        common_diff_bias_memory
                = dnnl::memory(leftmost_bwd_prim_desc.diff_bias_desc(), eng);
        reorder_common_diff_bias = true;
    }

    // dst_layer memory for backward pass
    auto leftmost_dst_layer_bwd_memory = leftmost_dst_layer_memory;
    if (leftmost_bwd_prim_desc.dst_layer_desc()
            != leftmost_dst_layer_bwd_memory.get_desc()) {
        leftmost_dst_layer_bwd_memory
                = dnnl::memory(leftmost_bwd_prim_desc.dst_layer_desc(), eng);
        reorder(leftmost_dst_layer_memory, leftmost_dst_layer_bwd_memory)
                .execute(s, leftmost_dst_layer_memory,
                        leftmost_dst_layer_bwd_memory);
    }

    auto rightmost_dst_layer_bwd_memory = rightmost_dst_layer_memory;
    if (rightmost_bwd_prim_desc.dst_layer_desc()
            != rightmost_dst_layer_bwd_memory.get_desc()) {
        rightmost_dst_layer_bwd_memory
                = dnnl::memory(rightmost_bwd_prim_desc.dst_layer_desc(), eng);
        reorder(rightmost_dst_layer_memory, rightmost_dst_layer_bwd_memory)
                .execute(s, rightmost_dst_layer_memory,
                        rightmost_dst_layer_bwd_memory);
    }

    // Similar to forward, the backward primitives are connected
    // via "iter" parameters.
    auto common_diff_weights_iter_memory = dnnl::memory(
            leftmost_bwd_prim_desc.diff_weights_iter_desc(), eng);

    auto leftmost_dst_iter_bwd_memory = leftmost_dst_iter_memory;
    if (leftmost_bwd_prim_desc.dst_iter_desc()
            != leftmost_dst_iter_bwd_memory.get_desc()) {
        leftmost_dst_iter_bwd_memory
                = dnnl::memory(leftmost_bwd_prim_desc.dst_iter_desc(), eng);
        reorder(leftmost_dst_iter_memory, leftmost_dst_iter_bwd_memory)
                .execute(s, leftmost_dst_iter_memory,
                        leftmost_dst_iter_bwd_memory);
    }

    auto leftmost_dst_iter_c_bwd_memory = leftmost_dst_iter_c_memory;
    if (leftmost_bwd_prim_desc.dst_iter_c_desc()
            != leftmost_dst_iter_c_bwd_memory.get_desc()) {
        leftmost_dst_iter_c_bwd_memory
                = dnnl::memory(leftmost_bwd_prim_desc.dst_iter_c_desc(), eng);
        reorder(leftmost_dst_iter_c_memory, leftmost_dst_iter_c_bwd_memory)
                .execute(s, leftmost_dst_iter_c_memory,
                        leftmost_dst_iter_c_bwd_memory);
    }

    // Construct the RNN primitive objects for backward
    lstm_backward rightmost_layer_bwd(rightmost_bwd_prim_desc);
    rightmost_layer_bwd.execute(s,
            {{DNNL_ARG_SRC_LAYER, rightmost_src_layer_bwd_memory},
                    {DNNL_ARG_SRC_ITER, rightmost_src_iter_memory},
                    {DNNL_ARG_SRC_ITER_C, rightmost_src_iter_c_memory},
                    {DNNL_ARG_WEIGHTS_LAYER, common_weights_layer_bwd_memory},
                    {DNNL_ARG_WEIGHTS_ITER, common_weights_iter_bwd_memory},
                    {DNNL_ARG_BIAS, common_bias_bwd_memory},
                    {DNNL_ARG_DST_LAYER, rightmost_dst_layer_bwd_memory},
                    {DNNL_ARG_DIFF_SRC_LAYER, rightmost_diff_src_layer_memory},
                    {DNNL_ARG_DIFF_SRC_ITER, rightmost_diff_src_iter_memory},
                    {DNNL_ARG_DIFF_SRC_ITER_C,
                            rightmost_diff_src_iter_c_memory},
                    {DNNL_ARG_DIFF_WEIGHTS_LAYER,
                            common_diff_weights_layer_memory},
                    {DNNL_ARG_DIFF_WEIGHTS_ITER,
                            common_diff_weights_iter_memory},
                    {DNNL_ARG_DIFF_BIAS, common_diff_bias_memory},
                    {DNNL_ARG_DIFF_DST_LAYER, rightmost_diff_dst_layer_memory},
                    {DNNL_ARG_WORKSPACE, rightmost_workspace_memory}});

    lstm_backward leftmost_layer_bwd(leftmost_bwd_prim_desc);
    leftmost_layer_bwd.execute(s,
            {{DNNL_ARG_SRC_LAYER, leftmost_src_layer_bwd_memory},
                    {DNNL_ARG_WEIGHTS_LAYER, common_weights_layer_bwd_memory},
                    {DNNL_ARG_WEIGHTS_ITER, common_weights_iter_bwd_memory},
                    {DNNL_ARG_BIAS, common_bias_bwd_memory},
                    {DNNL_ARG_DST_LAYER, leftmost_dst_layer_bwd_memory},
                    {DNNL_ARG_DST_ITER, leftmost_dst_iter_bwd_memory},
                    {DNNL_ARG_DST_ITER_C, leftmost_dst_iter_c_bwd_memory},
                    {DNNL_ARG_DIFF_SRC_LAYER, leftmost_diff_src_layer_memory},
                    {DNNL_ARG_DIFF_WEIGHTS_LAYER,
                            common_diff_weights_layer_memory},
                    {DNNL_ARG_DIFF_WEIGHTS_ITER,
                            common_diff_weights_iter_memory},
                    {DNNL_ARG_DIFF_BIAS, common_diff_bias_memory},
                    {DNNL_ARG_DIFF_DST_LAYER, leftmost_diff_dst_layer_memory},
                    {DNNL_ARG_DIFF_DST_ITER, leftmost_diff_dst_iter_memory},
                    {DNNL_ARG_DIFF_DST_ITER_C, leftmost_diff_dst_iter_c_memory},
                    {DNNL_ARG_WORKSPACE, leftmost_workspace_memory}});
    if (reorder_common_diff_weights_layer) {
        reorder(common_diff_weights_layer_memory,
                user_common_diff_weights_layer_memory)
                .execute(s, common_diff_weights_layer_memory,
                        user_common_diff_weights_layer_memory);
    }

    if (reorder_common_diff_bias) {
        reorder(common_diff_bias_memory, user_common_diff_bias_memory)
                .execute(s, common_diff_bias_memory,
                        user_common_diff_bias_memory);
    }

    //
    // User updates weights and bias using diffs
    //

    s.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
}
