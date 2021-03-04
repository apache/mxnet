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

/// @example lstm.cpp
/// > Annotated version: @ref lstm_example_cpp
///
/// @page lstm_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute an
/// [LSTM RNN](@ref dev_guide_rnn) primitive in forward training propagation
/// mode.
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor.
///
/// @page lstm_example_cpp LSTM RNN Primitive Example
/// @copydetails lstm_example_cpp_short
///
/// @include lstm.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void lstm_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 26, // batch size
            T = 6, // time steps
            C = 12, // channels
            G = 4, // gates
            L = 4, // layers
            D = 1; // directions

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {T, N, C};
    memory::dims weights_dims = {L, D, C, G, C};
    memory::dims bias_dims = {L, D, G, C};
    memory::dims dst_dims = {T, N, C};

    // Allocate buffers.
    std::vector<float> src_layer_data(product(src_dims));
    std::vector<float> weights_layer_data(product(weights_dims));
    std::vector<float> weights_iter_data(product(weights_dims));
    std::vector<float> dst_layer_data(product(dst_dims));
    std::vector<float> bias_data(product(bias_dims));

    // Initialize src, weights, and bias tensors.
    std::generate(src_layer_data.begin(), src_layer_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_layer_data.begin(), weights_layer_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(i++);
    });

    // Create memory descriptors and memory objects for src, bias, and dst.
    auto src_layer_md = memory::desc(src_dims, dt::f32, tag::tnc);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ldgo);
    auto dst_layer_md = memory::desc(dst_dims, dt::f32, tag::tnc);

    auto src_layer_mem = memory(src_layer_md, engine);
    auto bias_mem = memory(bias_md, engine);
    auto dst_layer_mem = memory(dst_layer_md, engine);

    // Create memory objects for weights using user's memory layout. In this
    // example, LDIGO is assumed.
    auto user_weights_layer_mem
            = memory({weights_dims, dt::f32, tag::ldigo}, engine);
    auto user_weights_iter_mem
            = memory({weights_dims, dt::f32, tag::ldigo}, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_layer_data.data(), src_layer_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);
    write_to_dnnl_memory(weights_layer_data.data(), user_weights_layer_mem);
    write_to_dnnl_memory(weights_iter_data.data(), user_weights_iter_mem);

    // Create memory descriptors for weights with format_tag::any. This enables
    // the LSTM primitive to choose the optimized memory layout.
    auto lstm_weights_layer_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto lstm_weights_iter_md = memory::desc(weights_dims, dt::f32, tag::any);

    // Optional memory descriptors for recurrent data.
    auto src_iter_md = memory::desc();
    auto src_iter_c_md = memory::desc();
    auto dst_iter_md = memory::desc();
    auto dst_iter_c_md = memory::desc();

    // Create operation descriptor.
    auto lstm_desc = lstm_forward::desc(prop_kind::forward_training,
            rnn_direction::unidirectional_left2right, src_layer_md, src_iter_md,
            src_iter_c_md, lstm_weights_layer_md, lstm_weights_iter_md, bias_md,
            dst_layer_md, dst_iter_md, dst_iter_c_md);

    // Create primitive descriptor.
    auto lstm_pd = lstm_forward::primitive_desc(lstm_desc, engine);

    // For now, assume that the weights memory layout generated by the primitive
    // and the ones provided by the user are identical.
    auto lstm_weights_layer_mem = user_weights_layer_mem;
    auto lstm_weights_iter_mem = user_weights_iter_mem;

    // Reorder the data in case the weights memory layout generated by the
    // primitive and the one provided by the user are different. In this case,
    // we create additional memory objects with internal buffers that will
    // contain the reordered data.
    if (lstm_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
        lstm_weights_layer_mem = memory(lstm_pd.weights_desc(), engine);
        reorder(user_weights_layer_mem, lstm_weights_layer_mem)
                .execute(engine_stream, user_weights_layer_mem,
                        lstm_weights_layer_mem);
    }

    if (lstm_pd.weights_iter_desc() != user_weights_iter_mem.get_desc()) {
        lstm_weights_iter_mem = memory(lstm_pd.weights_iter_desc(), engine);
        reorder(user_weights_iter_mem, lstm_weights_iter_mem)
                .execute(engine_stream, user_weights_iter_mem,
                        lstm_weights_iter_mem);
    }

    // Create the memory objects from the primitive descriptor. A workspace is
    // also required for LSTM.
    // NOTE: Here, the workspace is required for later usage in backward
    // propagation mode.
    auto src_iter_mem = memory(lstm_pd.src_iter_desc(), engine);
    auto src_iter_c_mem = memory(lstm_pd.src_iter_c_desc(), engine);
    auto weights_iter_mem = memory(lstm_pd.weights_iter_desc(), engine);
    auto dst_iter_mem = memory(lstm_pd.dst_iter_desc(), engine);
    auto dst_iter_c_mem = memory(lstm_pd.dst_iter_c_desc(), engine);
    auto workspace_mem = memory(lstm_pd.workspace_desc(), engine);

    // Create the primitive.
    auto lstm_prim = lstm_forward(lstm_pd);

    // Primitive arguments
    std::unordered_map<int, memory> lstm_args;
    lstm_args.insert({DNNL_ARG_SRC_LAYER, src_layer_mem});
    lstm_args.insert({DNNL_ARG_WEIGHTS_LAYER, lstm_weights_layer_mem});
    lstm_args.insert({DNNL_ARG_WEIGHTS_ITER, lstm_weights_iter_mem});
    lstm_args.insert({DNNL_ARG_BIAS, bias_mem});
    lstm_args.insert({DNNL_ARG_DST_LAYER, dst_layer_mem});
    lstm_args.insert({DNNL_ARG_SRC_ITER, src_iter_mem});
    lstm_args.insert({DNNL_ARG_SRC_ITER_C, src_iter_c_mem});
    lstm_args.insert({DNNL_ARG_DST_ITER, dst_iter_mem});
    lstm_args.insert({DNNL_ARG_DST_ITER_C, dst_iter_c_mem});
    lstm_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

    // Primitive execution: LSTM.
    lstm_prim.execute(engine_stream, lstm_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_layer_data.data(), dst_layer_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(lstm_example, parse_engine_kind(argc, argv));
}
