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

/// @example shuffle.cpp
/// > Annotated version: @ref shuffle_example_cpp
///
/// @page shuffle_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Shuffle](@ref dev_guide_shuffle) primitive.
///
/// Key optimizations included in this example:
/// - Shuffle along axis 1 (channels).
///
/// @page shuffle_example_cpp Shuffle Primitive Example
/// @copydetails shuffle_example_cpp_short
///
/// @include shuffle.cpp

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

void shuffle_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 72, // channels
            IH = 227, // tensor height
            IW = 227; // tensor width

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> dst_data(product(src_dims));

    // Initialize src.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Shuffle axis and group size.
    const int shuffle_axis = 1;
    const int group_size = 4;

    // Create memory descriptor and memory objects for src and dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto src_mem = memory(src_md, engine);

    auto dst_mem = memory({src_dims, dt::f32, tag::abcd}, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Create operation descriptor.
    auto shuffle_d = shuffle_forward::desc(
            prop_kind::forward_training, src_md, shuffle_axis, group_size);

    // Create primitive descriptor.
    auto shuffle_pd = shuffle_forward::primitive_desc(shuffle_d, engine);

    // Create the primitive.
    auto shuffle_prim = shuffle_forward(shuffle_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> shuffle_args;
    shuffle_args.insert({DNNL_ARG_SRC, src_mem});
    shuffle_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: shuffle.
    shuffle_prim.execute(engine_stream, shuffle_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            shuffle_example, parse_engine_kind(argc, argv));
}
