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

/// @example resampling.cpp
/// > Annotated version: @ref resampling_example_cpp
///
/// @page resampling_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Resampling](@ref dev_guide_resampling) primitive in forward training
/// propagation mode.
///
/// @page resampling_example_cpp Resampling Primitive Example
/// @copydetails resampling_example_cpp_short
///
/// @include resampling.cpp

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

void resampling_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 227, // input tensor height
            IW = 227, // input tensor width
            OH = 350, // output tensor height
            OW = 350; // output tensor width

    // Source (src) and destination (dst) dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims dst_dims = {N, IC, OH, OW};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src tensor.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create memory descriptors and memory objects for src and dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);

    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Create operation descriptor.
    auto resampling_d = resampling_forward::desc(prop_kind::forward_training,
            algorithm::resampling_linear, src_md, dst_md);

    // Create primitive descriptor.
    auto resampling_pd
            = resampling_forward::primitive_desc(resampling_d, engine);

    // Create the primitive.
    auto resampling_prim = resampling_forward(resampling_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> resampling_args;
    resampling_args.insert({DNNL_ARG_SRC, src_mem});
    resampling_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: resampling.
    resampling_prim.execute(engine_stream, resampling_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            resampling_example, parse_engine_kind(argc, argv));
}
