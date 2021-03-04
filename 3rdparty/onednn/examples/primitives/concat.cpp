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

/// @example concat.cpp
/// > Annotated version: @ref concat_example_cpp
///
/// @page concat_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Concat](@ref dev_guide_concat) primitive.
///
/// Key optimizations included in this example:
/// - Identical source (src) memory formats.
/// - Creation of optimized memory format for destination (dst) from the
///   primitive descriptor
///
/// @page concat_example_cpp Concat Primitive Example
/// @copydetails concat_example_cpp_short
///
/// @include concat.cpp

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

void concat_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 120, // tensor height
            IW = 120; // tensor width

    // Number of source (src) tensors.
    const int num_src = 10;

    // Concatenation axis.
    const int axis = 1;

    // src tensors dimensions
    memory::dims src_dims = {N, IC, IH, IW};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));

    // Initialize src.
    // NOTE: In this example, the same src memory buffer is used to demonstrate
    // concatenation for simplicity
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create a memory descriptor and memory object for each src tensor.
    std::vector<memory::desc> src_mds;
    std::vector<memory> src_mems;

    for (int n = 0; n < num_src; ++n) {
        auto md = memory::desc(src_dims, dt::f32, tag::nchw);
        auto mem = memory(md, engine);

        // Write data to memory object's handle.
        write_to_dnnl_memory(src_data.data(), mem);

        src_mds.push_back(md);
        src_mems.push_back(mem);
    }

    // Create primitive descriptor.
    auto concat_pd = concat::primitive_desc(axis, src_mds, engine);

    // Create destination (dst) memory object using the memory descriptor
    // created by the primitive.
    auto dst_mem = memory(concat_pd.dst_desc(), engine);

    // Create the primitive.
    auto concat_prim = concat(concat_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> concat_args;
    for (int n = 0; n < num_src; ++n)
        concat_args.insert({DNNL_ARG_MULTIPLE_SRC + n, src_mems[n]});
    concat_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: concatenation.
    concat_prim.execute(engine_stream, concat_args);

    // Wait for the computation to finalize.
    engine_stream.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors(concat_example, parse_engine_kind(argc, argv));
}
