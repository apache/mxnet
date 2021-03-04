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

/// @example reorder.cpp
/// > Annotated version: @ref reorder_example_cpp
///
/// @page reorder_example_cpp_short
///
/// This C++ API demonstrates how to create and execute a
/// [Reorder](@ref dev_guide_reorder) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes for output scaling.
///
/// @page reorder_example_cpp Reorder Primitive Example
/// @copydetails reorder_example_cpp_short
///
/// @include reorder.cpp

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

void reorder_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 227, // tensor height
            IW = 227; // tensor width

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<int8_t> dst_data(product(src_dims));

    // Initialize src tensor.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create memory descriptors and memory objects for src and dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto dst_md = memory::desc(src_dims, dt::s8, tag::nhwc);

    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Per-channel scales.
    std::vector<float> scales(IC);
    std::generate(scales.begin(), scales.end(), []() {
        static int i = 0;
        return 64 + 5 * i++;
    });

    // Dimension of the dst tensor where the output scales will be applied
    const int ic_dim = 1;

    // Create primitive post-ops (per-channel output scales)
    primitive_attr reorder_attr;
    reorder_attr.set_output_scales(0 | (1 << ic_dim), scales);

    // Create primitive descriptor.
    auto reorder_pd = reorder::primitive_desc(
            engine, src_md, engine, dst_md, reorder_attr);

    // Create the primitive.
    auto reorder_prim = reorder(reorder_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, src_mem});
    reorder_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: reorder with scaled sum.
    reorder_prim.execute(engine_stream, reorder_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            reorder_example, parse_engine_kind(argc, argv));
}
