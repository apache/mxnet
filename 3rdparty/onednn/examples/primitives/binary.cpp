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

/// @example binary.cpp
/// > Annotated version: @ref binary_example_cpp
///
/// @page binary_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Binary](@ref dev_guide_binary) primitive.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Primitive attributes with fused post-ops.
///
/// @page binary_example_cpp Binary Primitive Example
/// @copydetails binary_example_cpp_short
///
/// @include binary.cpp

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

void binary_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 150, // tensor height
            IW = 150; // tensor width

    // Source (src_0 and src_1) and destination (dst) tensors dimensions.
    memory::dims src_0_dims = {N, IC, IH, IW};
    memory::dims src_1_dims = {N, IC, IH, 1};

    // Allocate buffers.
    std::vector<float> src_0_data(product(src_0_dims));
    std::vector<float> src_1_data(product(src_1_dims));

    // Initialize src_0 and src_1 (src).
    std::generate(src_0_data.begin(), src_0_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(src_1_data.begin(), src_1_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });

    // Create src and dst memory descriptors.
    auto src_0_md = memory::desc(src_0_dims, dt::f32, tag::nchw);
    auto src_1_md = memory::desc(src_1_dims, dt::f32, tag::nchw);
    auto dst_md = memory::desc(src_0_dims, dt::f32, tag::nchw);

    // Create src memory objects.
    auto src_0_mem = memory(src_0_md, engine);
    auto src_1_mem = memory(src_1_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_0_data.data(), src_0_mem);
    write_to_dnnl_memory(src_1_data.data(), src_1_mem);

    // Create operation descriptor.
    auto binary_d
            = binary::desc(algorithm::binary_mul, src_0_md, src_1_md, dst_md);

    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops binary_ops;
    binary_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    primitive_attr binary_attr;
    binary_attr.set_post_ops(binary_ops);

    // Create primitive descriptor.
    auto binary_pd = binary::primitive_desc(binary_d, binary_attr, engine);

    // Create the primitive.
    auto binary_prim = binary(binary_pd);

    // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
    std::unordered_map<int, memory> binary_args;
    binary_args.insert({DNNL_ARG_SRC_0, src_0_mem});
    binary_args.insert({DNNL_ARG_SRC_1, src_1_mem});
    binary_args.insert({DNNL_ARG_DST, src_0_mem});

    // Primitive execution: binary with ReLU.
    binary_prim.execute(engine_stream, binary_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_0_data.data(), src_0_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(binary_example, parse_engine_kind(argc, argv));
}
