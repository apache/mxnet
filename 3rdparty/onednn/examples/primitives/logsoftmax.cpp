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

/// @example logsoftmax.cpp
/// > Annotated version: @ref logsoftmax_example_cpp
///
/// @page logsoftmax_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Logsoftmax](@ref dev_guide_logsoftmax) primitive in forward training
/// propagation mode.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Softmax along axis 1 (C) for 2D tensors.
///
/// @page logsoftmax_example_cpp Logsoftmax Primitive Example
/// @copydetails logsoftmax_example_cpp_short
///
/// @include logsoftmax.cpp

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

void logsoftmax_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 1000; // channels

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC};

    // Allocate buffer.
    std::vector<float> src_data(product(src_dims));

    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create src memory descriptor and memory object.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nc);
    auto src_mem = memory(src_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Logsoftmax axis.
    const int axis = 1;

    // Create operation descriptor.
    auto logsoftmax_d = logsoftmax_forward::desc(
            prop_kind::forward_training, src_md, axis);

    // Create primitive descriptor.
    auto logsoftmax_pd
            = logsoftmax_forward::primitive_desc(logsoftmax_d, engine);

    // Create the primitive.
    auto logsoftmax_prim = logsoftmax_forward(logsoftmax_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> logsoftmax_args;
    logsoftmax_args.insert({DNNL_ARG_SRC, src_mem});
    logsoftmax_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution.
    logsoftmax_prim.execute(engine_stream, logsoftmax_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(src_data.data(), src_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            logsoftmax_example, parse_engine_kind(argc, argv));
}
