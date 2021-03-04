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

/// @example lrn.cpp
/// > Annotated version: @ref lrn_example_cpp
///
/// @page lrn_example_cpp_short
///
/// This C++ API demonstrates how to create and execute a
/// [Local response normalization](@ref dev_guide_lrn) primitive in forward
/// training propagation mode.
///
/// @page lrn_example_cpp Local Response Normalization Primitive Example
/// @copydetails lrn_example_cpp_short
///
/// @include lrn.cpp

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

void lrn_example(dnnl::engine::kind engine_kind) {

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
    std::vector<float> dst_data(product(src_dims));

    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create src and dst memory descriptors and memory objects.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(src_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Create operation descriptor.
    const memory::dim local_size = 5;
    const float alpha = 1.e-4f;
    const float beta = 0.75f;
    const float k = 1.f;
    auto lrn_d = lrn_forward::desc(prop_kind::forward_training,
            algorithm::lrn_across_channels, src_md, local_size, alpha, beta, k);

    // Create primitive descriptor.
    auto lrn_pd = lrn_forward::primitive_desc(lrn_d, engine);

    // Create workspace memory object using memory descriptors created by the
    // primitive descriptor.
    // NOTE: Here, workspace may or may not be required in forward training
    // mode, and is used to speed-up the backward propagation.
    auto workspace_mem = memory(lrn_pd.workspace_desc(), engine);

    // Create the primitive.
    auto lrn_prim = lrn_forward(lrn_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> lrn_args;
    lrn_args.insert({DNNL_ARG_SRC, src_mem});
    lrn_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});
    lrn_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution.
    lrn_prim.execute(engine_stream, lrn_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(lrn_example, parse_engine_kind(argc, argv));
}
