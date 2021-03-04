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

/// @example pooling.cpp
/// > Annotated version: @ref pooling_example_cpp
///
/// @page pooling_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Pooling](@ref dev_guide_pooling) primitive in forward training propagation
/// mode.
///
/// @page pooling_example_cpp Pooling Primitive Example
/// @copydetails pooling_example_cpp_short
///
/// @include pooling.cpp

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

void pooling_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // input channels
            IH = 27, // input tensor height
            IW = 27, // input tensor width
            KH = 11, // kernel height
            KW = 11, // kernel width
            PH_L = 0, // height padding: left
            PH_R = 0, // height padding: right
            PW_L = 0, // width padding: left
            PW_R = 0, // width padding: right
            SH = 4, // height-wise stride
            SW = 4, // width-wise stride
            DH = 1, // height-wise dilation
            DW = 1; // width-wise dilation

    const memory::dim OH = (IH - ((KH - 1) * DH + KH) + PH_L + PH_R) / SH + 1;
    const memory::dim OW = (IW - ((KW - 1) * DW + KW) + PW_L + PW_R) / SW + 1;

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims dst_dims = {N, IC, OH, OW};

    // Kernel dimensions.
    memory::dims kernel_dims = {KH, KW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};
    memory::dims dilation = {DH, DW};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> dst_data(product(dst_dims));

    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create memory descriptors and memory objects for src and dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto src_mem = memory(src_md, engine);

    auto dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Create operation descriptor.
    auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_training,
            algorithm::pooling_max, src_md, dst_md, strides_dims, kernel_dims,
            dilation, padding_dims_l, padding_dims_r);

    // Create primitive descriptor.
    auto pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);

    // Create workspace memory objects using memory descriptor created by the
    // primitive descriptor.
    // NOTE: Here, the workspace is required to save the indices where maximum
    // was found, and is used in backward pooling to perform upsampling.
    auto workspace_mem = memory(pooling_pd.workspace_desc(), engine);

    // Create the primitive.
    auto pooling_prim = pooling_v2_forward(pooling_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> pooling_args;
    pooling_args.insert({DNNL_ARG_SRC, src_mem});
    pooling_args.insert({DNNL_ARG_DST, dst_mem});
    pooling_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

    // Primitive execution: pooling.
    pooling_prim.execute(engine_stream, pooling_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            pooling_example, parse_engine_kind(argc, argv));
}
