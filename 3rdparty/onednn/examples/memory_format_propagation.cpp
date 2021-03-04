/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

/// @example memory_format_propagation.cpp
/// @copybrief memory_format_propagation_cpp
/// > Annotated version: @ref memory_format_propagation_cpp

#include <iostream>
#include <sstream>
#include <string>

/// @page memory_format_propagation_cpp Memory Format Propagation
/// This example demonstrates memory format propagation, which is critical for
/// deep learning applications performance.
///
/// > Example code: @ref memory_format_propagation.cpp
///
/// Memory format propagation is one of the central notions that needs to be
/// well-understood to use oneDNN correctly.
///
/// Convolution and inner product primitives choose the memory format when you
/// create them with the placeholder memory format
/// #dnnl::memory::format_tag::any for input or output. The memory format
/// chosen is based on different circumstances such as hardware and
/// convolutional parameters. Using the placeholder memory format is the
/// recommended practice for convolutions, since they are the most
/// compute-intensive operations in most topologies where they are present.
///
/// Other primitives, such as Elementwise, LRN, batch normalization and other,
/// on forward propagation should use the same memory format as the preceding
/// layer thus propagating the memory format through multiple oneDNN primitives.
/// This avoids unnecessary reorders which may be expensive and should be
/// avoided unless a compute-intensive primitive requires a different format.
/// For performance reasons, backward computations of such primitives requires
/// consistent memory format with the corresponding forward computations.
/// Hence, when initializing there primitives for backward computations you
/// should use #dnnl::memory::format_tag::any memory format tag as well.
///
/// Below is the short summary when to use and not to use memory format
/// #dnnl::memory::format_tag::any during operation description initialization:
///
/// | Primitive Kinds                                                                                                               | Forward Propagation                                                                               | Backward Propagation                                                                                | No Propagation                                                                                    |
/// | :--                                                                                                                           | :--                                                                                               | :--                                                                                                 | :--                                                                                               |
/// | Compute intensive: (De-)convolution, Inner product, RNN                                                                       | Use #dnnl::memory::format_tag::any                                                                | Use #dnnl::memory::format_tag::any                                                                  | N/A                                                                                               |
/// | Memory-bandwidth limited: Pooling, Layer and Batch Normalization, Local Response Normalization, Elementwise, Shuffle, Softmax | Use memory format from preceding layer for inputs, and #dnnl::memory::format_tag::any for outputs | Use #dnnl::memory::format_tag::any for gradient tensors, and actual memory formats for data tensors | N/A                                                                                               |
/// | Memory-bandwidth limited: Reorder, Concat, Sum, Binary                                                                        | N/A                                                                                               | N/A                                                                                                 | Use memory format from preceding layer for inputs, and #dnnl::memory::format_tag::any for outputs |
///
/// Additional format synchronization is required between forward and backward
/// computations when running training workloads. This topic is covered in
/// [Training-Specific Aspects](@ref dev_guide_inference_and_training_aspects_training).
///
/// For better understanding of the architecture and design of oneDNN
/// as well as the concepts used in the library, please refer to @ref
/// dev_guide_understanding_memory_formats.
///
/// @section memory_format_propagation_intro Introduction to the tutorial
///
/// This C++ API example demonstrates how to use optimized memory formats
/// supported by oneDNN:
/// - How to configure primitives to use optimized memory formats.
/// - How to determine whether data needs to be reordered from/to optimized
///   memory formats.
///
/// This tutorial assumes that the reader has already reviewed the
/// @ref getting_started_cpp tutorial.
///
/// The example is built around a CNN consisting of a convolution followed by
/// a pooling and consists of the following steps:
/// 1. Create a pooling primitive descriptor based on the memory format chosen
///    by the convolution primitive.
/// 2. Create memory descriptors for input and output data in the NCHW memory
///    format.
/// 3. Determine if input and output data needs to be reordered from/to the
///    optimized memory format.
/// 4. Create memory objects; and necessary primitives and execute them.
///
/// These steps are implemented in the @ref memory_format_propagation_tutorial
/// which in turn is called from `main()` which is also responsible for error
/// handling.

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

/// @page memory_format_propagation_cpp
/// @section memory_format_propagation_tutorial memory_format_propagation() function
///
void memory_format_propagation_tutorial(engine::kind engine_kind) {
    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub1 Initialization
    ///
    /// We start by creating an engine and a stream that we will use when
    /// creating primitive descriptors and executing primitives.
    ///
    /// @snippet memory_format_propagation.cpp Initialize engine and stream
    // [Initialize engine and stream]
    engine eng(engine_kind, 0);
    stream s(eng);
    // [Initialize engine and stream]

    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub2 Create convolution and pooling primitives
    ///
    /// To specify that a primitive should pick an optimized format for the
    /// specified computation parameters, we create memory descriptors with
    /// memory format set to @ref dnnl::memory::format_tag::any.
    ///
    /// This approach works only for a limited set of primitives: convolutions
    /// and inner products. Additionally, @ref dnnl::memory::format_tag::any
    /// can be specified for destination memory descriptors which implies that
    /// destination will have the same memory format as the source.
    ///
    /// @snippet memory_format_propagation.cpp Create placeholder memory descriptors
    // [Create placeholder memory descriptors]
    // Tensor and kernel dimensions. We use the same 3x3 kernel with padding=1
    // for both convolution and pooling primitives, which means that the
    // activation tensor shapes do not change.
    const int N = 1, H = 14, W = 14, IC = 128, OC = 256, KH = 3, KW = 3;
    auto conv_src_md = memory::desc({N, IC, H, W}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );
    auto conv_weights_md = memory::desc(
            {OC, IC, KH, KW}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );
    auto conv_dst_md = memory::desc({N, OC, H, W}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );
    auto pool_dst_md = conv_dst_md; // shape does not change
    // [Create placeholder memory descriptors]

    /// @page memory_format_propagation_cpp
    ///
    /// Next, we pass the memory descriptors to primitive descriptors
    /// constructors.
    ///
    /// @snippet memory_format_propagation.cpp Create convolution and pooling primitive descriptors
    // [Create convolution and pooling primitive descriptors]
    auto conv_pd = convolution_forward::primitive_desc(
            {prop_kind::forward_inference, algorithm::convolution_auto,
                    conv_src_md, conv_weights_md,
                    conv_dst_md, // shape information
                    {1, 1}, // strides
                    {1, 1}, {1, 1}}, // left and right padding
            eng);
    auto pool_pd = pooling_forward::primitive_desc(
            {prop_kind::forward_inference, algorithm::pooling_max,
                    conv_pd.dst_desc(), pool_dst_md, // shape information
                    {1, 1}, {KH, KW}, // strides and kernel
                    {1, 1}, {1, 1}}, // left and right padding
            eng);
    // [Create convolution and pooling primitive descriptors]

    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub3 Create source and destination memory objects
    ///
    /// We assume that the 'user' source and destination memory format is
    /// NCHW. Since there is no result validation in this tutorial, we do not
    /// bother with filling the data with some values and let oneDNN
    /// allocate the memory.
    ///
    /// @snippet memory_format_propagation.cpp Create source and destination memory objects
    // [Create source and destination memory objects]
    auto src_mem = memory(
            {{N, IC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
            eng);
    auto weights_mem = memory({{OC, IC, KH, KW}, memory::data_type::f32,
                                      memory::format_tag::oihw},
            eng);
    auto dst_mem = memory(
            {{N, OC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
            eng);
    // [Create source and destination memory objects]

    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub4 Determine if source and destination need to be reordered
    ///
    /// The idiomatic way to check if a reorder is necessary between the memory
    /// format expected a primitive (the convolution in our case) and the
    /// available memory format is to compare the corresponding memory
    /// descriptors.
    ///
    /// @snippet memory_format_propagation.cpp Determine if source needs to be reordered
    // [Determine if source needs to be reordered]
    bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();
    // [Determine if source needs to be reordered]

    /// @page memory_format_propagation_cpp
    ///
    /// @warning It is by design that it is not possible to just compare
    /// memory tags. The reason behind this is that a memory format tags only
    /// provide a partial description of how data is laid out in memory and do
    /// not, for example, describe memory objects obtained via sub-memory
    /// constructor.
    ///
    /// We repeat the process for the weights and destination memory format
    /// descriptors as well.
    ///
    /// @snippet memory_format_propagation.cpp Determine if weights and destination need to be reordered
    // [Determine if weights and destination need to be reordered]
    bool need_reorder_weights
            = conv_pd.weights_desc() != weights_mem.get_desc();
    bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.get_desc();
    // [Determine if weights and destination need to be reordered]

    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub45 Allocate intermediate buffers if necessary
    ///
    /// Based on the flags computed before, we can now decide if we need extra
    /// intermediate buffers to hold the source and weights data for the
    /// convolution and the output of the pooling.
    ///
    /// Memory objects for the intermediate buffers are created based on the
    /// memory descriptors obtained from the primitive descriptors to ensure
    /// consistency.
    ///
    /// @snippet memory_format_propagation.cpp Allocate intermediate buffers if necessary
    // [Allocate intermediate buffers if necessary]
    auto conv_src_mem
            = need_reorder_src ? memory(conv_pd.src_desc(), eng) : src_mem;
    auto conv_weights_mem = need_reorder_weights
            ? memory(conv_pd.weights_desc(), eng)
            : weights_mem;
    auto conv_dst_mem = memory(conv_pd.dst_desc(), eng);
    auto pool_dst_mem
            = need_reorder_dst ? memory(pool_pd.dst_desc(), eng) : dst_mem;
    // [Allocate intermediate buffers if necessary]

    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub5 Perform reorders for source data if necessary
    ///
    /// Now we get to the part where we actually start executing things. We
    /// check if reorders are necessary based on the flags computed before and
    /// create and execute them immediately.
    ///
    /// @note We call @ref dnnl::stream::wait() before reorder primitives
    /// get out of scope and destroyed to accommodate for potentially
    /// asynchronous execution.
    ///
    /// @snippet memory_format_propagation.cpp Perform reorders for source data if necessary
    // [Perform reorders for source data if necessary]
    if (need_reorder_src) {
        auto reorder_src = reorder(src_mem, conv_src_mem);
        reorder_src.execute(
                s, {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, conv_src_mem}});
        s.wait(); // wait for the reorder to complete
    }

    if (need_reorder_weights) {
        auto reorder_weights = reorder(weights_mem, conv_weights_mem);
        reorder_weights.execute(s,
                {{DNNL_ARG_FROM, weights_mem},
                        {DNNL_ARG_TO, conv_weights_mem}});
        s.wait(); // wait for the reorder to complete
    }
    // [Perform reorders for source data if necessary]

    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub6 Create and execute convolution and pooling primitives
    ///
    /// After the reorders, we are now ready to compute convolution and
    /// pooling.
    ///
    /// @snippet memory_format_propagation.cpp Create and execute convolution and pooling primitives
    // [Create and execute convolution and pooling primitives]
    auto conv_scratchpad_mem = memory(conv_pd.scratchpad_desc(), eng);
    auto conv = convolution_forward(conv_pd);
    conv.execute(s,
            {{DNNL_ARG_SRC, conv_src_mem}, {DNNL_ARG_WEIGHTS, conv_weights_mem},
                    {DNNL_ARG_DST, conv_dst_mem}});
    auto pool_scratchpad_mem = memory(pool_pd.scratchpad_desc(), eng);
    auto pool = pooling_forward(pool_pd);
    pool.execute(
            s, {{DNNL_ARG_SRC, conv_dst_mem}, {DNNL_ARG_DST, pool_dst_mem}});
    s.wait();
    // [Create and execute convolution and pooling primitives]

    /// @page memory_format_propagation_cpp
    /// @subsection memory_format_propagation_sub7 Reorder destination data if necessary
    ///
    /// The only potentially remaining operation is a reorder from the pooling
    /// destination memory object to the user's one.  Similarly to the
    /// reorders for the source and weights memory objects, it is performed
    /// depending on the value of the previously computed flag.
    ///
    /// @snippet memory_format_propagation.cpp Reorder destination data if necessary
    // [Reorder destination data if necessary]
    if (need_reorder_dst) {
        auto reorder_dst = reorder(pool_dst_mem, dst_mem);
        reorder_dst.execute(
                s, {{DNNL_ARG_FROM, pool_dst_mem}, {DNNL_ARG_TO, dst_mem}});
        s.wait();
    }
    // [Reorder destination data if necessary]
}

int main(int argc, char **argv) {
    return handle_example_errors(
            memory_format_propagation_tutorial, parse_engine_kind(argc, argv));
}

/// @page memory_format_propagation_cpp Memory format propagation
/// @subsection memory_format_propagation_results Results
///
/// Upon compiling and run the example the output should be just:
///
/// ~~~sh
/// Example passed.
/// ~~~
///
/// It may be interesting to check what really happens during the run. We can
/// use `DNNL_VERBOSE` environment variable for that (see also @ref
/// dev_guide_verbose). Here's an example output:
///
/// ~~~sh
/// $ DNNL_VERBOSE=1 ./memory-format-propagation-cpp
/// dnnl_verbose,info,oneDNN <ver> (Git Hash <hash>)
/// dnnl_verbose,info,cpu,runtime:OpenMP
/// dnnl_verbose,info,cpu,isa:Intel AVX2
/// dnnl_verbose,info,gpu,runtime:none
/// dnnl_verbose,exec,cpu,reorder,jit:uni,undef,
///     src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,,,1x128x14x14,0.326904
/// dnnl_verbose,exec,cpu,reorder,jit:uni,undef,
///     src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd8b8a:f0,,,256x128x3x3,0.244141
/// dnnl_verbose,exec,cpu,convolution,jit:avx2,forward_inference,
///     src_f32::blocked:aBcd8b:f0 wei_f32::blocked:ABcd8b8a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd8b:f0,,
///     alg:convolution_direct,mb1_ic128oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1,1.20312
/// dnnl_verbose,exec,cpu,pooling,jit:avx,forward_inference,
///     src_f32::blocked:aBcd8b:f0 dst_f32::blocked:aBcd8b:f0 ws_undef::undef::f0,,
///     alg:pooling_max,mb1ic256_ih14oh14kh3sh1ph1_iw14ow14kw3sw1pw1,0.187012
/// dnnl_verbose,exec,cpu,reorder,jit:uni,undef,
///     src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,1x256x14x14,0.0419922
/// Example passed on CPU.
/// ~~~
///
/// From this output we can deduce that:
/// * The convolution primitive picked up @ref
///   dnnl::memory::format_tag::aBcd8b optimized memory format for
///   activations. In this format the channels dimension (denoted by letter B
///   since it is the second dimension; see also @ref dev_guide_conventions)
///   is blocked by a factor of 8. Because of this memory format is different
///   from the NCHW format the tutorial uses, the source and destination had
///   to be reordered to and from this optimized memory layout.
/// * The convolution primitive picked up @ref
///   dnnl::memory::format_tag::ABcd8b8a optimized memory format (output (A)
///   and input (B) channel dimensions blocked by 8) which we also had to
///   reorder the initial weights to since they are in the OIHW memory format.
