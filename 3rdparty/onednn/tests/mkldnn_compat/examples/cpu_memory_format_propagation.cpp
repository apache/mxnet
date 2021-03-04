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

/// @example cpu_memory_format_propagation.cpp
/// This example demonstrates memory format propagation, which is critical for
/// deep learning applications performance.
///
/// > Annotated version: @ref cpu_memory_format_propagation_cpp

#include <iostream>
#include <sstream>
#include <string>

/// @page cpu_memory_format_propagation_cpp Memory format propagation
/// > Example code: @ref cpu_memory_format_propagation.cpp
///
/// Format propagation is one of the central notions that needs to be
/// well-understood to use Intel MKL-DNN correctly.
///
/// Convolution and inner product primitives choose the memory format when you
/// create them with the placeholder memory format @ref
/// mkldnn::memory::format_tag::any for input or output. The memory format
/// chosen is based on different circumstances such as hardware and
/// convolutional parameters. Using the placeholder memory format is the
/// recommended practice for convolutions, since they are the most
/// compute-intensive operations in most topologies where they are present.
///
/// Other primitives, such as ReLU, LRN, batch normalization and other, should
/// use the same memory format as the preceding layer thus propagating the
/// memory format through multiple Intel MKL-DNN primitives. This avoids
/// unnecessary reorders which may be expensive and should be avoided unless a
/// compute-intensive primitive requires a different format.
///
/// Additional format synchronization is required between forward and backward
/// computations when running training workloads. This topic is covered in
/// [Training-Specific Aspects](@ref dev_guide_inference_and_training_aspects_training).
///
/// For better understanding of the architecture and design of Intel MKL-DNN
/// as well as the concepts used in the library, please refer to @ref
/// dev_guide_understanding_memory_formats.
///
/// @section cpu_memory_format_propagation_intro Introduction to the tutorial
///
/// This C++ API example demonstrates how to use optimized memory formats
/// supported by Intel MKL-DNN:
/// - How to configure primitives to use optimized memory formats.
/// - How to determine whether data needs to be reordered from/to optimized
///   memory formats.
///
/// This tutorial assumes that the reader has already reviewed the
/// @ref cpu_getting_started_cpp tutorial.
///
/// The example is built around a CNN consisting of a convolution followed by
/// a pooling and consists of the following steps:
/// 1. Create a pooling primitive descriptor based on the memory format chosen
///    by the convolution primitive.
/// 2. Create memory descriptors for input and output data in the NHWC memory
///    format.
/// 3. Determine if input and output data needs to be reordered from/to the
///    optimized memory format.
/// 4. Create memory objects; and necessary primitives and execute them.
///
/// These steps are implemented in the @ref cpu_memory_format_propagation_tutorial
/// which in turn is called from `main()` which is also responsible for error
/// handling.

#include "mkldnn.hpp"
#include "mkldnn_debug.h"
using namespace mkldnn;

/// @page cpu_memory_format_propagation_cpp
/// @section cpu_memory_format_propagation_tutorial cpu_memory_format_propagation() function
/// @page cpu_memory_format_propagation_cpp
void cpu_memory_format_propagation_tutorial() {
    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub1 Initialization
    ///
    /// We start by creating a CPU engine and a stream that we will use when
    /// creating primitive descriptors and executing primitives.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Initialize engine and stream
    // [Initialize engine and stream]
    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);
    // [Initialize engine and stream]

    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub2 Create convolution and pooling primitives
    ///
    /// To specify that a primitive should pick an optimized format for the
    /// specified computation parameters, we create memory descriptors with
    /// memory format set to @ref mkldnn::memory::format_tag::any.
    ///
    /// This approach works only for a limited set of primitives: convolutions
    /// and inner products. Additionally, @ref mkldnn::memory::format_tag::any
    /// can be specified for destination memory descriptors which implies that
    /// destination will have the same memory format as the source.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Create placeholder memory descriptors
    // [Create placeholder memory descriptors]
    // Tensor and kernel dimensions. We use the same 3x3 kernel with padding=1
    // for both convolution and pooling primitives, which means that the
    // activation tensor shapes do not change.
    const int N = 1, H = 14, W = 14, IC = 256, OC = IC, KH = 3, KW = 3;
    auto conv_src_md = memory::desc({N, IC, H, W}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );
    auto conv_weights_md = memory::desc(
            {IC, OC, KH, KW}, memory::data_type::f32,
            memory::format_tag::any // let convolution choose memory format
    );
    auto conv_dst_md = conv_src_md; // shape does not change
    auto pool_dst_md = conv_dst_md; // shape does not change
    // [Create placeholder memory descriptors]

    /// @page cpu_memory_format_propagation_cpp
    ///
    /// Next, we pass the memory descriptors to primitive descriptors
    /// constructors.
    ///
    // @snippet cpu_memory_format_propagation.cpp Create convolution and pooling primitive descriptors
    // [Create convolution and pooling primitive descriptors]
    auto conv_pd = convolution_forward::primitive_desc(
            {prop_kind::forward_inference, algorithm::convolution_auto,
                    conv_src_md, conv_weights_md,
                    conv_dst_md, // shape information
                    {1, 1}, // strides
                    {1, 1}, {1, 1}}, // left and right padding
            cpu_engine);
    auto pool_pd = pooling_forward::primitive_desc(
            {prop_kind::forward_inference, algorithm::pooling_max,
                    conv_pd.dst_desc(), pool_dst_md, // shape information
                    {1, 1}, {KH, KW}, // strides and kernel
                    {1, 1}, {1, 1}}, // left and right padding
            cpu_engine);
    // [Create convolution and pooling primitive descriptors]

    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub3 Create source and destination memory objects
    ///
    /// We assume that the 'user' source and destination memory format is
    /// NHWC. Since there is no result validation in this tutorial, we do not
    /// bother with filling the data with some values and let the Intel
    /// MKL-DNN library to allocate the memory.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Create source and destination memory objects
    // [Create source and destination memory objects]
    auto src_mem = memory(
            {{N, IC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
            cpu_engine);
    auto weights_mem = memory({{IC, OC, KH, KW}, memory::data_type::f32,
                                      memory::format_tag::oihw},
            cpu_engine);
    auto dst_mem = memory(
            {{N, IC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
            cpu_engine);
    // [Create source and destination memory objects]

    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub4 Determine if source and destination need to be reordered
    ///
    /// The idiomatic way to check if a reorder is necessary between the memory
    /// format expected a primitive (the convolution in our case) and the
    /// available memory format is to compare the corresponding memory
    /// descriptors.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Determine if source needs to be reordered
    // [Determine if source needs to be reordered]
    bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();
    // [Determine if source needs to be reordered]

    /// @page cpu_memory_format_propagation_cpp
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
    /// @snippet cpu_memory_format_propagation.cpp Determine if weights and destination need to be reordered
    // [Determine if weights and destination need to be reordered]
    bool need_reorder_weights
            = conv_pd.weights_desc() != weights_mem.get_desc();
    bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.get_desc();
    // [Determine if weights and destination need to be reordered]

    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub45 Allocate intermediate buffers if necessary
    ///
    /// Based on the flags computed before, we can now decide if we need extra
    /// intermediate buffers to hold the source and weights data for the
    /// convolution and the output of the pooling.
    ///
    /// Memory objects for the intermediate buffers are created based on the
    /// memory descriptors obtained from the primitive descriptors to ensure
    /// consistency.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Allocate intermediate buffers if necessary
    // [Allocate intermediate buffers if necessary]
    auto conv_src_mem = need_reorder_src
            ? memory(conv_pd.src_desc(), cpu_engine)
            : src_mem;
    auto conv_weights_mem = need_reorder_weights
            ? memory(conv_pd.weights_desc(), cpu_engine)
            : weights_mem;
    auto conv_dst_mem = memory(conv_pd.dst_desc(), cpu_engine);
    auto pool_dst_mem = need_reorder_dst
            ? memory(pool_pd.dst_desc(), cpu_engine)
            : dst_mem;
    // [Allocate intermediate buffers if necessary]

    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub5 Perform reorders for source data if necessary
    ///
    /// Now we get to the part where we actually start executing things. We
    /// check if reorders are necessary based on the flags computed before and
    /// create and execute them immediately.
    ///
    /// @note We call @ref mkldnn::stream::wait() before reorder primitives
    /// get out of scope and destroyed to accommodate for potentially
    /// asynchronous execution.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Perform reorders for source data if necessary
    // [Perform reorders for source data if necessary]
    if (need_reorder_src) {
        auto reorder_src = reorder(src_mem, conv_src_mem);
        reorder_src.execute(cpu_stream,
                {{MKLDNN_ARG_FROM, src_mem}, {MKLDNN_ARG_TO, conv_src_mem}});
        cpu_stream.wait(); // wait for the reorder to complete
    }

    if (need_reorder_weights) {
        auto reorder_weights = reorder(weights_mem, conv_weights_mem);
        reorder_weights.execute(cpu_stream,
                {{MKLDNN_ARG_FROM, weights_mem},
                        {MKLDNN_ARG_TO, conv_weights_mem}});
        cpu_stream.wait(); // wait for the reorder to complete
    }
    // [Perform reorders for source data if necessary]

    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub6 Create and execute convolution and pooling primitives
    ///
    /// After the reorders, we are now ready to compute convolution and
    /// pooling.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Create and execute convolution and pooling primitives
    // [Create and execute convolution and pooling primitives]
    auto conv_scratchpad_mem = memory(conv_pd.scratchpad_desc(), cpu_engine);
    auto conv = convolution_forward(conv_pd);
    conv.execute(cpu_stream,
            {{MKLDNN_ARG_SRC, conv_src_mem},
                    {MKLDNN_ARG_WEIGHTS, conv_weights_mem},
                    {MKLDNN_ARG_DST, conv_dst_mem}});
    auto pool_scratchpad_mem = memory(pool_pd.scratchpad_desc(), cpu_engine);
    auto pool = pooling_forward(pool_pd);
    pool.execute(cpu_stream,
            {{MKLDNN_ARG_SRC, conv_dst_mem}, {MKLDNN_ARG_DST, pool_dst_mem}});
    cpu_stream.wait();
    // [Create and execute convolution and pooling primitives]

    /// @page cpu_memory_format_propagation_cpp
    /// @subsection cpu_memory_format_propagation_sub7 Reorder destination data if necessary
    ///
    /// The only potentially remaining operation is a reorder from the pooling
    /// destination memory object to the users's one.  Similarly to the
    /// reorders for the source and weights memory objects, it is performed
    /// depending on the value of the previously computed flag.
    ///
    /// @snippet cpu_memory_format_propagation.cpp Reorder destination data if necessary
    // [Reorder destination data if necessary]
    if (need_reorder_dst) {
        auto reorder_dst = reorder(pool_dst_mem, dst_mem);
        reorder_dst.execute(cpu_stream,
                {{MKLDNN_ARG_FROM, pool_dst_mem}, {MKLDNN_ARG_TO, dst_mem}});
        cpu_stream.wait();
    }
    // [Reorder destination data if necessary]
}

int main(int argc, char **argv) {
    try {
        cpu_memory_format_propagation_tutorial();
    } catch (mkldnn::error &e) {
        std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
                  << "Error status: " << mkldnn_status2str(e.status)
                  << std::endl;
        return 1;
    } catch (std::string &e) {
        std::cerr << "Error in the example: " << e << std::endl;
        return 2;
    }

    std::cout << "Example passes" << std::endl;
    return 0;
}

/// Upon compiling and run the example the output should be just:
///
/// ~~~sh
/// Example passes
/// ~~~
///
/// It may be interesting to check what really happens during the run. We can
/// use `MKLDNN_VERBOSE` environment variable for that (see also @ref
/// dev_guide_verbose). Here's an example output:
///
/// ~~~sh
/// $ MKLDNN_VERBOSE=1 ./cpu_memory_format_propagation
/// mkldnn_verbose,info,Intel(R) MKL-DNN <ver> (Git Hash <hash>),Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
/// mkldnn_verbose,exec,reorder,jit:uni,undef,
///     src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd8b:f0,num:1,1x256x14x14,1.03101
/// mkldnn_verbose,exec,reorder,jit:uni,undef,
///     src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd8b8a:f0,num:1,256x256x3x3,5.69678
/// mkldnn_verbose,exec,convolution,jit:avx2,forward_inference,
///     src_f32::blocked:aBcd8b:f0 wei_f32::blocked:ABcd8b8a:f0 dst_f32::blocked:aBcd8b:f0,
///     alg:convolution_direct,mb1_ic256oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1,1.65698
/// mkldnn_verbose,exec,pooling,jit:avx,forward_inference,
///     src_f32::blocked:aBcd8b:f0 dst_f32::blocked:aBcd8b:f0,
///     alg:pooling_max,mb1ic256_ih14oh14kh3sh1ph1_iw14ow14kw3sw1pw1,0.322021
/// mkldnn_verbose,exec,reorder,jit:uni,
///     undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,num:1,1x256x14x14,0.333008
/// Example passes
/// ~~~
///
/// From this output we can deduce that:
/// * The convolution primitive picked up @ref
///   mkldnn::memory::format_tag::aBcd8b optimized memory format for
///   activations. In this format the channels dimension (denoted by letter B
///   since it is the second dimension; see also @ref dev_guide_conventions)
///   is blocked by a factor of 8. Because of this memory format is different
///   from the NHWC format the tutorial uses, the source and destination had
///   to be reordered to and from this optimized memory layout.
/// * The convolution primitive picked up @ref
///   mkldnn::memory::format_tag::ABcd8b8a optimized memory format (output (A)
///   and input (B) channel dimensions blocked by 8) which we also had to
///   reorder the initial weights to since they are in the OIHW memory format.
///
/// @page cpu_memory_format_propagation_cpp
