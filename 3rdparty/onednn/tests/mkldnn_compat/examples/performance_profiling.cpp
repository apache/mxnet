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

/// @example performance_profiling.cpp
/// This example demonstrates the best practices for application performance
/// optimizations with Intel MKL-DNN.
///
/// > Annotated version: @ref performance_profiling_cpp
/// @page performance_profiling_cpp Performance Profiling Example

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "example_utils.hpp"

/// > Example code: @ref performance_profiling.cpp
///
/// This example uses [MKLDNN_VERBOSE](@ref dev_guide_verbose) trace output
/// to tune Intel MKL-DNN code to align
/// with the [best practices](@ref dev_guide_inference).
///
/// It will assume knowledge of memory formats and their usage in
/// Intel MKL-DNN. You can read more about this topic
/// [here](@ref cpu_memory_format_propagation_cpp).
///
/// The example has three different implementations of the mathematical
/// operation:
/// 1. *Naive implementation* executes 2D convolution followed by
/// ReLU on the data in **NCHW** format. This implementation
/// does not align with Intel MKL-DNN best practices and results in
/// suboptimal performance.
/// 2. *Blocked format implementation* executes the same operations
/// sequence on the **blocked format** optimized for convolution
/// performance. This implementation uses `format_tag=ANY` to create a
/// convolution memory descriptor to determine the data format optimal
/// for the convolution implementation. It then **propagates the blocked
/// format** to the non-intensive ReLU. This implementation results
/// in better overall performance than the naive implementation.
/// 3. *Fused implementation* executes convolution fused with ReLU on
/// blocked data format. This implementation uses
/// `format_tag=ANY` to create a convolution memory descriptor, and then
/// adds ReLU as a **post-op** to the convolution primitive. This version
/// implements all of the best practices for inference resulting in the
/// best overall performance.
///
/// @section performance_profiling_cpp_walkthrough Walkthrough
///
/// The program in \ref performance_profiling.cpp includes all three
/// implementations introduced above. You can select the specific implementation
/// using command line options.
///
/// After compilation, you can execute each implementation with:
/// ~~~sh
/// ./program.exe implementation
/// ~~~
///
/// Before you run the program, set your `MKLDNN_VERBOSE` environment
/// variable to 1:
/// ~~~sh
/// export MKLDNN_VERBOSE=1
/// ~~~
///
/// The program starts by creating Intel MKL-DNN memory objects in **NCHW**
/// format. These are called `user_` because they are meant to represent the
/// user's source data entering Intel MKL-DNN with the NCHW format.
/// @page performance_profiling_cpp
/// @snippet performance_profiling.cpp Set dimensions
/// @page performance_profiling_cpp
/// @snippet performance_profiling.cpp Create memory objects
/// @page performance_profiling_cpp
/// @note You can change the batch size to easily increase/decrease the workload.
///
/// The following descriptions of each implementation will reference each other,
/// and are meant to be read in order.
///

using namespace mkldnn;

// [Prologue]
// Set Strides and Padding
const memory::dims strides = {4, 4};
const memory::dims padding = {0, 0};

// function to init data
void init_data(memory &m, float v) {
    size_t size = m.get_desc().get_size() / sizeof(float);
    std::vector<float> data(size);
    read_from_mkldnn_memory(data.data(), m);
    for (size_t i = 0; i < size; ++i)
        data[i] = v;
}

// function to execute non-fused relu
void create_and_execute_relu(memory &data, const engine &eng, const stream &s) {
    // relu operates on whatever data format is given to it

    // create a primitive
    auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, data.get_desc(), 0.f, 0.f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    auto relu = eltwise_forward(relu_pd);

    // execute it (in-place)
    relu.execute(s, {{MKLDNN_ARG_SRC, data}, {MKLDNN_ARG_DST, data}});
}

// [Create post_op attr with relu]
// function to create post-op attribute for fused relu
primitive_attr create_attr_with_relu_post_op() {
    // create a post-op with relu
    post_ops ops;
    ops.append_eltwise(1.f, algorithm::eltwise_relu, 0.f, 0.f);

    // create an attribute and set the corresponding post op
    primitive_attr attr;
    attr.set_post_ops(ops);

    return attr;
}
// [Create post_op attr with relu]

// Implementation for naive convolution on nchw (data) and oihw (weights),
// followed by execution of non-fused relu
void conv_relu_naive(const memory &user_src, const memory &user_wei,
        memory &user_dst, const engine &eng, const stream &s) {
    /// @section performance_profiling_cpp_implementation1 Naive Implementation
    /// This implementation is launched with the following shell code:
    /// ~~~sh
    /// ./program.exe naive
    /// ~~~
    /// The program will call the implementation defined in the function
    /// `conv_relu_naive()`.
    ///
    /// First it sets the dimensions and format for convolution memory
    /// descriptors (`_md`) to match `user_` values--one `md` each for source,
    /// destination, and weight data. Then it uses those `md` to create the
    /// convolution descriptor `conv_d`, which tells Intel MKL-DNN to use
    /// plain format (NCHW) for the convolution.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create mem_desc
    // [Create mem_desc]
    // copy the dimensions and format from user's memory
    auto conv_src_md = memory::desc(user_src.get_desc());
    auto conv_wei_md = memory::desc(user_wei.get_desc());
    auto conv_dst_md = memory::desc(user_dst.get_desc());
    // [Create mem_desc]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create conv_desc
    // [Create conv_desc]
    // create a convolution descriptor
    auto conv_d = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_wei_md,
            conv_dst_md, strides, padding, padding);
    // [Create conv_desc]
    /// Next the program creates a convolution primitive descriptor `conv_pd`
    /// and convolution primitive `conv`. These structs will inherit
    /// NCHW format from `md` by way of the `conv_d`. Finally it creates
    /// the convolution primitive `conv` and adds it to the stream `s`, and then
    /// executes the `create_and_execute_relu(user_dst)` function.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create conv_prim_desc
    // [Create conv_prim_desc]
    // create a convolution primitive descriptor
    auto conv_pd = convolution_forward::primitive_desc(conv_d, eng);
    // [Create conv_prim_desc]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create conv_primitive
    // [Create conv_primitive]
    // create convolution primitive
    auto conv = convolution_forward(conv_pd);
    // [Create conv_primitive]
    // @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Add to stream
    // [Add to stream]
    // execute convolution by adding it to the stream s
    conv.execute(s,
            {{MKLDNN_ARG_SRC, user_src}, {MKLDNN_ARG_WEIGHTS, user_wei},
                    {MKLDNN_ARG_DST, user_dst}});
    // [Add to stream]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create and execute relu
    // [Create and execute relu]
    // execute relu (on convolution's destination format, whatever it is)
    create_and_execute_relu(user_dst, eng, s);
    // [Create and execute relu]
    /// @page performance_profiling_cpp
    /// @note The function for creation and execution of ReLU primitive is
    /// defined elsewhere to keep this example clean. It is an non-intensive
    /// operation, so the `create_and_execute_relu()` function uses whatever
    /// the input data format is at the time it is called.
    ///
    /// Using NCHW data format may result in suboptimal performance for compute
    /// intensives primitives, as shown in the following MKLDNN_VERBOSE output
    /// by the convolution and relu execution
    /// times of 235.9 and 100.3 milliseconds, respectively.
    ///
    /// *MKLDNN_VERBOSE output (see configuration notice\*):*
    /// ~~~sh
    /// mkldnn_verbose,exec,convolution,gemm:jit,forward_inference,src_f32::
    ///         blocked:abcd:f0 wei_f32::blocked:abcd:f0 dst_f32::
    ///         blocked:abcd:f0,alg:convolution_direct,
    ///         mb1000_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,235.86
    /// mkldnn_verbose,exec,eltwise,jit:avx512_common,forward_inference,
    ///         data_f32::blocked:abcd:f0,alg:eltwise_relu,1000x96x55x55,100.264
    /// ~~~
    /// In *Blocked format implementation*, we will incorporate the best
    /// practice of letting Intel MKL-DNN determine the optimal format
    /// for convolution primitive.
}

// Implementation for convolution on blocked format for data and
// weights, followed by execution of non-fused relu
void conv_relu_blocked(memory &user_src, memory &user_wei, memory &user_dst,
        const engine &eng, const stream &s) {
    /// @page performance_profiling_cpp
    /// @section performance_profiling_cpp_implementation2 Blocked format implementation
    /// This implementation is launched with the following shell code:
    /// ~~~sh
    /// ./program.exe blocked
    /// ~~~
    /// The program will call the implementation defined in the function
    /// `conv_relu_blocked()`.
    ///
    /// First it creates the md as in **naive implementation**. Next it changes
    /// the mkldnn::memory::format_tag for each md to `ANY`. Then it uses those
    /// md to create the convolution descriptor conv_d, which tells Intel
    /// MKL-DNN to use whatever format it recommends for the convolution.
    /// Intel MKL-DNN will choose the a friendly blocked format.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create mem_desc with tag=any
    // [Create mem_desc with tag=any]
    // copy the dimensions and format from user's memory
    auto conv_src_md = memory::desc(user_src.get_desc());
    auto conv_wei_md = memory::desc(user_wei.get_desc());
    auto conv_dst_md = memory::desc(user_dst.get_desc());

    // reset format to "any" to allow convolution to pick the best implementation
    conv_src_md.data.format_kind = mkldnn_format_kind_any;
    conv_wei_md.data.format_kind = mkldnn_format_kind_any;
    conv_dst_md.data.format_kind = mkldnn_format_kind_any;
    // [Create mem_desc with tag=any]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create conv_desc implementation2
    // [Create conv_desc implementation2]
    // create a convolution descriptor
    auto conv_d = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_wei_md,
            conv_dst_md, strides, padding, padding);
    // [Create conv_desc implementation2]
    /// Next the program creates a convolution primitive descriptor conv_pd and
    /// convolution primitive conv as in naive implementation.
    /// However, in this implementation the structs will inherit blocked format
    /// from md by way of the conv_d.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create conv_prim_desc implementation2
    // [Create conv_prim_desc implementation2]
    // create a convolution primitive descriptor and primitive
    auto conv_pd = convolution_forward::primitive_desc(conv_d, eng);
    // [Create conv_prim_desc implementation2]
    /// Since the resulting convolution primitive will expect
    /// blocked source data, conditional reorders are inserted to convert
    /// input data to blocked format if required.
    /// The input data user_src is NCHW, so this conditional will be triggered:
    ///
    /// @note The reoders are applied using Intel MKL-DNN `reorder` primitive.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Conditionally create and execute reorder prims
    // [Conditionally create and execute reorder prims]
    // prepare convolution source
    memory conv_src = user_src;
    if (conv_pd.src_desc() != user_src.get_desc()) {
        conv_src = memory(conv_pd.src_desc(), eng);
        auto r_pd = reorder::primitive_desc(user_src, conv_src);
        reorder(r_pd).execute(s, user_src, conv_src);
    }

    // prepare convolution weights
    memory conv_wei = user_wei;
    if (conv_pd.weights_desc() != user_wei.get_desc()) {
        conv_wei = memory(conv_pd.weights_desc(), eng);
        auto r_pd = reorder::primitive_desc(user_wei, conv_wei);
        reorder(r_pd).execute(s, user_wei, conv_wei);
    }

    // prepare convolution destination
    memory conv_dst = user_dst;
    if (conv_pd.dst_desc() != user_dst.get_desc())
        conv_dst = memory(conv_pd.dst_desc(), eng);
    // [Conditionally create and execute reorder prims]
    /// Finally it creates the convolution primitive `conv` and adds it to the
    /// stream `s` with the reordered data (`conv_src`, `conv_wei`, `conv_dst1`)
    /// as inputs and then executes the
    /// `create_and_execute_relu(conv_dst)` function.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create conv_primitive implementation2
    // [Create conv_primitive implementation2]
    // create convolution primitive
    auto conv = convolution_forward(conv_pd);
    // [Create conv_primitive implementation2]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Add to stream implementation2
    // [Add to stream implementation2]
    // execute convolution by adding it to the stream s
    conv.execute(s,
            {{MKLDNN_ARG_SRC, conv_src}, {MKLDNN_ARG_WEIGHTS, conv_wei},
                    {MKLDNN_ARG_DST, conv_dst}});
    // [Add to stream implementation2]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create and execute relu implementation2
    // [Create and execute relu implementation2]
    // execute relu (on convolution's destination format, whatever it is)
    create_and_execute_relu(conv_dst, eng, s);
    // [Create and execute relu implementation2]
    if (conv_pd.dst_desc() != user_dst.get_desc()) {
        auto r_pd = reorder::primitive_desc(conv_dst, user_dst);
        reorder(r_pd).execute(s, conv_dst, user_dst);
    }
    /// @page performance_profiling_cpp
    /// Blocked memory format is recommended for Intel MKL-DNN primitive
    /// execution and provides better performance, as shown in the
    /// MKLDNN_VERBOSE output by the convolution and relu execution times of
    /// 119.6 and 34.4 milliseconds (down from 235.9 and 100.3 in
    /// *naive implementation*), respectively.
    /// In this implementation, there is an additional reorder operation that
    /// executes before and after the the conv + relu. This small cost is worth
    /// the gain from executing in blocked format. If fact, it becomes
    /// negligible when chaining together multiple Intel Mkl-DNN operations in
    /// succession. In these situations, you can do one reorder at the beginning
    /// and one at the end of the chain, and only pay the reorder penalty at
    /// those points in the execution.
    ///
    /// *MKLDNN_VERBOSE output (see configuration notice\*):*
    /// ~~~sh
    /// mkldnn_verbose,exec,reorder,jit:uni,undef,src_f32::blocked:abcd:f0
    ///         dst_f32::blocked:Acdb16a:f0,num:1,96x3x11x11,3.71387
    /// mkldnn_verbose,exec,convolution,jit:avx512_common,forward_inference,
    ///         src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0
    ///         dst_f32::blocked:aBcd16b:f0,alg:convolution_direct,
    ///         mb1000_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,119.649
    /// mkldnn_verbose,exec,eltwise,jit:avx512_common,forward_inference,
    ///         data_f32::blocked:aBcd16b:f0,alg:eltwise_relu,1000x96x55x55,34.417
    /// mkldnn_verbose,exec,reorder,jit:uni,undef,src_f32::blocked:aBcd16b:f0
    ///         dst_f32::blocked:abcd:f0,num:1,1000x96x55x55,97.3352
    /// ~~~
    /// This inference implementation is closer to best practices than
    /// *naive implementation* because it uses Intel MKL-DNN recommended memory
    /// format. *fused implementation* will futher optimize the performance by
    /// using a fused version of the conv + ReLU primitive emplying the Intel
    /// MKL-DNN [post-ops attribute](@ref dev_guide_attributes_post_ops)
    // reorder data to the user's format if needed.
}

// Implementation for convolution on blocked format for data and
// weights and the relu operation fused via a post-op attribute added to the
// convolution prim_descriptor
void conv_relu_fused(memory &user_src, memory &user_wei, memory &user_dst,
        const engine &eng, const stream &s) {
    /// @section performance_profiling_cpp_implementation3 Fused Implementation
    /// This implementation is launched with the following shell code:
    /// ~~~sh
    /// ./program.exe fused
    /// ~~~
    /// The program will call the implementation defined in the function
    /// `conv_relu_fused()`.
    /// @page performance_profiling_cpp
    ///
    /// First the memory descriptors and convolution descriptor are created as
    /// in *naive implementation*.
    // copy the dimensions and format from user's memory
    auto conv_src_md = memory::desc(user_src.get_desc());
    auto conv_wei_md = memory::desc(user_wei.get_desc());
    auto conv_dst_md = memory::desc(user_dst.get_desc());

    // reset format to any to allow convolution to pick the best implementation
    conv_src_md.data.format_kind = mkldnn_format_kind_any;
    conv_wei_md.data.format_kind = mkldnn_format_kind_any;
    conv_dst_md.data.format_kind = mkldnn_format_kind_any;

    // create a convolution descriptor
    auto conv_d = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_wei_md,
            conv_dst_md, strides, padding, padding);
    /// Then in preparation for the convolution prim desctiptor, a ReLU post-op
    /// is built and added to the primitive attribute `attr`:
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create post_op attr with relu

    // Next the convolution prim descriptor is created, which inherits the ReLU
    /// post-op by way of the attributes `attr`:
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create prim_desc with attr
    // [Create prim_desc with attr]
    // create an attribute for fused relu
    auto attr = create_attr_with_relu_post_op();

    // create a convolution primitive descriptor
    auto conv_pd = convolution_forward::primitive_desc(conv_d, attr, eng);
    // [Create prim_desc with attr]
    /// Then conditional reorders are applied as in *blocked format
    /// implementation* to convert `user_` format NCHW to blocked. Finally, it
    /// creates the convolution primitive `conv` and adds it to the stream `s`
    /// with the reordered data (`conv_src`, `conv_wei`, `conv_dst1`).
    // prepare convolution source
    memory conv_src = user_src;
    if (conv_pd.src_desc() != user_src.get_desc()) {
        conv_src = memory(conv_pd.src_desc(), eng);
        auto r_pd = reorder::primitive_desc(user_src, conv_src);
        reorder(r_pd).execute(s, user_src, conv_src);
    }

    // prepare convolution weights
    memory conv_wei = user_wei;
    if (conv_pd.weights_desc() != user_wei.get_desc()) {
        conv_wei = memory(conv_pd.weights_desc(), eng);
        auto r_pd = reorder::primitive_desc(user_wei, conv_wei);
        reorder(r_pd).execute(s, user_wei, conv_wei);
    }

    // prepare convolution destination
    memory conv_dst = user_dst;
    if (conv_pd.dst_desc() != user_dst.get_desc())
        conv_dst = memory(conv_pd.dst_desc(), eng);
    /// @page performance_profiling_cpp
    /// @note There is no separate addition to the stream for the ReLU
    /// operation because it has been added as a post-op to the `conv` primitive.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create conv_primitive implementation3
    // [Create conv_primitive implementation3]
    // create convolution primitive
    auto conv = convolution_forward(conv_pd);
    // [Create conv_primitive implementation3]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Add to stream implementation3
    // [Add to stream implementation3]
    // execute convolution by adding it to the stream s
    conv.execute(s,
            {{MKLDNN_ARG_SRC, conv_src}, {MKLDNN_ARG_WEIGHTS, conv_wei},
                    {MKLDNN_ARG_DST, conv_dst}});
    // [Add to stream implementation3]
    // reorder data to user's format if needed
    if (conv_pd.dst_desc() != user_dst.get_desc()) {
        auto r_pd = reorder::primitive_desc(conv_dst, user_dst);
        reorder(r_pd).execute(s, conv_dst, user_dst);
    }
    /// @page performance_profiling_cpp
    /// This implementation complies with best practices for f32 inference by
    /// using the Intel MKL-DNN recommended blocked format for convolution and
    /// adding ReLU as a post-op to execute a fused version of conv + ReLU.
    /// The consequence to following best practices can be seen in the execution
    /// time of the fused primitive of 103.9 milliseconds.
    ///
    /// *MKLDNN_VERBOSE output (see configuration notice\*):*
    /// ~~~sh
    /// mkldnn_verbose,exec,convolution,jit:avx512_common,forward_inference,
    ///         src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0
    ///         dst_f32::blocked:aBcd16b:f0,alg:convolution_direct,
    ///         mb1000_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,103.916
    /// ~~~
}

/// @page performance_profiling_cpp
/// @section performance_profiling_cpp_roundup Performance summary
///
/// | Implmentation | Time, ms | Cumulative speedup |
/// | :--            |     --: |                --: |
/// | Naive          |   336.1 |               1.0  |
/// | Blocked format |   154.0 |               2.2 |
/// | Fused          |   103.9 |               3.2 |
///
/// **  **
/// @page performance_profiling_cpp
/// @section performance_profiling_cpp_config Configuration Notice
/// @note This example is meant to demonstrate Intel MKL-DNN best practices.
/// @note It is not meant for benchmarking purposes. The platform is not fully
/// @note optimized, so the primitive execution times are only relevant in
/// @note relation to the other times in this example.
///
/// Runtime Settings:
/// * OMP_NUM_THREADS=14
/// * KMP_AFFINITY=granularity=fine,compact,1,0
///
/// Platform:
/// * CPU: Intel(R) Xeon(R) Platinum 8180M CPU @ 2.50GHz
/// * Thread(s) per core:    2
/// * Core(s) per socket:    28
/// * Socket(s):             2
/// * NUMA node(s):          2
/// * RAM (DDR4): 1.45 TB

int main(int argc, char *argv[]) {
    // Initialize engine
    engine::kind engine_kind = parse_engine_kind(argc, argv, 1);
    engine eng(engine_kind, 0);

    // Initialize stream
    stream s(eng);
    // [Set dimensions]
    // set dimensions for synthetic data and weights
    const memory::dim BATCH = 1;
    const memory::dim IC = 3, OC = 96;
    const memory::dim IH = 227, KH = 11, OH = 55;
    const memory::dim IW = 227, KW = 11, OW = 55;
    // [Set dimensions]

    // [Create memory objects]
    // create MKL-DNN memory objects for user's tensors (in nchw and oihw formats)
    // @note here the library allocates memory
    auto user_src = memory({{BATCH, IC, IH, IW}, memory::data_type::f32,
                                   memory::format_tag::nchw},
            eng);
    auto user_wei = memory({{OC, IC, KH, KW}, memory::data_type::f32,
                                   memory::format_tag::oihw},
            eng);
    auto user_dst = memory({{BATCH, OC, OH, OW}, memory::data_type::f32,
                                   memory::format_tag::nchw},
            eng);
    // [Create memory objects]

    // fill source, destination, and weights with synthetic data
    init_data(user_src, 1);
    init_data(user_dst, -1);
    init_data(user_wei, .5);

    // set implementation ("naive"||"blocked"||"fused") setting implementation
    // to "validation" will run all implementations
    std::string implementation;
    if (argc <= 2)
        implementation = "validation";
    else if (argc == 3)
        implementation = argv[2];

    if (!(implementation == "validation" || implementation == "naive"
                || implementation == "blocked" || implementation == "fused")) {
        std::cout << "The implementation can be one of:\n";
        std::cout << " - naive: NCHW format without fusion\n";
        std::cout << " - blocked: format propagation without fusion\n";
        std::cout << " - fused: format propagation with fusion\n";
        std::cout << " - validation: runs all implementations\n\n";
        std::cout << "Validation will run if no parameters are specified\n\n";

        return -1;
    }

    if (implementation == "naive" || implementation == "validation") {
        std::cout << "implementation: naive\n";
        // run conv + relu w/o fusing
        conv_relu_naive(user_src, user_wei, user_dst, eng, s);
        std::cout << "conv + relu w/ nchw format completed\n";
    }

    if (implementation == "blocked" || implementation == "validation") {
        std::cout << "implementation: blocked\n";
        // run conv + relu w/o fusing
        conv_relu_blocked(user_src, user_wei, user_dst, eng, s);
        std::cout << "conv + relu w/ blocked format completed\n";
    }

    if (implementation == "fused" || implementation == "validation") {
        std::cout << "implementation: fused\n";
        // run conv + relu w/ fusing
        conv_relu_fused(user_src, user_wei, user_dst, eng, s);
        std::cout << "conv + relu w/ fusing completed\n";
    }

    return 0;
}
