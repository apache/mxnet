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
/// @copybrief performance_profiling_cpp
/// > Annotated version: @ref performance_profiling_cpp

/// @page performance_profiling_cpp Performance Profiling Example
/// This example demonstrates the best practices for application performance
/// optimizations with oneDNN.
///
/// > Example code: @ref performance_profiling.cpp
///
/// This example uses [DNNL_VERBOSE](@ref dev_guide_verbose) trace output
/// to tune oneDNN code to align
/// with the [best practices](@ref dev_guide_inference).
///
/// It assumes knowledge of memory formats and their usage in
/// oneDNN. You can read more about this topic
/// [here](@ref memory_format_propagation_cpp).
///
/// Additionally, see the [article for recommended environment for
/// running benchmarks](@ref dev_guide_performance_settings).
///
/// The example has three different implementations of the mathematical
/// operation:
/// 1. *Naive implementation* executes 2D convolution followed by
/// ReLU on the data in **NCHW** format. This implementation
/// does not align with oneDNN best practices and results in
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
/// ./program.exe [cpu|gpu] [implementation]
/// ~~~
///
/// Before you run the program, set your `DNNL_VERBOSE` environment
/// variable to 1:
/// ~~~sh
/// export DNNL_VERBOSE=1
/// ~~~
///
/// The program starts by creating oneDNN memory objects in **NCHW**
/// format. These are called `user_` because they are meant to represent the
/// user's source data entering oneDNN with the NCHW format.
/// @page performance_profiling_cpp
/// @snippet performance_profiling.cpp Set dimensions
/// @page performance_profiling_cpp
/// @note Here the library allocates memory.
/// @page performance_profiling_cpp
/// @snippet performance_profiling.cpp Create memory objects
/// @page performance_profiling_cpp
/// @note You can change the batch size to easily increase/decrease the workload.
///
/// The following descriptions of each implementation will reference each other,
/// and are meant to be read in order.
///

#include <iostream>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

// [Prologue]

// Set Strides and Padding
const memory::dims strides = {4, 4};
const memory::dims padding = {0, 0};

// [Prologue]
//
// function to init data
void init_data(memory &m, float v) {
    size_t size = m.get_desc().get_size() / sizeof(float);
    std::vector<float> data(size, v);
    write_to_dnnl_memory(data.data(), m);
}

// function to execute non-fused relu
void create_and_execute_relu(memory &data, engine &eng, stream &s) {
    // relu operates on whatever data format is given to it

    // create a primitive
    auto relu_d = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, data.get_desc(), 0.f, 0.f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    auto relu = eltwise_forward(relu_pd);

    // execute it (in-place)
    relu.execute(s, {{DNNL_ARG_SRC, data}, {DNNL_ARG_DST, data}});
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
        memory user_dst, engine &eng, stream &s) {
    /// @section performance_profiling_cpp_implementation1 Naive Implementation
    /// This implementation is launched with the following shell code:
    /// ~~~sh
    /// ./program.exe cpu naive
    /// ~~~
    /// The program will call the implementation defined in the function
    /// `conv_relu_naive()`.
    ///
    /// First it sets the dimensions and format for convolution memory
    /// descriptors (`_md`) to match `user_` values--one `md` each for source,
    /// destination, and weight data. Then it uses those `md` to create the
    /// convolution descriptor `conv_d`, which tells oneDNN to use
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
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Add to stream
    // [Add to stream]
    // execute convolution by adding it to the stream s
    conv.execute(s,
            {{DNNL_ARG_SRC, user_src}, {DNNL_ARG_WEIGHTS, user_wei},
                    {DNNL_ARG_DST, user_dst}});
    // [Add to stream]
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create and execute relu
    // [Create and execute relu]
    // execute relu (on convolution's destination format, whatever it is)
    create_and_execute_relu(user_dst, eng, s);
    s.wait();
    // [Create and execute relu]
    /// @page performance_profiling_cpp
    /// @note The function for creation and execution of ReLU primitive is
    /// defined elsewhere to keep this example clean. It is an non-intensive
    /// operation, so the `create_and_execute_relu()` function uses whatever
    /// the input data format is at the time it is called.
    ///
    /// Using NCHW data format may result in suboptimal performance for compute
    /// intensive primitives, as shown in the following DNNL_VERBOSE output
    /// by the convolution and relu execution
    /// times of 38.3 and 2.9 milliseconds, respectively.
    ///
    /// *DNNL_VERBOSE output (see configuration notice\*):*
    /// ~~~sh
    /// dnnl_verbose,exec,cpu,convolution,gemm:jit,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_undef::undef::f0 dst_f32::blocked:abcd:f0,,alg:convolution_direct,mb128_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,38.314
    /// dnnl_verbose,exec,cpu,eltwise,jit:avx512_common,forward_inference,data_f32::blocked:abcd:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,128x96x55x55,2.87695
    /// ~~~
    /// In *Blocked format implementation*, we will incorporate the best
    /// practice of letting oneDNN determine the optimal format
    /// for convolution primitive.
}

// Implementation for convolution on blocked format for data and
// weights, followed by execution of non-fused relu
void conv_relu_blocked(memory user_src, memory user_wei, memory user_dst,
        engine &eng, stream &s) {
    /// @page performance_profiling_cpp
    /// @section performance_profiling_cpp_implementation2 Blocked format implementation
    /// This implementation is launched with the following shell code:
    /// ~~~sh
    /// ./program.exe cpu blocked
    /// ~~~
    /// The program will call the implementation defined in the function
    /// `conv_relu_blocked()`.
    ///
    /// First it creates the md as in **naive implementation**. Next it changes
    /// the dnnl::memory::format_tag for each md to `ANY`. Then it uses those
    /// md to create the convolution descriptor conv_d, which tells oneDNN
    /// to use whatever format it recommends for the convolution.
    /// oneDNN will choose a friendly blocked format.
    /// @page performance_profiling_cpp
    /// @snippet performance_profiling.cpp Create mem_desc with tag=any
    // [Create mem_desc with tag=any]
    // copy the dimensions and format from user's memory
    auto conv_src_md = memory::desc(user_src.get_desc());
    auto conv_wei_md = memory::desc(user_wei.get_desc());
    auto conv_dst_md = memory::desc(user_dst.get_desc());

    // reset format to "any" to allow convolution to pick the best implementation
    conv_src_md.data.format_kind = dnnl_format_kind_any;
    conv_wei_md.data.format_kind = dnnl_format_kind_any;
    conv_dst_md.data.format_kind = dnnl_format_kind_any;
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
    /// @note The reoders are applied using oneDNN `reorder` primitive.
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
            {{DNNL_ARG_SRC, conv_src}, {DNNL_ARG_WEIGHTS, conv_wei},
                    {DNNL_ARG_DST, conv_dst}});
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
    s.wait();
    /// @page performance_profiling_cpp
    /// Blocked memory format is recommended for oneDNN primitive
    /// execution and provides better performance, as shown in the
    /// DNNL_VERBOSE output by the convolution and relu execution times of
    /// 18.3 and 2.7 milliseconds (down from 38.3 and 2.9 in
    /// *naive implementation*), respectively.
    /// In this implementation, there is an additional reorder operation that
    /// executes before and after the the conv + relu. This small cost is worth
    /// the gain from executing in blocked format. If fact, it becomes
    /// negligible when chaining together multiple oneDNN operations in
    /// succession. In these situations, you can do one reorder at the beginning
    /// and one at the end of the chain, and only pay the reorder penalty at
    /// those points in the execution.
    ///
    /// *DNNL_VERBOSE output (see configuration notice\*):*
    /// ~~~sh
    /// dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb16a:f0,,,96x3x11x11,0.0310059
    /// dnnl_verbose,exec,cpu,convolution,jit:avx512_common,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,,alg:convolution_direct,mb128_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,18.3101
    /// dnnl_verbose,exec,cpu,eltwise,jit:avx512_common,forward_inference,data_f32::blocked:aBcd16b:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,128x96x55x55,2.66895
    /// dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,128x96x55x55,4.80396
    /// ~~~
    /// This inference implementation is closer to best practices than
    /// *naive implementation* because it uses oneDNN recommended memory
    /// format. *fused implementation* will futher optimize the performance by
    /// fusing convolution with ReLU using oneDNN
    /// [post-ops](@ref dev_guide_attributes_post_ops).
    // reorder data to the user's format if needed.
}

// Implementation for convolution on blocked format for data and
// weights and the relu operation fused via a post-op attribute added to the
// convolution prim_descriptor
void conv_relu_fused(memory user_src, memory user_wei, memory user_dst,
        const engine &eng, stream &s) {
    /// @section performance_profiling_cpp_implementation3 Fused Implementation
    /// This implementation is launched with the following shell code:
    /// ~~~sh
    /// ./program.exe cpu fused
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
    conv_src_md.data.format_kind = dnnl_format_kind_any;
    conv_wei_md.data.format_kind = dnnl_format_kind_any;
    conv_dst_md.data.format_kind = dnnl_format_kind_any;

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
            {{DNNL_ARG_SRC, conv_src}, {DNNL_ARG_WEIGHTS, conv_wei},
                    {DNNL_ARG_DST, conv_dst}});
    // [Add to stream implementation3]
    // reorder data to user's format if needed
    if (conv_pd.dst_desc() != user_dst.get_desc()) {
        auto r_pd = reorder::primitive_desc(conv_dst, user_dst);
        reorder(r_pd).execute(s, conv_dst, user_dst);
    }
    s.wait();
    /// @page performance_profiling_cpp
    /// This implementation complies with best practices for f32 inference by
    /// using the oneDNN recommended blocked format for convolution and
    /// adding ReLU as a post-op to execute a fused version of conv + ReLU.
    /// The consequence to following best practices can be seen in the execution
    /// time of the fused primitive of 18.0 milliseconds.
    ///
    /// *DNNL_VERBOSE output (see configuration notice\*):*
    /// ~~~sh
    /// dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb16a:f0,,,96x3x11x11,0.0148926
    /// dnnl_verbose,exec,cpu,convolution,jit:avx512_common,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,post_ops:'eltwise_relu;';,alg:convolution_direct,mb128_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,17.968
    /// dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,128x96x55x55,4.66797
    /// ~~~
}

/// @page performance_profiling_cpp
/// @section performance_profiling_cpp_roundup Performance summary
///
/// | Implementation | Time, ms | Cumulative speedup |
/// | :--            |      --: |                --: |
/// | Naive          |     41.2 |                1.0 |
/// | Blocked format |     21.0 |                2.0 |
/// | Fused          |     18.0 |                2.3 |
///
/// **  **
/// @page performance_profiling_cpp
/// @section performance_profiling_cpp_config Configuration Notice
/// @note This example is meant to demonstrate oneDNN best practices.
/// @note It is not meant for benchmarking purposes. The platform is not fully
/// @note optimized, so the primitive execution times are only relevant in
/// @note relation to the other times in this example.
///
/// Runtime Settings:
/// * OMP_NUM_THREADS=14
/// * KMP_AFFINITY=granularity=fine,compact
///
/// Platform:
/// * CPU: Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
/// * Thread(s) per core:    1
/// * Core(s) per socket:    28
/// * Socket(s):             2
/// * NUMA node(s):          2
/// * RAM (DDR4): 192 GB

void performance_profiling(engine::kind engine_kind, int argc, char **argv) {
    // Initialize engine
    engine eng(engine_kind, 0);

    // Initialize stream
    stream s(eng);
    // [Set dimensions]
    // set dimensions for synthetic data and weights
    const memory::dim BATCH = 128;
    const memory::dim IC = 3, OC = 96;
    const memory::dim IH = 227, KH = 11, OH = 55;
    const memory::dim IW = 227, KW = 11, OW = 55;
    // [Set dimensions]

    // [Create memory objects]
    // create oneDNN memory objects for user's tensors (in nchw and oihw formats)
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
        std::cout << "Validation will run if no parameters are specified.\n\n";

        throw std::invalid_argument("Incorrect input arguments.");
    }

    if (implementation == "naive" || implementation == "validation") {
        std::cout << "Implementation: naive.\n";
        // run conv + relu w/o fusing
        conv_relu_naive(user_src, user_wei, user_dst, eng, s);
        std::cout << "Conv + ReLU w/ nchw format completed.\n";
    }

    if (implementation == "blocked" || implementation == "validation") {
        std::cout << "Implementation: blocked.\n";
        // run conv + relu w/o fusing
        conv_relu_blocked(user_src, user_wei, user_dst, eng, s);
        std::cout << "Conv + ReLU w/ blocked format completed.\n";
    }

    if (implementation == "fused" || implementation == "validation") {
        std::cout << "Implementation: fused.\n";
        // run conv + relu w/ fusing
        conv_relu_fused(user_src, user_wei, user_dst, eng, s);
        std::cout << "Conv + ReLU w/ fusing completed.\n";
    }
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv, 1);
    return handle_example_errors(
            performance_profiling, engine_kind, argc, argv);
}
