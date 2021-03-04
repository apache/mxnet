Transition from v0.x to v1.x {#dev_guide_transition_to_v1}
==========================================================

> **NOTE**
>
> Starting with version 1.4
> Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
> is renamed to oneAPI Deep Neural Network Library (oneDNN).
> For consistency, only this guide uses Intel MKL-DNN nomenclature.

## Introduction

This article describes user-visible and some important internal changes to
Intel MKL-DNN that occurred between v0.20 and v1.0.

The v0.x branch ([mnt-v0](https://github.com/oneapi-src/oneDNN/tree/mnt-v0)) is
deprecated and users are strongly encouraged to migrate to
[v1.x](https://github.com/oneapi-src/oneDNN).

@sa
Discussion on the API changes occurred in PR #384:
[RFC: API changes for the upcoming v1.0](https://github.com/oneapi-src/oneDNN/pull/384).

## Summary of Changes

We tried to keep changes minimal to make migration as simple as possible. In
particular, the Intel MKL-DNN programming model stays the same. Nevertheless,
the new version brings a lot of incompatible changes requiring developers to
revisit significant portions of the integrated code.

All changes can be split into the following groups:
1. Minor API changes
2. Improving the library robustness
3. Simplified execution model
4. Changes in memory description
5. Changes in the build system

These groups are discussed in detail below.

## 1. Minor API Changes

### 1.1. Remove deprecated functionality

| Deprecated functionality               | Replacement
| :---                                   | :---
| ReLU primitive                         | [Eltwise](@ref dnnl::eltwise_forward) with algorithm kind [ReLU](@ref dnnl::algorithm::eltwise_relu)
| ConvolutionReLU (single primitive)     | Convolution with ReLU as a [post operation](@ref dev_guide_attributes_post_ops)
| Double precision scales                | Single precision scales
| RNN backward pd w/o forward pd hint    | RNN backward pd w/ forward pd hint
| `mkldnn_omit_stats` batch norm. flag   | `mkldnn_use_global_stats`
| `mkldnn_eltwise_desc_t.negative_slope` | `mkldnn_eltwise_desc_t.alpha`
| `mkldnn_rnn_cell_flags_t`              | Not available anymore -- RNN primitives are separated into RNN, LSTM, and GRU
| `mkldnn_padding_kind_t`                | Not used anymore

The complete list of the removed C functions:
~~~cpp
    mkldnn_relu_forward_desc_init(...);
    mkldnn_relu_backward_desc_init(...);
    mkldnn_convolution_relu_desc_init(...);
    mkldnn_rnn_cell_desc_init(...);
    mkldnn_rnn_cell_get_gates_count(...);
    mkldnn_rnn_cell_get_states_count(...);
    mkldnn_rnn_forward_desc_init(...);
    mkldnn_rnn_backward_desc_init(...);
~~~

The complete list of the removed C++ classes and functions:
~~~cpp
    struct mkldnn::convolution_relu_forward {}
    struct mkldnn::relu_forward {}
    struct mkldnn::relu_backward {}
    struct mkldnn::rnn_cell {}
    struct mkldnn::rnn_forward {}
    struct mkldnn::rnn_backward {}

    mkldnn::sum::primitive_desc(const memory::desc &output, std::vector<double> scale, std::vector<memory::primitive_desc> inputs);
    mkldnn::sum::primitive_desc(std::vector<double> scale, std::vector<memory::primitive_desc> inputs);
    mkldnn::eltwise_forward::desc(prop_kind aprop_kind, const memory::desc &src_desc, T negative_slope);
    mkldnn::eltwise_backward::desc(const memory::desc &diff_data_desc, const memory::desc &data_desc, T negative_slope);
~~~

### 1.2. Rename `foo_v2()` to `foo()` and remove old `foo()` (C API only)

The functions like:
~~~cpp
    mkldnn_primitive_desc_create_v2(...);
~~~
were renamed to:
~~~cpp
    mkldnn_primitive_desc_create(...);
~~~

In v0.x, the `foo_v2()` functions typically were used to pass
[attributes](@ref dev_guide_attributes), and `foo()` assumed empty attributes.
In v1.0, the attributes parameter is mandatory. A user can still pass `NULL` to
indicate that the default (empty) attributes should be used.

The list of functions that had the `_v2` suffix:

~~~cpp
    mkldnn_primitive_desc_iterator_create_v2(...);
    mkldnn_primitive_desc_create_v2(...);
    mkldnn_reorder_primitive_desc_create_v2(...);
~~~

### 1.3. Remove s16 (int16_t) data type support

The experimental `s16` data type is not supported any more and has been dropped.

### 1.4. Disallow setting the rounding mode

Rounding mode that was a part of attributes has been dropped. All computations
respect the MXCSR register when performing rounding. Unless the rounding mode is
set explicitly, rounding to the nearest even integer (RNE) is used.

### 1.5. Rename a few types, enumerations, and functions

#### 1.5.1. Types

| API | v0.x                              | v1.0
| :-- | :--                               | :--
| C   | mkldnn_batch_normalization_flag_t | [mkldnn_normalization_flags_t](@ref dnnl_normalization_flags_t)
| C   | mkldnn_format_t                   | [mkldnn_format_tag_t](@ref dnnl_format_tag_t)
| C++ | mkldnn::batch_normalization_flag  | [mkldnn::normalization_flags](@ref dnnl::normalization_flags::use_global_stats)
| C++ | mkldnn::memory::format            | [mkldnn::memory::format_tag](@ref dnnl::memory::format_tag)

#### 1.5.2. Enumerations

| API | v0.x                   | v1.0
| :-- | :--                    | :--
| C   | mkldnn_fuse_bn_relu    | [mkldnn_fuse_norm_relu](@ref dnnl_fuse_norm_relu)
| C++ | mkldnn::fuse_bn_relu   | [mkldnn::normalization_flags::fuse_norm_relu](@ref dnnl::normalization_flags::fuse_norm_relu)
| C++ | mkldnn::query::eengine | [mkldnn::query::engine](@ref dnnl::query::engine)

#### 1.5.3. Functions

| API | v0.x                      | v1.0
| :-- | :--                       | :--
| C   | mkldnn_memory_desc_init() | [mkldnn_memory_desc_init_by_tag()](@ref dnnl_memory_desc_init_by_tag)

### 1.6. Unscoped enumerations become scoped (C++ API only)

All `enum` became `enum class`. This requires the following changes:

| Type                        | Value in v0.x             | Value in v1.0
| :--                         | :--                       | :--
| mkldnn::prop_kind           | mkldnn::forward_inference | [mkldnn::prop_kind::forward_inference](@ref dnnl::prop_kind::forward_inference)
| mkldnn::algorithm           | mkldnn::eltwise_tanh      | [mkldnn::algorithm::eltwise_tanh](@ref dnnl::algorithm::eltwise_tanh)
| mkldnn::normalization_flags | mkldnn::fuse_bn_norm_relu | [mkldnn::normalization_flags::fuse_norm_relu](@ref dnnl::normalization_flags::fuse_norm_relu)
| mkldnn::query               | mkldnn::eengine           | [mkldnn::query::engine](@ref dnnl::query::engine)
| mkldnn::memory::data_type   | mkldnn::memory::f32       | [mkldnn::memory::data_type::f32](@ref dnnl::memory::data_type::f32)
| mkldnn::memory::format_tag  | mkldnn::memory::nchw      | [mkldnn::memory::format_tag::nchw](@ref dnnl::memory::format_tag::nchw)

### 1.7. Remove view primitive

Version 0.x had an implementation of view that was simply an alias for memory.
In Intel MKL-DNN v1.0, we removed view as a type and replaced it with a
memory descriptor directly. In order to initialize sub-memory, use
[mkldnn::memory::desc::submemory_desc()](@ref dnnl::memory::desc::submemory_desc()).

@sa
For more detail, refer to section
[4. View rework](https://github.com/oneapi-src/oneDNN/tree/rfc-api-changes-v1.0/doc/rfc/api-v1.0#4-view-rework)
of the [RFC for v1.0](https://github.com/oneapi-src/oneDNN/pull/384).

### 1.8. RNN-specific changes

Each type of [RNN](@ref dnnl_api_rnn) (Vanilla RNN, LSTM, and two types of GRU)
is now initialized by a separate function/operation descriptor constructor.

For instance, instead of using mkldnn::rnn_forward with specified RNN types
a user is expected to use:
- [mkldnn::vanilla_rnn_forward](@ref dnnl::vanilla_rnn_forward) for Vanilla RNN
- [mkldnn::lstm_forward](@ref dnnl::lstm_forward) for LSTM
- [mkldnn::gru_forward](@ref dnnl::gru_forward) for GRU
- [mkldnn::lbr_gru_forward](@ref dnnl::lbr_gru_forward) for the linear-before-reset variant of GRU

Also, the hidden and cell states in LSTM are now separated. This means that
instead of one `src_iter` tensor of shape
`(layers, directions, states, batch, channels)` a user passes
`src_iter` tensor of shape `(layers, directions, batch, channels)` for hidden
states and
`src_iter_c` tensor of shape `(layers, directions, batch, channels)` for cell
states.
The same applies to `dst_iter`; the hidden state and the cell state are split
into `dst_iter` and `dst_iter_c` respectively.

### 1.9. GEMM API changes

Intel MKL-DNN provides three GEMM-like functions:
- [mkldnn_sgemm()](@ref dnnl_sgemm) -- Single precision matrix-matrix multiply
- [mkldnn_gemm_u8s8s32()](@ref dnnl_gemm_u8s8s32) -- u8/s8 integer matrix-matrix multiply
- [mkldnn_gemm_s8s8s32()](@ref dnnl_gemm_s8s8s32) -- s8/s8 integer matrix-matrix multiply

With version 1.0 we switched from a Fortran-style to a C-style API, meaning that
the parameters are passed by value rather than by address, and matrices are
assumed to be in row-major format rather than column-major format.

Moreover, to broaden the applicability of integer matrix-matrix multiply
functions we changed the formula from:
\f[
    C_{s32} =
        \alpha
        \cdot
        (op(A_{i8}) + o_A) \cdot (op(B_{s8}) + o_B)
        + \beta \cdot C_{s32}
        + o_C
\f]
to
\f[
    C_{s32} =
        \alpha
        \cdot
        (op(A_{i8}) - o_A) \cdot (op(B_{s8}) - o_B)
        + \beta \cdot C_{s32}
        + o_C
\f]

where for both [mkldnn_gemm_u8s8s32()](@ref dnnl_gemm_u8s8s32) and
[mkldnn_gemm_s8s8s32()](@ref dnnl_gemm_s8s8s32) the types of
offsets for matrices A and B correspond to the type of the matrices themselves;
that is:
- `typeof(o_A) == typeof(*A)` and
- `typeof(o_B) == typeof(*B)`.

### 1.10. Primitive descriptor queries for memory descriptors

In version 0.x when querying the primitive descriptor for a memory descriptor
that is not used, the C API returned NULL and the C++ API threw an exception. In
version 1.0, both the C and C++ APIs return a zero memory descriptor.

Zero memory descriptor means that the number of dimensions equals 0 and all the
fields are set to zero. A memory object created with such a memory descriptor
does not require any buffer allocations.

These changes enable simplifying the code that handles
[workspace](@ref dev_guide_inference_and_training_aspects_workspace) or
[scratchpad](@ref dev_guide_attributes_scratchpad):

~~~cpp
    // The code works fine even if scratchpad is not required.
    // In this case the memory would be just zero memory.

    auto scratchpad_md = pd.scratchpad_desc();
    auto scratchpad = memory(scratchpad_md, pd.get_engine());

    primitive.execute(stream, {
        ...,
        {MKLDNN_SCRATCHPAD, scratchpad}};
~~~

### 1.11. Default constructors for C++ classes (C++ API only)

In Intel MKL-DNN v1.0, all C++ objects (primitives, memory objects, engines,
and streams) now have default empty constructors. This enables defining the
object, and then initializing it later on. An attempt to use any methods of an
uninitialized object will result in the throwing of an exception.

This improvement can be especially useful when Intel MKL-DNN objects are
members of the user's classes. For example:

~~~cpp
    class RELU_layer {
    public:
        RELU_layer() {} // no need to initialize eltwise here

        void init() {
            ...
            // deferred initialization
            eltwise = eltwise_forward(...);
        }

    private:
        eltwise_forward eltwise;
    };
~~~

## 2. Improving the Library Robustness

### 2.1. Memory allocation in the C API

In Intel MKL-DNN v1.0, constructing a memory object using special value
`MKLDNN_MEMORY_ALLOCATE` for a handle results in the buffer being allocated by
the library. This makes the behavior of the C API memory object constructor
aligned with its C++ API `mkldnn::memory` counterpart. Note that the C++ API
memory object class still has an extra constructor that does not take a handle
at all, and asks the library to allocate the buffer (that is, the same behavior
as calling with the handle equal to `MKLDNN_MEMORY_ALLOCATE`).

### 2.2. Explicit scratchpad management

Intel MKL-DNN primitives may require temporary
[scratchpad memory](@ref dev_guide_attributes_scratchpad) for storing
intermediate computational results. For instance, convolution backward by
weights typically requires extra space to perform a reduction of the
`diff_weights` computed by different threads (the work is divided across
images). Starting with version 1.0, the library supports two modes:
1. Implicit scratchpad, managed by the library (**default**).
   See [mkldnn::scratchpad_mode::library](#dnnl::scratchpad_mode::library).
2. Explicit scratchpad, provided by the user.
   See [mkldnn::scratchpad_mode::user](#dnnl::scratchpad_mode::user).

The former mode matches the behavior of Intel MKL-DNN v0.x. It is kept for
user convenience and cases in which memory is not a concern.

In the explicit scratchpad mode, a new `mkldnn_query_scratchpad_md` query will
return the amount of scratchpad memory needed for a primitive, and the user
will be responsible for allocating and providing the scratchpad memory to a
primitive at runtime. The explicit scratchpad mode should be *explicitly*
enabled by passing an attribute with `mkldnn::scratchpad_mode::user` to
primitive descriptors.

@warning
[Scratchpad](@ref dev_guide_attributes_scratchpad) memory is not the same as
[workspace](@ref dev_guide_inference_and_training_aspects_workspace).

With explicit scratchpad it is possible to make Intel MKL-DNN primitives
stateless and hence thread safe: the same primitive can be executed in multiple
independent threads as long as different threads use different scratchpads.

However, if a user chooses implicit scratchpad mode, there is no thread-safety
guarantee.

## 3. Simplified Execution Model

This is the most notable change in the library. The main idea was to change the
execution API so that memory arguments are specified at primitive execution time
and not at primitive creation time. This leads to the following changes.

### 3.1. Memory is not a primitive anymore

In version 0.x, memory had a type of primitive. With the new API, memory becomes
a distinct data type. Moreover, a memory primitive descriptor becomes redundant
and has been dropped. The functions that use memory primitive descriptors now
take memory descriptor and (optionally) engine, if the latter cannot be
inferred.

These changes bring new data types and functions, such as:

~~~cpp
    #define MKLDNN_NATIVE_HANDLE_ALLOCATE  ((void *)-1)
    #define MKLDNN_NATIVE_HANDLE_NONE      ((void *)0)

    struct mkldnn_memory_t; // memory type, no more equal to mkldnn_primitive_t

    // create a memory
    // native_handle can:
    //  - point to the user allocated memory, i.e. valid handle. In this case the
    //    library does not own allocated memory.
    //  - be MKLDNN_NATIVE_HANDLE_ALLOCATE to ask the library to allocate and
    //    attach memory. In this case the library owns allocated memory.
    //  - be MKLDNN_NATIVE_HANDLE_NONE to create mkldnn_memory w/o attached memory.
    mkldnn_status_t mkldnn_memory_create(mkldnn_memory_t *mem,
            const mkldnn_memory_desc_t *md, mkldnn_engine_t engine,
            void *handle);
~~~

### 3.2. Operation primitives cannot be used as inputs (use memory instead)

Version 0.x allowed passing an operation primitive as an input to another
primitive. For instance, a convolution primitive could be passed as an input to
a consequent ReLU. During the execution the ReLU primitive queried the
convolution for its output memory and used it as an input.

In version 1.0, users are allowed to pass only memory type as inputs and outputs
for primitives.

### 3.3. Remove the `mkldnn_primitive_at_t` type

Another consequence is that `mkldnn_primitive_at_t`, which is logically
equivalent to `{primitive, output_index}`, becomes redundant. Previously the
type was used to specify the exact memory to use (if a primitive had several
outputs).

### 3.4. Passing stream and input/output memories at primitive execution

Finally, users are now able to directly run primitives by calling an `execute`
function instead of putting primitives into a stream and running the latter.
This change affects how primitives interact with streams and input/output
memory objects: with the new API they become arguments to be passed to the
primitive execution function.

The change significantly simplifies primitive creation, which now requires a
primitive descriptor only:

~~~cpp
    mkldnn_status_t mkldnn_primitive_create(mkldnn_primitive_t *primitive,
            const_mkldnn_primitive_desc_t *pd);
~~~

To remove the ambiguity in which order input and output memories need to be
passed, we introduced a map-like argument in which each memory argument is
paired with a tag indicating what kind of argument it is: destination, source,
weights, and so on.

~~~cpp
    // types
    #define MKLDNN_ARG_SRC_0 1
    #define MKLDNN_ARG_SRC   MKLDNN_ARG_SRC_0
    #define MKLDNN_ARG_FROM  MKLDNN_ARG_SRC_0
    // ...

    // C API
    typedef struct {
        int arg; // MKLDNN_ARG_SRC, ...
        mkldnn_memory_t memory;
    } mkldnn_exec_arg_t;

    mkldnn_status_t mkldnn_primitive_execute(mkldnn_primitive_t prim,
            mkldnn_stream_t stream, int nargs, const mkldnn_exec_arg_t *args);

    // C++ API
    convolution_forward::execute(mkldnn::stream &stream,
            const std::map<int, mkldnn::memory> &exec_args);
    // ... other primitives ...


    // example C, convolution forward w/ bias
    mkldnn_exec_arg_t conv_exec_args[] = {
        {MKLDNN_ARG_SRC, src_mem},
        {MKLDNN_ARG_WEIGHTS, weights_mem},
        {MKLDNN_ARG_BIAS, bias_mem},
        {MKLDNN_ARG_DST, dst_mem},
    };
    mkldnn_primitive_execute(conv_fwd, stream, 4, conv_exec_args);


    // example C++, in-place eltwise
    eltwise.execute(stream, {{MKLDNN_ARG_SRC, mem}, {MKLDNN_ARG_DST, mem}});
~~~

### 3.5 Short summary

The example below shows conceptual code transformations between versions. The
C++ API is used for brevity.

#### Version 0.x:
~~~cpp
    // create a convolution, specify all inputs and outputs
    auto conv = convolution(conv_pd,
                {src_mem, 0}, {wei_mem, 0}, dst_conv_mem);

    // create a relu (note that one of inputs is the convolution)
    auto relu = relu(relu_pd,
                {conv, 0}, dst_relu_mem);

    // create a stream, submit convolution and relu, and wait for the result
    stream().submit({conv, relu}).wait();
~~~

#### Version 1.0:
~~~cpp
    // create convolution and relu. no inputs/outputs
    auto conv = convolution(conv_pd);
    auto relu = relu(relu_pd);

    // create stream (based on engine)
    stream s(engine, 0);

    // execute the convolution with given inputs, outputs
    conv.execute(s, {
            {MKLDNN_ARG_SRC, src_mem},
            {MKLDNN_ARG_WEIGHTS, wei_mem},
            {MKLDNN_ARG_DST, dst_conv_mem}});

    // execute the relu. cannot pass convolution as an input, only memory is allowed
    relu.execute(s, {
            {{MKLDNN_ARG_SRC, dst_conv_mem},
            {MKLDNN_ARG_DST, dst_relu_mem}});

    s.wait(); // wait for async streams
~~~

## 4. Changes in Memory Description

The way of describing memory format in version 0.x had multiple issues. From
the user's perspective, the main issues were:
- Some memory formats were missing. For example, the `iohw` format was not
  available.
- There were multiple ambiguous ways to describe memory. For example, `oihw`
  described memory in the same way as `nchw`, but these formats were different
  (see [gh#153](https://github.com/oneapi-src/oneDNN/issues/153)).
- Support for custom formats was limited.
- Support for memory views was limited.

There were more substantial issues from the library development perspective:
code bloat to support special cases, etc.

We addressed the issues above by reworking memory descriptors. From the user's
perspective, the main changes are:
1. Memory descriptors support arbitrary strides for plain layouts. For
   example, initializing a memory descriptor with `strides={h*w, o*h*w, w, 1}`
   should be a valid way to define `iohw` format even if Intel MKL-DNN does not
   support it explicitly. Functions to use:
   - C++ API: [mkldnn::memory::desc::desc(dims, data_type, strides)](@ref dnnl::memory::desc::desc),
   - C API: [mkldnn_memory_desc_init_by_strides()](@ref dnnl_memory_desc_init_by_strides).
2. Dimensions are of type `int64_t` instead of int, and the maximum number
   of tensor dimensions is decreased from 16 to 12. The `mkldnn_strides_t`
   is removed; use `mkldnn_dims_t` instead.
3. The `memory_desc_t.format` field is replaced with
   `memory_desc_t.format_kind`, which also has different semantics.

While the first two items are self-explanatory, the last one requires some
elaboration.

In version 0.x, most memory formats could be described directly by using
appropriate format names (for example, `nchw`) that fully describe how data is
laid out in memory. However, Intel MKL-DNN also had the `blocked` memory format
and the corresponding `memory_desc_t.layout_desc.blocking_desc` structure,
which could describe a memory format in a unified fashion by specifying block
sizes and strides. The original idea was to use format tags like `nchw` during
memory descriptor initialization only, and always use the `blocked` format
internally. Unfortunately, that was never implemented.

With the new design, Intel MKL-DNN starts distinguishing between the actual
memory format and convenience memory format tags that can be used to describe
memory format concisely.

Users are still able to initialize memory descriptors with format tags like
`nchw` using [mkldnn::memory::desc::desc(dims, data_type, format_tag)](@ref dnnl::memory::desc::desc)
or [mkldnn_memory_desc_init_by_tag()](@ref dnnl_memory_desc_init_by_tag),
but the `memory_desc_t.format_kind` is set
to a canonicalized kind like `blocked`, and the format name is not recorded in
the memory descriptor structure. Initialization with strides will always result
in `blocked` format. The API also uses different types for memory format tags
and kinds to aid correctness.

For more details, refer to the
[Memory descriptor article](https://github.com/oneapi-src/oneDNN/blob/rfc-api-changes-v1.0/doc/rfc/api-v1.0/rfc_memory_desc.md)
of the [RFC for v1.0](https://github.com/oneapi-src/oneDNN/pull/384).

## 5. Changes in the Build System

The build options were slightly changed in the new version of Intel MKL-DNN.
That was done mainly to avoid name collisions with other projects that include
Intel MKL-DNN as a subproject and to accommodate future extensions to the
library. The change are:

| Old option       | New option            | Notes                                                                     |
| :--              | :--                   | :--                                                                       |
| WITH_EXAMPLE     | MKLDNN_BUILD_EXAMPLES |                                                                           |
| WITH_TEST        | MKLDNN_BUILD_TESTS    |                                                                           |
| MKLDNN_THREADING | MKLDNN_CPU_RUNTIME    |                                                                           |
| MKLDNN_USE_MKL   | N/A                   | Intel MKL-DNN does not use Intel MKL anymore                              |
| VTUNEROOT        | N/A                   | Not required, as Intel MKL-DNN contains all the necessary code internally |

By default, the `-Werror` flag is disabled. `MKLDNN_WERROR` controls the
behavior.

For more information about build options, refer to @ref dev_guide_build_options.
