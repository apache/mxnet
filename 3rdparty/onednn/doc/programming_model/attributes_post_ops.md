Primitive Attributes: Post-ops {#dev_guide_attributes_post_ops}
===============================================================

oneDNN implements some basic capabilities of operation fusion using the
**post-ops attributes** API. The operation fusion typically reduces the memory
bandwidth pressure hence leading to higher performance.

*Post-ops* are operations that are appended after a primitive. They are
implemented using the @ref dev_guide_attributes mechanism. If there are
multiple post-ops, they are executed in the order they have been appended.

Currently the following post-ops are supported by the library:
* [Eltwise](@ref dev_guide_attributes_post_ops_eltwise)
* [Sum](@ref dev_guide_attributes_post_ops_sum)
* [Depthwise](@ref dev_guide_attributes_post_ops_depthwise)
* [Binary](@ref dev_guide_attributes_post_ops_binary)

Just like @ref dev_guide_attributes, the post-ops are represented by an opaque
structure (@ref dnnl_post_ops_t in C API and @ref dnnl::post_ops in C++ API)
which is copied once it is attached to the attributes using the C++ @ref
dnnl::primitive_attr::set_post_ops or C @ref dnnl_primitive_attr_set_post_ops
functions. The attributes then must be passed to a primitive descriptor
creation function to take effect. Below is a simple skeleton for the C++ API:

~~~cpp
dnnl::post_ops po; // default empty post-ops
assert(po.len() == 0); // no post-ops attached

po.append_SOMETHING(params); // append some particular post-op
po.append_SOMETHING_ELSE(other_params); // append one more post-op

// (!) Note that the order in which post-ops are appended matters!
assert(po.len() == 2);

dnnl::primitive_attr attr; // default attributes
attr.set_post_ops(po); // attach the post-ops to the attr

// further po changes would not affect attr

primitive::primitive_desc op_pd(params, attr); // create a pd with the attr
~~~

@note
    Different post-ops can be chained together by appending one
    after another. Note that the appending order matters: the sequence of
    the post operations is executed in the order of appearance. The maximum
    number of post operations supported in the library is 32.

@warning
    Different primitives may have different post-ops support. Each primitive
    documentation page contains information about what kind of post operations
    it supports. Moreover, the support might also depend on the actual
    implementation of a primitive. For instance, the library may not support
    post-ops for primitive reference implementations (which are typically very
    slow, so there is no point in doing the actual fusion). Robust code should
    handle errors accordingly. See the
    [section on attributes error handling](@ref dev_guide_attributes_error_handling).

@note
    Post-ops do not change the memory format of the operation destination memory
    object.

The post-op object can be inspected using the @ref dnnl::post_ops::kind()
function that takes an index of the post-op (which must be less than the value
returned by @ref dnnl::post_ops::len()), and returns its kind.

## Supported Post-ops

@anchor dev_guide_attributes_post_ops_eltwise
### Eltwise Post-op

The eltwise post-op enables fusing a primitive with an @ref dev_guide_eltwise
primitive. This is probably one of the most popular kinds of fusion:
an eltwise (typically an activation function) with preceding convolution
or inner product.

The @ref dnnl::primitive::kind of this post-op
is #dnnl::primitive::kind::eltwise.

API:
- C: @ref dnnl_post_ops_append_eltwise
- C++: @ref dnnl::post_ops::append_eltwise

The parameters (C++ API for simplicity):
~~~cpp
void dnnl::post_ops::append_eltwise(
        float scale, // scaling factor (described below)
        algorithm alg, float alpha, float beta // same as in eltwise primitive
        );
~~~

The `alg`, `alpha`, and `beta` parameters are the same as in @ref dev_guide_eltwise.

The eltwise post-op replaces:
\f[
    \dst[:] = \operatorname{Op}(...)
\f]

with

\f[
    \dst[:] = scale \cdot \operatorname{eltwise}( \operatorname{Op}(...) )
\f]

The intermediate result of \f$\operatorname{Op}(...)\f$ is not preserved.
Hence, in most cases this kind of fusion cannot be used during training.

The \f$scale\f$ factor is supported in
[INT8](@ref dev_guide_attributes_quantization) inference only. For all other
cases the scale must be `1.0`.

@anchor dev_guide_attributes_post_ops_sum
### Sum Post-op

The sum post-op accumulates the result of a primitive with the existing data.
Prior to accumulating the result, the existing value would be multiplied by
scale.

The kind of this post-op is #dnnl::primitive::kind::sum.

This feature might improve performance for cases like residual learning
blocks, where the result of a convolution is accumulated to the previously
computed activations. The scale parameter can be used in [INT8](@ref
dev_guide_attributes_quantization) inference only when the result and previous
activations have different magnitudes. For all other cases the scale must be
`1.0`.

The sum post-op replaces
\f[
    \dst[:] = \operatorname{Op}(...)
\f]

with

\f[
    \dst[:] = scale \cdot \dst[:] + \operatorname{Op}(...)
\f]


If the data type parameter is specified, the original destination tensor will be
reinterpreted as a tensor with the provided data type. Because it is a
reinterpretation, data_type and the destination data type must have the same size. As a
result, the computation will be:

\f[
    \dst(:) = scale \cdot \operatorname{as_data_type}(\dst(:)) + \operatorname{Op}(...)
\f]

@note
* Currently only a u8/s8 data type parameter is supported.
**CPU**
    - No support for different destination and sum data type.

@anchor dev_guide_attributes_post_ops_depthwise
### Depthwise Post-op

Appends a Depthwise convolution as a post-op. This post-op can only be fused
with 1x1 convolution as generally seen in models (like MobileNet_v1) that use a
stack of Separable convolutions: Depthwise convolution followed by 1x1
convolution. The stack of these Separable convolutions (like in MobileNet_v1)
provide an opportunity to fuse 1x1-Convolution with bandwidth-limited Depthwise
convolution.

The @ref dnnl::primitive::kind of this post-op
is #dnnl::primitive::kind::convolution.

There are two variants of this post-op: dw_k3s1p1 and dw_k3_s2p1 for stride-1
and and stride-2 respectively.

API:
- C: @ref dnnl_post_ops_append_dw_k3s1p1 , @ref dnnl_post_ops_append_dw_k3s2p1
- C++: @ref dnnl::post_ops::append_dw_k3s1p1 , @ref dnnl::post_ops::append_dw_k3s2p1

For better readability, below we assume a 2D convolution and use the following
notations:

  `conv_1x1` Convolution with weights spatial=1 i.e., `kh` = `kw` = 1.

  `conv_dw` Depthwise convolution with weights spatial=3 i.e., `kh` = `kw` = 3,
  `g` = `oc` = `ic` and `pad_l` = `pad_r` = {1, 1}.

The Depthwise post-op replaces

\f[
    dst[:] = Conv_{1x1}(...)
\f]

with

\f[
    dst[:] = Conv_{dw}(Conv_{1x1}(...))
\f]


The final output dimensions of the after post-op is defined as

\f[
    dst_{conv_dw} = \{ n, oc_{1x1}, \operatorname{ceil}(oh_{conv_{1x1}}/stride),
     \operatorname{ceil}(ow_{conv_{1x1}}/stride) \}
\f]

where `oh_conv_1x1`, `ow_conv_1x1` are height and width of conv_1x1 destination.

![Fusion](images/img_depthwise_fusion.jpg)

Supported data types

| conv 1x1 output data type        | depthwise post-op output data type | depthwise post-op weights data type | depthwise post-op bias data type
| :--                              | :--                                | :--                                 | :--
| u8, s8                           | u8, s8, s32, f32                   | s8                                  | f32, s32
| f32                              | f32                                | f32                                 | f32
| bf16                             | bf16, f32                          | bf16                                | f32, bf16

@note
  * Currently only supported for 2D 1x1 convolution.

  * Only eltwise post-op can be part of post-op chain (i.e., sum post-op is
    not supported)

  * The `dst_1x1`, `wei_dw` and `dst_dw` are assumed to be #dnnl_format_tag_any.

@anchor dev_guide_attributes_post_ops_binary
### Binary Post-op

The binary post-op enables fusing a primitive with a @ref dev_guide_binary
primitive.

The @ref dnnl::primitive::kind of this post-op is
#dnnl::primitive::kind::binary.

API:
- C: @ref dnnl_post_ops_append_binary
- C++: @ref dnnl::post_ops::append_binary

The parameters (C++ API for simplicity):
~~~cpp
void dnnl::post_ops::append_binary(
        algorithm alg, // binary algorithm to apply
        const memory::desc &src1 // memory descriptor for a second memory operand
        );
~~~

The `alg` and `src1` parameters are the same as in @ref dev_guide_binary.

The binary post-op replaces:
\f[
    \dst[:] = \operatorname{Op}(...)
\f]

with

\f[
    \dst[:] = \operatorname{binary}(\operatorname{Op}(...), Source\_1[:])
\f]

The intermediate result of \f$\operatorname{Op}(...)\f$ is not preserved.
Hence, in most cases this kind of fusion cannot be used during training.

Currently the following scenarios are supported:
* Per tensor broadcast, when \f$Source\_1\f$ is represented as a one-element
  tensor, i.e. {1, 1, 1, 1} for 2D spatial \f$\operatorname{Op}(...)\f$.
* Per channels (i.e. dimension 1) broadcast, when a `dim[1]` value of
  \f$Source\_1\f$ coincides with a `dim[1]` value of
  \f$\operatorname{Op}(...)\f$, i.e. {1, C, 1, 1} for 2D spatial
  \f$\operatorname{Op}(...)\f$.

Scenario when \f$Source\_1\f$ represents a full tensor as
\f$\operatorname{Op}(...)\f$ is not supported yet.

## Examples of Chained Post-ops

Different post-ops can be chained together by appending one after another.
Note that the order matters: the post-ops are executed in the order they have
been appended.

Let's consider some examples.

### Sum -> ReLU

This pattern is pretty common for the CNN topologies of the ResNet family.

~~~cpp
dnnl::post_ops po;
po.append_sum(
        /* scale = */ 1.f);
po.append_eltwise(
        /* scale     = */ 1.f,
        /* alg kind  = */ dnnl::algorithm::eltwise_relu,
        /* neg slope = */ 0.f,
        /* unused for relu */ 0.f);

dnnl::primitive_attr attr;
attr.set_post_ops(po);

convolution_forward::primitive_desc(conv_d, attr, engine);
~~~

This will lead to the following primitive behavior:

\f[
    \dst[:] = \operatorname{ReLU}(\dst[:] + \operatorname{conv}(\src[:], \weights[:])
\f]


@anchor dev_guide_attributes_post_ops_with_scales
### Tanh -> Sum -> ScaleShift

This is a hypothetical example that illustrates the sequence of operations
applied.  We also set all the scales to values other than 1.0 and use @ref
dnnl::primitive_attr::set_output_scales which will be covered in @ref
dev_guide_attributes_quantization.

~~~cpp
dnnl::post_ops po;
po.append_eltwise(
        /* scale     = */ s_tanh,
        /* alg kind  = */ dnnl::algorithm::eltwise_tanh,
        /* unused for tanh */ 0.f,
        /* unused for tanh */ 0.f);
po.append_sum(
        /* scale     = */ s_sum);
po.append_eltwise(
        /* scale     = */ s_linear,
        /* alg kind  = */ dnnl::algorithm::eltwise_linear,
        /* scale     = */ alpha,
        /* shift     = */ beta);

dnnl::primitive_attr attr;
attr.set_output_scales(0, {s_conv});
attr.set_post_ops(po);

convolution_forward::primitive_desc(conv_d, attr, engine);
~~~

This will lead to the following primitive behavior (for better readability
the tensors are designated by their names only; i.e., `[:]` is omitted):

\f[
    \dst
        =
        s_{linear} \cdot
        (
            \alpha \cdot
            (
                s_{sum} \cdot \dst
                +
                s_{tanh} \cdot \tanh
                (
                    s_{conv} \cdot \operatorname{conv}(\src, \weights)
                )
            )
            + \beta
        )
\f]


@anchor dev_guide_attributes_post_ops_depthwise_fusion
### Relu -> Depthwise -> Relu

An example of fusing depthwise convolution with 1x1 convolution in MobileNet.

~~~cpp
dnnl::post_ops po;

po.append_eltwise(
        /* scale     = */ 1.f,
        /* alg kind  = */ dnnl::algorithm::eltwise_relu,
        /* neg slope = */ 0.f,
        /* unused for relu */ 0.f);

po.append_dw_k3s1p1( /* or po.append_dw_k3s2p1 for depthwise with stride=2*/
        /* depthwise weights data type = */ dnnl::memory::data_type::s8,
        /* depthwise bias data type (undef implies no bias) = */ dnnl::memory::data_type::undef,
        /* depthwise destination data type = */ dnnl::memory::data_type::u8,
        /* mask for output scales of depthwise output = */ mask,
        /* output scales for depthwise output = */ scales_depthwise)

po.append_eltwise(
        /* scale     = */ 1.f,
        /* alg kind  = */ dnnl::algorithm::eltwise_relu,
        /* neg slope = */ 0.f,
        /* unused for relu */ 0.f);

dnnl::primitive_attr attr;
attr.set_output_scales(0, {output_scales_1x1_conv});
attr.set_post_ops(po);

auto cpd = convolution_forward::primitive_desc(conv_1x1, attr, engine);
auto dw_weight_md = cpd.query(query::exec_arg_md,
                DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
auto dw_bias_md = cpd.query(query::exec_arg_md,
                DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

~~~

This will lead to the following primitive behaviour:

\f[
    dst
        =
        ReLU_{depthwise}
        (
            scales_{depthwise} \cdot
            (
                conv_{depthwise}
                (
                    ReLU_{1x1}
                    (
                        scales_{conv_{1x1}} \cdot
                        (
                            conv_{1x1}()
                        )
                    )
                )
            )
        )
\f]

@anchor dev_guide_attributes_post_ops_binary_fusion
### Binary

An example of fusing convolution with binary post-op with per channel addition.

~~~cpp
dnnl::memory::desc conv_dst_md {MB, C, H, W}; /* 2D conv destination memory desc */

dnnl::post_ops po;

/* Append eltwise post-op prior the binary post-op */
po.append_eltwise(
        /* scale     = */ 1.f,
        /* alg kind  = */ dnnl::algorithm::eltwise_relu,
        /* neg slope = */ 0.f,
        /* unused for relu */ 0.f);

/* Note that `C` coincides with the one from `conv_dst_md`. Also note that only
 * supported memory format for src1 memory is `nchw` (or `abcd`) format. */
po.append_binary(
        /* alg kind = */ dnnl::algorithm::binary_add,
        /* src1_md = */ dnnl::memory::desc(
                {1, C, 1, 1},
                dnnl::memory::data_type::f32,
                dnnl::memory::format_tag::abcd));

dnnl::primitive_attr attr;
attr.set_post_ops(po);

auto cpd = convolution_forward::primitive_desc(conv, attr, engine);

/* To set memory argument for binary post-op, the following should take place: */
std::unordered_map<int, memory> args;

args.insert(DNNL_ARG_SRC, conv_src_memory);
...
int binary_post_op_position = 1; /* hard coded here, but may be queried */
args.insert(
        DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) | DNNL_ARG_SRC_1, /* note parentheses around index */
        binary_post_op_src1_memory);
~~~
