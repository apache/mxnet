Convolution {#dev_guide_convolution}
=====================================

>
> [API Reference](@ref dnnl_api_convolution)
>

## General

The convolution primitive computes forward, backward, or weight update for a
batched convolution operation on 1D, 2D, or 3D spatial data with bias.

The convolution operation is defined by the following formulas. We show formulas
only for 2D spatial data which are straightforward to generalize to cases of
higher and lower dimensions. Variable names follow the standard
@ref dev_guide_conventions.

Let \src, \weights and \dst be \f$N \times IC \times IH \times
IW\f$, \f$OC \times IC \times KH \times KW\f$, and \f$N \times OC \times OH
\times OW\f$ tensors respectively. Let \bias be a 1D tensor with \f$OC\f$
elements.

Furthermore, let the remaining convolution parameters be:

| Parameter                            | Depth      | Height     | Width      | Comment                                                                                                                |
| --:--                                | :--        | :--        | :--        | :--                                                                                                                    |
| Padding: <br>Front, top, and left    | \f$PD_L\f$ | \f$PH_L\f$ | \f$PW_L\f$ | In the API we use `padding_l` to indicate the corresponding vector of paddings (`_l` in the name stands for **left**)  |
| Padding: <br>Back, bottom, and right | \f$PD_R\f$ | \f$PH_R\f$ | \f$PW_R\f$ | In the API we use `padding_r` to indicate the corresponding vector of paddings (`_r` in the name stands for **right**) |
| Stride                               | \f$SD\f$   | \f$SH\f$   | \f$SW\f$   | Convolution without strides is defined by setting the stride parameters to 1                                           |
| Dilation                             | \f$DD\f$   | \f$DH\f$   | \f$DW\f$   | Non-dilated convolution is defined by setting the dilation parameters to 0                                             |

The following formulas show how oneDNN computes convolutions. They are
broken down into several types to simplify the exposition, but in reality the
convolution types can be combined.

To further simplify the formulas, we assume that \f$\src(n, ic, ih, iw) = 0\f$
if \f$ih < 0\f$, or \f$ih \geq IH\f$, or \f$iw < 0\f$, or \f$iw \geq IW\f$.

### Forward

#### Regular Convolution

\f[\dst(n, oc, oh, ow) =  \bias(oc) \\
    + \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
        \src(n, ic, oh \cdot SH + kh - PH_L, ow \cdot SW + kw - PW_L)
        \cdot
        \weights(oc, ic, kh, kw).\f]

Here:

- \f$OH = \left\lfloor{\frac{IH - KH + PH_L + PH_R}{SH}} \right\rfloor + 1,\f$

- \f$OW = \left\lfloor{\frac{IW - KW + PW_L + PW_R}{SW}} \right\rfloor + 1.\f$

#### Convolution with Groups

In the API, oneDNN adds a separate groups dimension to memory objects
representing \weights tensors and represents weights as \f$G \times OC_G \times
IC_G \times KH \times KW \f$ 5D tensors for 2D convolutions with groups.

\f[
    \dst(n, g \cdot OC_G + oc_g, oh, ow) =
        \bias(g \cdot OC_G + oc_g) \\
        +
        \sum_{ic_g=0}^{IC_G-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            \src(n, g \cdot IC_G + ic_g, oh \cdot SH + kh - PH_L,
                    ow \cdot SW + kw - PW_L)
            \cdot
            \weights(g, oc_g, ic_g, kh, kw),
\f]

where
- \f$IC_G = \frac{IC}{G}\f$,
- \f$OC_G = \frac{OC}{G}\f$, and
- \f$oc_g \in [0, OC_G).\f$

The case when \f$OC_G = IC_G = 1\f$ is also known as *a depthwise convolution*.

#### Convolution with Dilation

\f[
    \dst(n, oc, oh, ow) =
        \bias(oc) \\
        +
        \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            \src(n, ic, oh \cdot SH + kh \cdot (DH + 1) - PH_L,
                    ow \cdot SW + kw \cdot (DW + 1) - PW_L)
            \cdot
            \weights(oc, ic, kh, kw).
\f]

Here:

- \f$OH = \left\lfloor{\frac{IH - DKH + PH_L + PH_R}{SH}}
        \right\rfloor + 1,\f$ where \f$DKH = 1 + (KH - 1) \cdot (DH + 1)\f$, and

- \f$OW = \left\lfloor{\frac{IW - DKW + PW_L + PW_R}{SW}}
        \right\rfloor + 1,\f$ where \f$DKW = 1 + (KW - 1) \cdot (DW + 1)\f$.

#### Deconvolution (Transposed Convolution)

Deconvolutions (also called fractionally strided convolutions or transposed
convolutions) work by swapping the forward and backward passes of a
convolution. One way to put it is to note that the weights define a
convolution, but whether it is a direct convolution or a transposed
convolution is determined by how the forward and backward passes are computed.

#### Difference Between Forward Training and Forward Inference

There is no difference between the #dnnl_forward_training
and #dnnl_forward_inference propagation kinds.

### Backward

The backward propagation computes \diffsrc based on \diffdst and
\weights.

The weights update computes \diffweights and \diffbias based on
\diffdst and \src.

@note The *optimized* memory formats \src and \weights might be
different on forward propagation, backward propagation, and weights
update.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index                                                  |
| ---                    | ---                                                                       |
| \src                   | DNNL_ARG_SRC                                                              |
| \weights               | DNNL_ARG_WEIGHTS                                                          |
| \bias                  | DNNL_ARG_BIAS                                                             |
| \dst                   | DNNL_ARG_DST                                                              |
| \diffsrc               | DNNL_ARG_DIFF_SRC                                                         |
| \diffweights           | DNNL_ARG_DIFF_WEIGHTS                                                     |
| \diffbias              | DNNL_ARG_DIFF_BIAS                                                        |
| \diffdst               | DNNL_ARG_DIFF_DST                                                         |
| \f$depthwise\f$        | DNNL_ARG_ATTR_POST_OP_DW                                                  |
| \f$binary post-op\f$   | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1 |

## Implementation Details

### General Notes

N/A.

### Data Types

Convolution primitive supports the following combination of data types for
source, destination, and weights memory objects:

| Propagation        | Source    | Weights   | Destination       | Bias             |
| :--                | :--       | :--       | :--               | :--              |
| forward / backward | f32       | f32       | f32               | f32              |
| forward            | f16       | f16       | f16               | f16              |
| forward            | u8, s8    | s8        | u8, s8, s32, f32  | u8, s8, s32, f32 |
| forward            | bf16      | bf16      | f32, bf16         | f32, bf16        |
| backward           | f32, bf16 | bf16      | bf16              |                  |
| weights update     | bf16      | f32, bf16 | bf16              | f32, bf16        |

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_conv_impl_limits) section below.

### Data Representation

Like other CNN primitives, the convolution primitive expects the following
tensors:

| Spatial | Source / Destination                        | Weights
| :--     | :--                                         | :--
| 1D      | \f$N \times C \times W\f$                   | \f$[G \times ] OC \times IC \times KW\f$
| 2D      | \f$N \times C \times H \times W\f$          | \f$[G \times ] OC \times IC \times KH \times KW\f$
| 3D      | \f$N \times C \times D \times H \times W\f$ | \f$[G \times ] OC \times IC \times KD \times KH \times KW\f$

Physical format of data and weights memory objects is critical for convolution
primitive performance. In the oneDNN programming model, convolution is
one of the few primitives that support the placeholder memory format tag
 #dnnl::memory::format_tag::any (shortened to `any` from now on) and can
define data and weight memory objects format based on the primitive parameters.
When using `any` it is necessary to first create a convolution primitive
descriptor and then query it for the actual data and weight memory objects
formats.

While convolution primitives can be created with memory formats specified
explicitly, the performance is likely to be suboptimal.

The table below shows the combinations for which **plain** memory formats
the convolution primitive is optimized for.

| Spatial    | Convolution Type | Data / Weights logical tensor | Implementation optimized for memory formats
| :--        | :--              | :--                           | :--
| 1D, 2D, 3D |                  | `any`                         | *optimized*
| 1D         | f32, bf16        | NCW / OIW, GOIW               | #dnnl_ncw (#dnnl_abc) / #dnnl_oiw (#dnnl_abc), #dnnl_goiw (#dnnl_abcd)
| 1D         | \"               | \"                            | #dnnl_nwc (#dnnl_acb) / #dnnl_wio (#dnnl_cba), #dnnl_wigo (#dnnl_dcab)
| 1D         | int8             | NCW / OIW                     | #dnnl_nwc (#dnnl_acb) / #dnnl_wio (#dnnl_cba)
| 2D         | f32, bf16        | NCHW / OIHW, GOIHW            | #dnnl_nchw (#dnnl_abcd) / #dnnl_oihw (#dnnl_abcd), #dnnl_goihw (#dnnl_abcde)
| 2D         | \"               | \"                            | #dnnl_nhwc (#dnnl_acdb) / #dnnl_hwio (#dnnl_cdba), #dnnl_hwigo (#dnnl_decab)
| 2D         | int8             | NCHW / OIHW, GOIHW            | #dnnl_nhwc (#dnnl_acdb) / #dnnl_hwio (#dnnl_cdba), #dnnl_hwigo (#dnnl_decab)
| 3D         | f32, bf16        | NCDHW / OIDHW, GOIDHW         | #dnnl_ncdhw (#dnnl_abcde) / #dnnl_oidhw (#dnnl_abcde), #dnnl_goidhw (#dnnl_abcdef)
| 3D         | \"               | \"                            | #dnnl_ndhwc (#dnnl_acdeb) / #dnnl_dhwio (#dnnl_cdeba), #dnnl_dhwigo (#dnnl_defcab)
| 3D         | int8             | NCDHW / OIDHW                 | #dnnl_ndhwc (#dnnl_acdeb) / #dnnl_dhwio (#dnnl_cdeba)

### Post-ops and Attributes

Post-ops and attributes enable you to modify the behavior of the convolution
primitive by applying the output scale to the result of the primitive and by
chaining certain operations after the primitive. The following attributes and
post-ops are supported:

| Propagation | Type      | Operation                                                    | Description                                                                   | Restrictions                        |
| :--         | :--       | :--                                                          | :--                                                                           | :--                                 |
| forward     | attribute | [Output scale](@ref dnnl::primitive_attr::set_output_scales) | Scales the result of convolution by given scale factor(s)                     | int8 convolutions only              |
| forward     | attribute | [Zero points](@ref dnnl::primitive_attr::set_zero_points)    | Sets zero point(s) for the corresponding tensors                              | int8 convolutions only              |
| forward     | post-op   | [Eltwise](@ref dnnl::post_ops::append_eltwise)               | Applies an @ref dnnl_api_eltwise operation to the result                      |                                     |
| forward     | post-op   | [Sum](@ref dnnl::post_ops::append_sum)                       | Adds the operation result to the destination tensor instead of overwriting it |                                     |
| forward     | post-op   | [Binary](@ref dnnl::post_ops::append_binary)                 | Applies a @ref dnnl_api_binary operation to the result                        | General binary post-op restrictions |

To facilitate dynamic quantization, the primitive supports run-time output
scales. That means a user could configure attributes with output scales set to
the #DNNL_RUNTIME_F32_VAL wildcard value instead of the actual scales,
if the scales are not known at the primitive descriptor creation stage.
In this case, the user must provide the scales as an additional input memory
object with argument `DNNL_ARG_ATTR_OUTPUT_SCALES` during the execution stage.

Similarly to run-time output scales, the primitive supports run-time zero
points. The wildcard value for zero points is #DNNL_RUNTIME_S32_VAL. During
the execution stage, the corresponding memory object must be passed as an
argument with its index set to
(`DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_${MEMORY_INDEX}`). Possible
`${MEMORY_INDEX}` values are `DNNL_ARG_SRC` and `DNNL_ARG_DST`.
- For instance, a source tensor zero points memory argument would be passed with
  index (`DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC`).

@note The library does not prevent using post-ops in training, but note that
not all post-ops are feasible for training usage. For instance, using ReLU
with non-zero negative slope parameter as a post-op would not produce an
additional output `workspace` that is required to compute backward propagation
correctly. Hence, in this particular case one should use separate convolution
and eltwise primitives for training.

The library supports any number and order of post operations, but only the
following sequences deploy optimized code:

| Type of convolutions      | Post-ops sequence supported
| :--                       | :--
| f32 and bf16 convolution  | eltwise, sum, sum -> eltwise
| int8 convolution          | eltwise, sum, sum -> eltwise, eltwise -> sum

The attributes and post-ops take effect in the following sequence:
- Source zero point attribute,
- Output scale attribute,
- Post-ops, in the order they were attached,
- Destination zero point attribute.

The operations during attributes and post-ops applying are done in single
precision floating point data type. The conversion to the actual destination
data type happens just before the actual storing.

#### Example 1

Consider the following pseudo-code:

~~~
    primitive_attr attr;
    attr.set_output_scale(/* mask */ 0, alpha);
    attr.set_post_ops({
            { sum={scale=beta} },
            { eltwise={scale=gamma, type=tanh, alpha=ignore, beta=ignored }
        });

    convolution_forward(src, weights, dst, attr)
~~~

The would lead to the following:

\f[
    \dst(\overline{x}) =
        \gamma \cdot \tanh \left(
            \alpha \cdot conv(\src, \weights) +
            \beta  \cdot \dst(\overline{x})
        \right)
\f]

#### Example 2

The following pseudo-code:

~~~
    primitive_attr attr;
    attr.set_output_scale(/* mask */ 0, alpha);
    attr.set_post_ops({
            { eltwise={scale=gamma, type=relu, alpha=eta, beta=ignored }
            { sum={scale=beta} },
        });

    convolution_forward(src, weights, dst, attr)
~~~

That would lead to the following:

\f[
    \dst(\overline{x}) =
        \beta \cdot \dst(\overline{x}) +
        \gamma \cdot ReLU \left(
            \alpha \cdot conv(\src, \weights),
            \eta
        \right)
\f]

#### Example 3

The following pseudo-code:

~~~
    primitive_attr attr;
    attr.set_output_scale(/* mask */ 0, alpha);
    attr.set_zero_point(src, /* mask */ 0, shift_src);
    attr.set_zero_point(dst, /* mask */ 0, shift_dst);
    attr.set_post_ops({
            { eltwise={scale=gamma, type=relu, alpha=eta, beta=ignored }
        });

    convolution_forward(src, weights, dst, attr)
~~~

That would lead to the following:

\f[
    \dst(\overline{x}) =
        \gamma \cdot ReLU \left(
            \alpha \cdot conv(\src - shift_{src}, \weights),
            \eta
        \right) + shift_{dst}
\f]

## Algorithms

oneDNN implements convolution primitives using several different
algorithms:

- _Direct_. The convolution operation is computed directly using SIMD
  instructions. This is the algorithm used for the most shapes and supports
  int8, f32 and bf16 data types.

- _Winograd_. This algorithm reduces computational complexity of convolution
  at the expense of accuracy loss and additional memory operations. The
  implementation is based on the [Fast Algorithms for Convolutional Neural
  Networks by A. Lavin and S. Gray](https://arxiv.org/abs/1509.09308). The
  Winograd algorithm often results in the best performance, but it is
  applicable only to particular shapes. Moreover, Winograd only supports
  int8 and f32 data types.

- _Implicit GEMM_. The convolution operation is reinterpreted in terms of
  matrix-matrix multiplication by rearranging the source data into a
  [scratchpad memory](@ref dev_guide_attributes_scratchpad). This is a fallback
  algorithm that is dispatched automatically when other implementations are
  not available. GEMM convolution supports the int8, f32, and bf16 data types.

#### Direct Algorithm

oneDNN supports the direct convolution algorithm on all supported
platforms for the following conditions:

- Data and weights memory formats are defined by the convolution primitive
  (user passes `any`).

- The number of channels per group is a multiple of SIMD width for grouped
  convolutions.

- For each spatial direction padding does not exceed one half of the
  corresponding dimension of the weights tensor.

- Weights tensor width does not exceed 14.

In case any of these constraints are not met, the implementation will silently
fall back to an explicit GEMM algorithm.

#### Winograd Convolution

oneDNN supports the Winograd convolution algorithm on systems with
Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) support and
Intel Deep Learning Boost (Intel DL Boost)
under the following conditions:

- Data and weights memory formats are defined by the convolution primitive
  (user passes `any` as the data format).

- The spatial domain is two-dimensional.

- The weights shape is 3x3, there are no groups, dilation or strides
  (\f$KH = KW = 3\f$, \f$SH = SW = 1\f$, and \f$DH = DW = 0\f$).

- The data type is either int8 or f32.

In case any of these constraints is not met, the implementation will silently
fall back to the direct algorithm.

The Winograd convolution algorithm implementation additionally chooses tile
size based on the problem shape and
[propagation kind](@ref dnnl_prop_kind_t):

- For `forward_inference` oneDNN supports
  \f$F(2 \times 2, 3 \times 3)\f$ or
  \f$F(4 \times 4, 3 \times 3)\f$

- oneDNN supports only \f$F(4 \times 4, 3 \times 3)\f$ Winograd for all
  the training propagation kinds.

The following side effects should be weighed against the (potential)
performance boost achieved from using the Winograd algorithm:

- _Memory consumption_. Winograd implementation in oneDNN requires additional
  scratchpad memory to store intermediate results. As more convolutions using
  Winograd are added to the topology, the amount of memory required can grow
  significantly. This growth can be controlled if the scratchpad memory can be
  reused across multiple primitives. See @ref dev_guide_attributes_scratchpad
  for more details.

- _Accuracy_. In some cases Winograd convolution produce results that are
  significantly less accurate than results from the direct convolution.

Create a Winograd convolution by simply creating a convolution descriptor
(step 6 in [simple network example](@ref cnn_inference_f32_cpp) specifying
the Winograd algorithm. The rest of the steps are exactly the same.

~~~cpp
auto conv1_desc = convolution_forward::desc(
    prop_kind::forward_inference, algorithm::convolution_winograd,
    conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
    conv1_strides, conv1_padding_l, conv1_padding_r);
~~~

#### Automatic Algorithm Selection

oneDNN supports `dnnl::algorithm::convolution_auto` algorithm that
instructs the library to automatically select the *best* algorithm based on
the heuristics that take into account tensor shapes and the number of logical
processors available.  (For automatic selection to work as intended, use the
same thread affinity settings when creating the convolution as when executing
the convolution.)

@anchor dg_conv_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **CPU**
   - Winograd are implemented only for processors with Intel AVX-512 and
     Intel DL Boost instruction sets
   - Run-time output scales are not supported

3. **GPU**
    - No support for Winograd algorithm

## Performance Tips

- Use #dnnl::memory::format_tag::any for source, weights, and destinations
  memory format tags when create a convolution primitive to allow the library
  to choose the most appropriate memory format.

## Examples

| Engine  | Name                         | Comments
| :--     | :--                          | :--
| CPU/GPU | @ref convolution_example_cpp | @copydetails convolution_example_cpp_short
