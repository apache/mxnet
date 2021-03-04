Eltwise {#dev_guide_eltwise}
============================

>
> [API Reference](@ref dnnl_api_eltwise)
>

## General

### Forward

The eltwise primitive applies an operation to every element of the tensor (the
variable names follow the standard @ref dev_guide_conventions):

\f[
    \dst(\overline{s}) = Operation(\src(\overline{s})),
\f]

where \f$\overline{s} = (s_n, .., s_0)\f$.

The following operations are supported:

| Operation    | oneDNN algorithm kind                                              | Forward formula                                                                                                                                             | Backward formula (from src)                                                                                                        | Backward formula (from dst)                                                                                            |
| :--          | :--                                                                | :--                                                                                                                                                         | :--                                                                                                                                | :--                                                                                                                    |
| abs          | #dnnl_eltwise_abs                                                  | \f$ d = \begin{cases} s & \text{if}\ s > 0 \\ -s & \text{if}\ s \leq 0 \end{cases} \f$                                                                      | \f$ ds = \begin{cases} dd & \text{if}\ s > 0 \\ -dd & \text{if}\ s < 0 \\ 0 & \text{if}\ s = 0 \end{cases} \f$                     | --                                                                                                                     |
| bounded_relu | #dnnl_eltwise_bounded_relu                                         | \f$ d = \begin{cases} \alpha & \text{if}\ s > \alpha \geq 0 \\ s & \text{if}\ 0 < s \leq \alpha \\ 0 & \text{if}\ s \leq 0 \end{cases} \f$                  | \f$ ds = \begin{cases} dd & \text{if}\ 0 < s \leq \alpha, \\ 0 & \text{otherwise}\ \end{cases} \f$                                 | --                                                                                                                     |
| clip         | #dnnl_eltwise_clip                                                 | \f$ d = \begin{cases} \beta & \text{if}\ s > \beta \geq \alpha \\ s & \text{if}\ \alpha < s \leq \beta \\ \alpha & \text{if}\ s \leq \alpha \end{cases} \f$ | \f$ ds = \begin{cases} dd & \text{if}\ \alpha < s \leq \beta \\ 0 & \text{otherwise}\ \end{cases} \f$                              | --                                                                                                                     |
| elu          | #dnnl_eltwise_elu <br> #dnnl_eltwise_elu_use_dst_for_bwd           | \f$ d = \begin{cases} s & \text{if}\ s > 0 \\ \alpha (e^s - 1) & \text{if}\ s \leq 0 \end{cases} \f$                                                        | \f$ ds = \begin{cases} dd & \text{if}\ s > 0 \\ dd \cdot \alpha e^s & \text{if}\ s \leq 0 \end{cases} \f$                          | \f$ ds = \begin{cases} dd & \text{if}\ d > 0 \\ dd \cdot (d + \alpha) & \text{if}\ d \leq 0 \end{cases}. See\ (2). \f$ |
| exp          | #dnnl_eltwise_exp <br> #dnnl_eltwise_exp_use_dst_for_bwd           | \f$ d = e^s \f$                                                                                                                                             | \f$ ds = dd \cdot e^s \f$                                                                                                          | \f$ ds = dd \cdot d \f$                                                                                                |
| gelu_erf     | #dnnl_eltwise_gelu_erf                                             | \f$ d = 0.5 s (1 + \mathop{erf}[\frac{s}{\sqrt{2}}])\f$                                                                                                     | \f$ ds = dd \cdot \left(0.5 + 0.5 \, \mathop{erf}\left({\frac{s}{\sqrt{2}}}\right) + \frac{s}{\sqrt{2\pi}}e^{-0.5s^{2}}\right) \f$ | --                                                                                                                     |
| gelu_tanh    | #dnnl_eltwise_gelu_tanh                                            | \f$ d = 0.5 s (1 + \tanh[\sqrt{\frac{2}{\pi}} (s + 0.044715 s^3)])\f$                                                                                       | \f$ See\ (1). \f$                                                                                                                  | --                                                                                                                     |
| linear       | #dnnl_eltwise_linear                                               | \f$ d = \alpha s + \beta \f$                                                                                                                                | \f$ ds = \alpha \cdot dd \f$                                                                                                       | --                                                                                                                     |
| log          | #dnnl_eltwise_log                                                  | \f$ d = \log_{e}{s} \f$                                                                                                                                     | \f$ ds = \frac{dd}{s} \f$                                                                                                          | --                                                                                                                     |
| logistic     | #dnnl_eltwise_logistic <br> #dnnl_eltwise_logistic_use_dst_for_bwd | \f$ d = \frac{1}{1+e^{-s}} \f$                                                                                                                              | \f$ ds = \frac{dd}{1+e^{-s}} \cdot (1 - \frac{1}{1+e^{-s}}) \f$                                                                    | \f$ ds = dd \cdot d \cdot (1 - d) \f$                                                                                  |
| pow          | #dnnl_eltwise_pow                                                  | \f$ d = \alpha s^{\beta} \f$                                                                                                                                | \f$ ds = dd \cdot \alpha \beta s^{\beta - 1} \f$                                                                                   | --                                                                                                                     |
| relu         | #dnnl_eltwise_relu <br> #dnnl_eltwise_relu_use_dst_for_bwd         | \f$ d = \begin{cases} s & \text{if}\ s > 0 \\ \alpha s & \text{if}\ s \leq 0 \end{cases} \f$                                                                | \f$ ds = \begin{cases} dd & \text{if}\ s > 0 \\ \alpha \cdot dd & \text{if}\ s \leq 0 \end{cases} \f$                              | \f$ ds = \begin{cases} dd & \text{if}\ d > 0 \\ \alpha \cdot dd & \text{if}\ d \leq 0 \end{cases}. See\ (2). \f$       |
| round        | #dnnl_eltwise_round                                                | \f$ d = round(s) \f$                                                                                                                                        | --                                                                                                                                 | --                                                                                                                     |
| soft_relu    | #dnnl_eltwise_soft_relu                                            | \f$ d = \log_{e}(1+e^s) \f$                                                                                                                                 | \f$ ds = \frac{dd}{1 + e^{-s}} \f$                                                                                                 | --                                                                                                                     |
| sqrt         | #dnnl_eltwise_sqrt <br> #dnnl_eltwise_sqrt_use_dst_for_bwd         | \f$ d = \sqrt{s} \f$                                                                                                                                        | \f$ ds = \frac{dd}{2\sqrt{s}} \f$                                                                                                  | \f$ ds = \frac{dd}{2d} \f$                                                                                             |
| square       | #dnnl_eltwise_square                                               | \f$ d = s^2 \f$                                                                                                                                             | \f$ ds = dd \cdot 2 s \f$                                                                                                          | --                                                                                                                     |
| swish        | #dnnl_eltwise_swish                                                | \f$ d = \frac{s}{1+e^{-\alpha s}} \f$                                                                                                                       | \f$ ds = \frac{dd}{1 + e^{-\alpha s}}(1 + \alpha s (1 - \frac{1}{1 + e^{-\alpha s}})) \f$                                          | --                                                                                                                     |
| tanh         | #dnnl_eltwise_tanh <br> #dnnl_eltwise_tanh_use_dst_for_bwd         | \f$ d = \tanh{s} \f$                                                                                                                                        | \f$ ds = dd \cdot (1 - \tanh^2{s}) \f$                                                                                             | \f$ ds = dd \cdot (1 - d^2) \f$                                                                                        |

\f$ (1)\ ds = dd \cdot 0.5 (1 + tanh[\sqrt{\frac{2}{\pi}} (s + 0.044715 s^3)]) \cdot (1 + \sqrt{\frac{2}{\pi}} (s + 0.134145 s^3) \cdot (1 -  tanh[\sqrt{\frac{2}{\pi}} (s + 0.044715 s^3)]) ) \f$

\f$ (2)\ \text{Operation is supported only for } \alpha \geq 0. \f$

#### Difference Between Forward Training and Forward Inference

There is no difference between the #dnnl_forward_training
and #dnnl_forward_inference propagation kinds.

### Backward

The backward propagation computes \f$\diffsrc(\overline{s})\f$, based on
\f$\diffdst(\overline{s})\f$ and \f$\src(\overline{s})\f$. However, some
operations support a computation using \f$\dst(\overline{s})\f$ memory produced
during forward propagation. Refer to the table above for a list of operations
supporting destination as input memory and the corresponding formulas.

#### Exceptions
The eltwise primitive with algorithm round does not support backward
propagation.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index                                                  |
| ---                    | ---                                                                       |
| \src                   | DNNL_ARG_SRC                                                              |
| \dst                   | DNNL_ARG_DST                                                              |
| \diffsrc               | DNNL_ARG_DIFF_SRC                                                         |
| \diffdst               | DNNL_ARG_DIFF_DST                                                         |
| \f$binary post-op\f$   | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1 |

## Implementation Details

### General Notes

1. All eltwise primitives have a common initialization function (e.g.,
   dnnl::eltwise_forward::desc::desc()) which takes both parameters
   \f$\alpha\f$, and \f$\beta\f$. These parameters are ignored if they are
   unused.

2. The memory format and data type for \src and \dst are assumed to be the
   same, and in the API are typically referred as `data` (e.g., see `data_desc`
   in dnnl::eltwise_forward::desc::desc()). The same holds for
   \diffsrc and \diffdst. The corresponding memory descriptors are referred
   to as `diff_data_desc`.

3. Both forward and backward propagation support in-place operations, meaning
   that \src can be used as input and output for forward propagation, and
   \diffdst can be used as input and output for backward propagation. In case of
   an in-place operation, the original data will be overwritten. Note, however,
   that some algorithms for backward propagation require original \src, hence
   the corresponding forward propagation should not be performed in-place for
   those algorithms. Algorithms that use \dst for backward propagation can be
   safely done in-place.

4. For some operations it might be beneficial to compute backward
   propagation based on \f$\dst(\overline{s})\f$, rather than on
   \f$\src(\overline{s})\f$, for improved performance.

@note For operations supporting destination memory as input, \dst can be
used instead of \src when backward propagation is computed. This enables
several performance optimizations (see the tips below).

### Data Type Support

The eltwise primitive supports the following combinations of data types:

| Propagation        | Source / Destination | Intermediate data type
| :--                | :--                  | :--
| forward / backward | f32, bf16            | f32
| forward            | f16                  | f16
| forward            | s32 / s8 / u8        | f32

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_eltwise_impl_limits) section
    below.

Here the intermediate data type means that the values coming in are first
converted to the intermediate data type, then the operation is applied, and
finally the result is converted to the output data type.

### Data Representation

The eltwise primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions.

### Post-ops and Attributes

| Propagation | Type    | Operation                                    | Description                                            | Restrictions                        |
| :--         | :--     | :--                                          | :--                                                    | :--                                 |
| Forward     | Post-op | [Binary](@ref dnnl::post_ops::append_binary) | Applies a @ref dnnl_api_binary operation to the result | General binary post-op restrictions |

@anchor dg_eltwise_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for
   limitations related to data types support.

## Performance Tips

1. For backward propagation, use the same memory format for \src, \diffdst,
   and \diffsrc (the format of the \diffdst and \diffsrc are always the
   same because of the API). Different formats are functionally supported but
   lead to highly suboptimal performance.

2. Use in-place operations whenever possible (see caveats in General Notes).

3. As mentioned above for all operations supporting destination memory as input,
   one can use the \dst tensor instead of \src. This enables the
   following potential optimizations for training:

    - Such operations can be safely done in-place.

    - Moreover, such operations can be fused as a
      [post-op](@ref dev_guide_attributes) with the previous operation if that
      operation does not require its \dst to compute the backward
      propagation (e.g., if the convolution operation satisfies these
      conditions).

## Examples

| Engine  | Name                     | Comments
| :--     | :--                      | :--
| CPU/GPU | @ref eltwise_example_cpp | @copydetails eltwise_example_cpp_short
