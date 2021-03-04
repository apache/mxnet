Layer Normalization {#dev_guide_layer_normalization}
====================================================

>
> [API Reference](@ref dnnl_api_layer_normalization)
>

## General

The layer normalization primitive performs a forward or backward layer
normalization operation on a 2-5D data tensor.

### Forward

The layer normalization operation performs normalization over the last logical
axis of the data tensor and is defined by the following formulas. We show
formulas only for 3D data, which are straightforward to generalize to
cases of higher dimensions. Variable names follow the standard
@ref dev_guide_conventions.

\f[
    \dst(t, n, c) =
       \gamma(c) \cdot
       \frac{\src(t, n, c) - \mu(t, n)} {\sqrt{\sigma^2(t, n) + \varepsilon}}
       + \beta(c),
\f]

where

- \f$\gamma(c), \beta(c)\f$ are optional scale and shift for a channel
(see #dnnl_use_scaleshift flag),

- \f$\mu(t, n), \sigma^2(t, n)\f$ are mean and variance (see
  #dnnl_use_global_stats flag), and

- \f$\varepsilon\f$ is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and variance are computed at runtime, the following formulas are used:

- \f$\mu(t, n) = \frac{1}{C} \sum\limits_{c} \src(t, n, c)_{}\f$,

- \f$\sigma^2(t, n) = \frac{1}{C} \sum\limits_{c} {}_{} (\src(t, n, c) - \mu(t, n))^2\f$.

The \f$\gamma(c)\f$ and \f$\beta(c)\f$ tensors are considered learnable.

#### Difference Between Forward Training and Forward Inference

 * If mean and variance are computed at runtime (i.e., #dnnl_use_global_stats
   is not set), they become outputs for the propagation kind
   #dnnl_forward_training (because they would be required during the backward
   propagation). Data layout for mean and variance must be specified during
   initialization of the layer normalization descriptor by passing the memory
   descriptor for statistics (e.g., by passing stat_desc in
   dnnl::layer_normalization_forward::desc::desc()). Mean and variance are
   not exposed for the propagation kind #dnnl_forward_inference.

### Backward

The backward propagation computes
\f$\diffsrc(t, n, c)\f$,
\f$\diffgamma(c)^*\f$, and \f$\diffbeta(c)^*\f$
based on
\f$\diffdst(t, n, c)\f$, \f$src(t, n, c)\f$, \f$\mu(t, n)\f$,
\f$\sigma^2(t, n)\f$, \f$\gamma(c) ^*\f$, and \f$\beta(c) ^*\f$.

The tensors marked with an asterisk are used only when the primitive is
configured to use \f$\gamma(c)\f$, and \f$\beta(c)\f$
(i.e., #dnnl_use_scaleshift is set).

## Execution Arguments

Depending on the [flags](@ref dnnl_normalization_flags_t) and
[propagation kind](@ref dnnl_prop_kind_t), the layer normalization primitive
requires different inputs and outputs. For clarity, a summary is shown below.

|                                                | #dnnl_forward_inference                                                                       | #dnnl_forward_training                                                                        | #dnnl_backward                                                                                                                     | #dnnl_backward_data        |
| :--                                            | :--                                                                                           | :--                                                                                           | :--                                                                                                                                | :--                        |
| #dnnl_normalization_flags_none                 | *Inputs*: \src <br><br> *Outputs*: \dst                                                       | *Inputs*: \src <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$                            | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \diffsrc                                                   | Same as for #dnnl_backward |
| #dnnl_use_global_stats                         | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \dst                            | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \dst                            | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \diffsrc                                                   | Same as for #dnnl_backward |
| #dnnl_use_scaleshift                           | *Inputs*: \src, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst                            | *Inputs*: \src, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$ | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \diffsrc, \diffgamma, \diffbeta | Not supported              |
| #dnnl_use_global_stats \| #dnnl_use_scaleshift | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \diffsrc, \diffgamma, \diffbeta | Not supported              |

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output  | Execution argument index  |
| ---                     | ---                       |
| \src                    | DNNL_ARG_SRC              |
| \f$\gamma, \beta\f$     | DNNL_ARG_SCALE_SHIFT      |
| mean (\f$\mu\f$)        | DNNL_ARG_MEAN             |
| variance (\f$\sigma\f$) | DNNL_ARG_VARIANCE         |
| \dst                    | DNNL_ARG_DST              |
| \diffdst                | DNNL_ARG_DIFF_DST         |
| \diffsrc                | DNNL_ARG_DIFF_SRC         |
| \diffgamma, \diffbeta   | DNNL_ARG_DIFF_SCALE_SHIFT |


## Implementation Details

### General Notes

1. The different flavors of the primitive are partially controlled by the @p
   flags parameter that is passed to the operation descriptor initialization
   function (e.g., dnnl::layer_normalization_forward::desc::desc()). Multiple
   flags can be set using the bitwise OR operator (`|`).

2. For forward propagation, the mean and variance might be either computed at
   runtime (in which case they are outputs of the primitive) or provided by
   a user (in which case they are inputs). In the latter case, a user must set
   the #dnnl_use_global_stats flag. For the backward propagation, the mean and
   variance are always input parameters.

3. The memory format and data type for `src` and `dst` are assumed to be the
   same, and in the API they are typically referred to as `data` (e.g., see
   `data_desc` in dnnl::layer_normalization_forward::desc::desc()). The same is
   true for `diff_src` and `diff_dst`. The corresponding memory descriptors are
   referred to as `diff_data_desc`.

4. Both forward and backward propagation support in-place operations, meaning
   that \src can be used as input and output for forward propagation, and
   \diffdst can be used as input and output for backward propagation. In case of
   an in-place operation, the original data will be overwritten. Note, however,
   that backward propagation requires original \src, hence the corresponding
   forward propagation should not be performed in-place.

### Data Type Support

The operation supports the following combinations of data types:

| Propagation        | Source / Destination | Mean / Variance / ScaleShift
| :--                | :--                  | :--
| forward / backward | f32                  | f32
| forward            | f16                  | f32

### Data Representation

#### Mean and Variance

The mean (\f$\mu\f$) and variance (\f$\sigma^2\f$) are separate tensors with
number of dimensions equal to (\f$data\_ndims - 1\f$) and size
\f$(data\_dim[0], data\_dim[1], ..., data\_dim[ndims - 2])\f$.

The corresponding memory object can have an arbitrary memory format. Unless mean
and variance are computed at runtime and not exposed (i.e., propagation kind is
#dnnl_forward_inference and #dnnl_use_global_stats is not set), the user should
provide a memory descriptor for statistics when initializing the layer
normalization descriptor. For best performance, it is advised to use the memory
format that follows the data memory format; i.e., if the data format is
#dnnl_tnc, the best performance can be expected for statistics with the
#dnnl_tn format and suboptimal for statistics with the #dnnl_nt format.

#### Scale and Shift

If used, the scale (\f$\gamma\f$) and shift (\f$\beta\f$) are
combined in a single 2D tensor of shape \f$2 \times C\f$.

The format of the corresponding memory object must be #dnnl_nc (#dnnl_ab).

#### Source, Destination, and Their Gradients

The layer normalization primitive works with an arbitrary data tensor; however,
it was designed for RNN data tensors (i.e., #dnnl_nc, #dnnl_tnc, #dnnl_ldnc).
Unlike CNN data tensors, RNN data tensors have a single feature dimension.
Layer normalization performs normalization over the last logical dimension
(feature dimension for RNN tensors) across non-feature dimensions.

The layer normalization primitive is optimized for the following memory formats:

| Logical tensor | Implementations optimized for memory formats
| :--            | :--
| NC             | #dnnl_nc (#dnnl_ab)
| TNC            | #dnnl_tnc (#dnnl_abc), #dnnl_ntc (#dnnl_bac)
| LDNC           | #dnnl_ldnc (#dnnl_abcd)

## Performance Tips
1. For data tensors (`src`, `dst`, `diff_src`, `diff_dst`), use memory formats
   for which the last logical axis is the last in the physical memory layout.

2. For `mean`/`variance`, use the memory format that follows the data memory
   format; i.e., if the data format is #dnnl_tnc, the best performance can be
   expected for statistics with #dnnl_tn and suboptimal for statistics with the
   #dnnl_nt format.

3. For backward propagation, use the same memory format for `src`, `diff_dst`,
   and `diff_src` (the format of `diff_dst` and `diff_src` are always the same
   because of the API). Different formats are functionally supported but lead to
   highly suboptimal performance.

4. Use in-place operations whenever possible (see caveats in General Notes).

## Examples

| Engine  | Name                                 | Comments
| :--     | :--                                  | :--
| CPU/GPU | @ref layer_normalization_example_cpp | @copydetails layer_normalization_example_cpp_short
