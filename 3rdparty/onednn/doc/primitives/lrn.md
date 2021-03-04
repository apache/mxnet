Local Response Normalization (LRN) {#dev_guide_lrn}
====================================================

>
> [API Reference](@ref dnnl_api_lrn)
>

## General

The LRN primitive performs a forward or backward local response normalization.

### Forward

The LRN operation is defined by the following formulas (the variable names
follow the standard @ref dev_guide_conventions):

LRN [across channels](#dnnl_lrn_across_channels):

\f[
    \dst(n, c, h, w) =
        \left\{k + \frac{\alpha}{n_{l}}
            \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2-1}
                (\src(n, c+i, h, w))^2
        \right\}^{-\beta}
        \cdot
        \src(n, c, h, w),
\f]

LRN [within channel](#dnnl_lrn_within_channel):

\f[
    \dst(n, c, h, w) =
        \left\{k + \frac{\alpha}{n_{l}}
            \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2-1}
            \sum\limits_{j=-(n_{l}-1)/2}^{(n_{l}+1)/2-1}
                (\src(n, c, h+i, w+j))^2
        \right\}^{-\beta}
        \cdot
        \src(n, c, h, w),
\f]

where \f$n_{l}\f$ is the @p local_size. Formulas are provided for 2D spatial
data case.

### Backward

The backward propagation computes \f$\diffsrc(n, c, h, w)\f$, based on
\f$\diffdst(n, c, h, w)\f$ and \f$\src(n, c, h, w)\f$.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index |
| ---                    | ---                      |
| \src                   | DNNL_ARG_SRC             |
| \dst                   | DNNL_ARG_DST             |
| workspace              | DNNL_ARG_WORKSPACE       |
| \diffsrc               | DNNL_ARG_DIFF_SRC        |
| \diffdst               | DNNL_ARG_DIFF_DST        |


## Implementation Details

### General Notes

1. During training, LRN might or might not require a workspace on forward and
   backward passes. The behavior is implementation specific. Optimized
   implementations typically require a workspace and use it to save some
   intermediate results from the forward pass that accelerate computations on
   the backward pass. To check whether a workspace is required, query the LRN
   primitive descriptor for the workspace. Success indicates that the workspace
   is required and its description will be returned.

2. The memory format and data type for `src` and `dst` are assumed to be the
   same, and in the API are typically referred to as `data` (e.g., see
   `data_desc` in dnnl::lrn_forward::desc::desc()). The same holds for
   `diff_src` and `diff_dst`. The corresponding memory descriptors are referred
   to as `diff_data_desc`.

### Data Type Support

The LRN primitive supports the following combinations of data types:

| Propagation        | Source / Destination |
| :--                | :--                  |
| forward / backward | f32, bf16            |
| forward            | f16                  |

@warning
    There might be hardware and/or implementation specific restrictions. Check
    the [Implementation Limitations](@ref dg_lrn_impl_limits) section below.

### Data Representation

#### Source, Destination, and Their Gradients

Like most other primitives, the LRN primitive expects the following
tensors:

| Spatial | Source / Destination
| :--     | :--
| 0D      | \f$N \times C\f$
| 1D      | \f$N \times C \times W\f$
| 2D      | \f$N \times C \times H \times W\f$
| 3D      | \f$N \times C \times D \times H \times W\f$

The LRN primitive is optimized for the following memory formats:

| Spatial | Logical tensor | Implementations optimized for memory formats
| :--     | :--            | :--
| 2D      | NCHW           | #dnnl_nchw (#dnnl_abcd), #dnnl_nhwc (#dnnl_acdb), *optimized^*

Here *optimized^* means the format that
[comes out](@ref memory_format_propagation_cpp)
of any preceding compute-intensive primitive.

### Post-ops and Attributes

The LRN primitive does not support any post-ops or attributes.


@anchor dg_lrn_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **GPU**
    - Supports only 2D spatial case.


## Performance Tips

1. For backward propagation, use the same memory format for `src`, `diff_dst`,
   and `diff_src` (the format of the `diff_dst` and `diff_src` are always the
   same because of the API). Different formats are functionally supported but
   lead to highly suboptimal performance.

## Examples

| Engine  | Name                 | Comments
| :--     | :--                  | :--
| CPU/GPU | @ref lrn_example_cpp | @copydetails lrn_example_cpp_short
