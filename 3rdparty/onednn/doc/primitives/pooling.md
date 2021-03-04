Pooling {#dev_guide_pooling}
============================

>
> [API Reference](@ref dnnl_api_pooling)
>

## General

The pooling primitive performs forward or backward max or average pooling
operation on 1D, 2D, or 3D spatial data.

### Forward

The pooling operation is defined by the following formulas.
We show formulas only for 2D spatial data which are straightforward to
generalize to cases of higher and lower dimensions. Variable names follow the
standard @ref dev_guide_conventions.

Max pooling:

\f[
    \dst(n, c, oh, ow) =
        \max\limits_{kh, kw}
        \left(
            \src(n, c, oh \cdot SH + kh \cdot (DH + 1) - PH_L, ow \cdot SW + kw \cdot (DW + 1) - PW_L)
        \right)
\f]

Average pooling:

\f[
    \dst(n, c, oh, ow) =
        \frac{1}{DENOM}
        \sum\limits_{kh, kw}
            \src(n, c, oh \cdot SH + kh \cdot (DH + 1) - PH_L, ow \cdot SW + kw \cdot (DW + 1) - PW_L)
\f]

Here output spatial dimensions are calculated similarly to how they are done in
@ref dev_guide_convolution.

Average pooling supports two algorithms:
- #dnnl_pooling_avg_include_padding, in which case \f$DENOM = KH \cdot KW\f$,
- #dnnl_pooling_avg_exclude_padding, in which case \f$DENOM\f$ equals to the
  size of overlap between an averaging window and images.

> TODO: a picture would be nice here.

#### Difference Between Forward Training and Forward Inference

- Max pooling requires a `workspace` for the #dnnl_forward_training propagation
  kind, and does not require it for #dnnl_forward_inference (see details below).

### Backward

The backward propagation computes \f$\diffsrc(n, c, h,
w)\f$, based on \f$\diffdst(n, c, h, w)\f$ and (in
case of max pooling) `workspace`.

## Execution Arguments
When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index                                                  |
| ---                    | ---                                                                       |
| \src                   | DNNL_ARG_SRC                                                              |
| \dst                   | DNNL_ARG_DST                                                              |
| workspace              | DNNL_ARG_WORKSPACE                                                        |
| \diffsrc               | DNNL_ARG_DIFF_SRC                                                         |
| \diffdst               | DNNL_ARG_DIFF_DST                                                         |
| \f$binary post-op\f$   | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1 |

## Implementation Details

### General Notes

1. During training, max pooling requires a workspace on forward
   (#dnnl_forward_training) and backward passes to save indices where a maximum
   was found. The workspace format is opaque, and the indices cannot be restored
   from it. However, one can use backward pooling to perform up-sampling (used
   in some detection topologies). The workspace can be created via
   `workspace_desc()` from the pooling primitive descriptor.

2. A user can use memory format tag #dnnl_format_tag_any for `dst` memory
   descriptor when creating pooling forward propagation. The library would
   derive the appropriate format from the `src` memory descriptor. However,
   the `src` itself must be defined. Similarly, a user can use memory format tag
   #dnnl_format_tag_any for the `diff_src` memory descriptor when creating
   pooling backward propagation.

### Data Type Support

The pooling primitive supports the following combinations of data types:

| Propagation        | Source / Destination | Accumulation data type (used for average pooling only)
| :--                | :--                  | :--
| forward / backward | f32, bf16            | f32
| forward            | f16                  | f16
| forward            | s8, u8, s32          | s32
| forward inference  | s8, u8 / f32         | f32
| forward inference  | f32 / s8, u8         | f32

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_pool_impl_limits) section below.

### Data Representation

#### Source, Destination, and Their Gradients

Like other CNN primitives, the pooling primitive expects data to be
an \f$N \times C \times W\f$ tensor for the 1D spatial case,
an \f$N \times C \times H \times W\f$ tensor for the 2D spatial case, and
an \f$N \times C \times D \times H \times W\f$ tensor for the 3D spatial case.

The pooling primitive is optimized for the following memory formats:

| Spatial | Logical tensor | Data type   | Implementations optimized for memory formats                       |
| :--     | :--            | :--         | :--                                                                |
| 1D      | NCW            | f32         | #dnnl_ncw (#dnnl_abc), #dnnl_nwc (#dnnl_acb), *optimized^*         |
| 1D      | NCW            | s32, s8, u8 | #dnnl_nwc (#dnnl_acb), *optimized^*                                |
| 2D      | NCHW           | f32         | #dnnl_nchw (#dnnl_abcd), #dnnl_nhwc (#dnnl_acdb), *optimized^*     |
| 2D      | NCHW           | s32, s8, u8 | #dnnl_nhwc (#dnnl_acdb), *optimized^*                              |
| 3D      | NCDHW          | f32         | #dnnl_ncdhw (#dnnl_abcde), #dnnl_ndhwc (#dnnl_acdeb), *optimized^* |
| 3D      | NCDHW          | s32, s8, u8 | #dnnl_ndhwc (#dnnl_acdeb), *optimized^*                            |

Here *optimized^* means the format that
[comes out](@ref memory_format_propagation_cpp)
of any preceding compute-intensive primitive.

### Post-ops and Attributes

| Propagation | Type    | Operation                                    | Description                                            | Restrictions                        |
| :--         | :--     | :--                                          | :--                                                    | :--                                 |
| Forward     | Post-op | [Binary](@ref dnnl::post_ops::append_binary) | Applies a @ref dnnl_api_binary operation to the result | General binary post-op restrictions |

@anchor dg_pool_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **CPU**
    - Different data types of source and destination in forward inference
      are not supported.

## Performance Tips

N/A

## Examples

| Engine  | Name                     | Comments
| :--     | :--                      | :--
| CPU/GPU | @ref pooling_example_cpp | @copydetails pooling_example_cpp_short
