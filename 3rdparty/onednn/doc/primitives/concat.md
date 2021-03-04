Concat {#dev_guide_concat}
==========================

>
> [API Reference](@ref dnnl_api_concat)
>

## General

The concat primitive concatenates \f$N\f$ tensors over `concat_dimension` (here
designated \f$C\f$) and is defined as (the variable names follow the
standard @ref dev_guide_conventions):

\f[
    \dst(\overline{ou}, c, \overline{in}) =
        \src_i(\overline{ou}, c', \overline{in}),
\f]

where \f$c = C_1 + .. + C_{i-1} {}_{} + c'\f$.

The concat primitive does not have a notion of forward or backward
propagation. The backward propagation for the concatenation operation is
simply an identity operation.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index |
| ---                    | ---                      |
| \src                   | DNNL_ARG_MULTIPLE_SRC    |
| \dst                   | DNNL_ARG_DST             |

## Implementation Details

### General Notes

1. The \dst memory format can be either specified by a user or derived by
   the primitive. The recommended way is to allow the primitive to choose the
   most appropriate format.

2. The concat primitive requires all source and destination tensors to have the
   same shape except for the `concat_dimension`. The destination dimension for the
   `concat_dimension` must be equal to the sum of the `concat_dimension` dimensions of
   the sources (i.e. \f$C = \sum_i C_i\f$).
   Implicit broadcasting is not supported.

### Data Types Support

The concat primitive supports arbitrary data types for source and destination
tensors according to the @ref dev_guide_data_types page. However, it is
required that all source tensors are of the same data type (but not necessarily
matching the data type of the destination tensor).

### Data Representation

The concat primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions.

### Post-ops and Attributes

The concat primitive does not support any post-ops or attributes.

## Implementation Limitations

1. The primitive works with several memory formats, such as plain formats
   #dnnl_nchw, #dnnl_nhwc, and blocked formats #dnnl_nChw16c, #dnnl_nCdhw8c that
   appear in convolutions. The primitive does not support non-blocked formats
   that are typically used in prepacked weights, such as:
   - [Winograd](@ref dev_guide_convolution) format #dnnl_format_kind_wino,
   - [RNN](@ref dev_guide_rnn) format #dnnl_format_kind_rnn_packed, or
   - Blocked format with attached [compensation](@ref dg_i8_comp_s12)
     (#dnnl_memory_extra_flag_compensation_conv_s8s8),
     that is used in `s8s8` convolutions (see @ref dev_guide_int8_computations).

2. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

## Performance Tips

1. Whenever possible, avoid specifying the destination memory format so that the
   primitive is able to choose the most appropriate one.

2. The concat primitive is highly optimized for the cases in which all source
   tensors have same memory format and data type matches the destination tensor
   data type. For other cases, more general but slower code is working.
   Consider reordering sources to the same data format before using the concat
   primitive.

## Examples

| Engine  | Name                    | Comments
| :--     | :--                     | :--
| CPU/GPU | @ref concat_example_cpp | @copydetails concat_example_cpp_short
