Binary {#dev_guide_binary}
====================

>
> [API Reference](@ref dnnl_api_binary)
>

## General

The binary primitive computes the result of a binary elementwise operation
between tensors source 0 and source 1 (the variable names follow the standard
@ref dev_guide_conventions):

\f[
    \dst(\overline{x}) =
        \src_0(\overline{x}) \mathbin{op} \src_1(\overline{x}),
\f]

where \f$op\f$ is addition, multiplication, division, get maximum value or get
minimum value.

The binary primitive does not have a notion of forward or backward propagations.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index                                                  |
| ---                    | ---                                                                       |
| \f$\src_0\f$           | DNNL_ARG_SRC_0                                                            |
| \f$\src_1\f$           | DNNL_ARG_SRC_1                                                            |
| \dst                   | DNNL_ARG_DST                                                              |
| \f$binary post-op\f$   | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1 |

## Implementation Details

### General Notes

 * The binary primitive requires all source and destination tensors to have the
   same number of dimensions.

 * The binary primitive supports implicit broadcast semantics for source 1. It
   means that if some dimension has value of one, this value will be used to
   compute an operation with each point of source 0 for this dimension.

 * The \dst memory format can be either specified explicitly or by
   #dnnl::memory::format_tag::any (recommended), in which case the primitive
   will derive the most appropriate memory format based on the format of the
   source 0 tensor. The \dst tensor dimensions must match the ones of the
   source 0 tensor.

 * The binary primitive supports in-place operations, meaning that source 0
   tensor may be used as the destination, in which case its data will
   be overwritten. In-place mode requires the \dst and source 0 data types to be
   either the same or an arbitrary combination of `int8` data types. 

### Post-ops and Attributes

The following attributes are supported:

| Type      | Operation                                       | Description                                                                    | Restrictions
| :--       | :--                                             | :--                                                                            | :--
| Attribute | [Scales](@ref dnnl::primitive_attr::set_scales) | Scales the corresponding input tensor by the given scale factor(s).            | The corresponding tensor has integer data type. Only one scale per tensor is supported. Input tensors only. |
| Post-op   | [Sum](@ref dnnl::post_ops::append_sum)          | Adds the operation result to the destination tensor instead of overwriting it. |                                                                                                             |
| Post-op   | [Eltwise](@ref dnnl::post_ops::append_eltwise)  | Applies an @ref dnnl_api_eltwise operation to the result.                      |                                                                                                             |
| Post-op   | [Binary](@ref dnnl::post_ops::append_binary)    | Applies a @ref dnnl_api_binary operation to the result                         | General binary post-op restrictions                                                                         |

### Data Types Support

The source and destination tensors may have `f32`, `bf16`, or `int8` data types.
See @ref dev_guide_data_types page for more details.

### Data Representation

#### Sources, Destination

The binary primitive works with arbitrary data tensors. There is no special
meaning associated with any of tensors dimensions.

## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **CPU**
    - Destination memory descriptor should completely match source 0 memory
      descriptor.

## Performance Tips

1. Whenever possible, avoid specifying different memory formats for source
   tensors.

## Examples

| Engine  | Name                    | Comments
| :--     | :--                     | :--
| CPU/GPU | @ref binary_example_cpp | @copydetails binary_example_cpp_short
