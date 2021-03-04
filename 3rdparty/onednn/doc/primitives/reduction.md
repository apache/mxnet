Reduction {#dev_guide_reduction}
============================
>
> [API Reference](@ref dnnl_api_reduction)
>

## General

The reduction primitive performs reduction operation on arbitrary data. Each
element in the destination is the result of reduction operation with specified
algorithm along one or multiple source tensor dimensions:

\f[
    \dst(f) = \mathop{reduce\_op}\limits_{r}\src(r),
\f]

where \f$reduce\_op\f$ can be max, min, sum, mul, mean, Lp-norm and
Lp-norm-power-p, \f$f\f$ is an index in an idle dimension and \f$r\f$ is an
index in a reduction dimension.

Mean:

\f[
    \dst(f) = \frac{\sum\limits_{r}\src(r)} {R},
\f]

where \f$R\f$ is the size of a reduction dimension.

Lp-norm:

\f[
    \dst(f) = \root p \of {\mathop{eps\_op}(\sum\limits_{r}|src(r)|^p, eps)},
\f]

where \f$eps\_op\f$ can be max and sum.

Lp-norm-power-p:

\f[
    \dst(f) = \mathop{eps\_op}(\sum\limits_{r}|src(r)|^p, eps),
\f]

where \f$eps\_op\f$ can be max and sum.

### Notes
 * The reduction primitive requires the source and destination tensors to have
   the same number of dimensions.
 * Reduction dimensions are of size 1 in a destination tensor.
 * The reduction primitive does not have a notion of forward or backward
   propagations.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index |
| ---                    | ---                      |
| \src                   | DNNL_ARG_SRC             |
| \dst                   | DNNL_ARG_DST             |

## Implementation Details

### General Notes
 * The \dst memory format can be either specified explicitly or by
   #dnnl::memory::format_tag::any (recommended), in which case the primitive
   will derive the most appropriate memory format based on the format of the
   source tensor.

### Data Type Support

The source and destination tensors may have `f32`, `bf16`, or `int8` data types.
See @ref dev_guide_data_types page for more details.

### Data Representation

#### Sources, Destination

The reduction primitive works with arbitrary data tensors. There is no special
meaning associated with any of the dimensions of a tensor.

## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.
2. GPU is not supported


## Performance Tips

1. Whenever possible, avoid specifying different memory formats for source
   and destination tensors.

## Examples

| Engine | Name                       | Comments
| :--    | :--                        | :--
| CPU    | @ref reduction_example_cpp | @copydetails reduction_example_cpp_short
