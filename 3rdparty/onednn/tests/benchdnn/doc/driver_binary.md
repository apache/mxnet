# Binary Driver

## Usage
``` sh
    ./benchdnn --binary [benchdnn-knobs] [binary-knobs] [binary-desc] ...
```

where *binary-knobs* are:

 - `--sdt={f32 [default], bf16}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], bf16}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={nchw:nchw [default], ...}` -- physical src memory layout.
            Refer to ``Inputs`` below.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={ADD [default], MUL, MAX, MIN, DIV}` -- algorithm for binary
            operations.
            Refer to [binary primitive](https://oneapi-src.github.io/oneDNN/dev_guide_binary.html)
            for details.
 - `--attr-scales="STRING"` -- per argument scales primitive attribute. No
            scales are set by default. Refer to [attributes](knobs_attr.md) for
            details.
 - `--attr-post-ops="STRING"` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            Default is `false`.

and *binary-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN:NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Inputs
The input specification is similar to the concat driver except that binary
primitive accepts exactly two input shapes delimited by ':'.  The `--stag` and
`--sdt` options, if present, must have same number of arguments.

## Element broadcasting
Element broadcasting supported for the second tensor: it can have fewer
dimensions than the first one. The trailing dimensions are implicitly padded
with dimensions of size 1. For example, for a 8x7x6:1x7 problem the 1x7 tensor
dimensions are first padded to 1x7x1. Then, according to the definition of the
primitive, each element of the second tensor is broadcast across the first and
the last dimensions when applying a binary operation.

## Essence of Testing
Input data is initialized with floating point values while ensuring that there
is no overflow in f16 or bf16 data types.

## Examples

Run the set of binary primitive problems from `binary/test_binary_all` with the
default settings:
``` sh
    ./benchdnn --binary --batch=inputs/binary/test_binary_all
```

Run a specific binary primitive problem:
- Data type is `f32` for source and destination tensors.
- Source tensors use `nhwc` memory format.
- The operation is out-of-place.
- The operation is element-wise multiplication.
- Second source is broadcast across 3 innermost dimensions of the first
  source.
``` sh
    ./benchdnn --binary --sdt=f32:f32 --ddt=f32 --stag=nhwc:nhwc \
               --alg=MUL --inplace=false 8x8x3x5:8
```

More examples with different driver options can be found at
inputs/binary/test_binary_all. Examples with different benchdnn options can be
found at driver_conv.md.
