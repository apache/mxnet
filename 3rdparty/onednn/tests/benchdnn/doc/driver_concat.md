# Concat Driver

## Usage
``` sh
    ./benchdnn --concat [benchdnn-knobs] [concat-knobs] [concat-desc] ...
```

where *concat-knobs* are:

 - `--sdt={f32 [default], s32, s8, u8, bf16, f16}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], s32, s8, u8, bf16, f16}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={nchw:nchw [default], ...}` -- physical src memory layout.
            Refer to ``Inputs`` below.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={undef [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--axis=INT` -- dimension on which operation will be performed.
            Default is `1`; corresponds to channels in logical memory layout.

and *concat-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN:NxNxNxNxN[:NxNxNxNxN...]
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Inputs
The concat primitive accepts at least two input sources. That is why it has a
slightly different interface than almost every other driver. Specifying several
inputs is done via the special ':' delimiter, e.g. 2x3x2x2:2x5x2x2, which means
that two tensors of the same shape except one dimension will be concatenated.
`--stag` option must either specify a single tag (this tag will be used for all
input tensors) or specify the same number of tags as the number of tensors
delimited by ':'.


## Essence of Testing
Fill input data with integers so that an output will not overflow in f16 or bf16
data types. As no compute operations are executed, we set a zero threshold and
expect a precise answer with the reference implementation.


## Examples

Run the set of concat from concat/test_concat_all with the default settings:
``` sh
    ./benchdnn --concat --batch=inputs/concat/test_concat_all
```

Run a specific concat problem with three inputs of s32 data type, with each
input in nhwc physical memory layout, with output in u8 data type and nhwc
layout, over the `h` axis, resulting in an 8x8x10x5 tensor:
``` sh
    ./benchdnn --concat --sdt=s32 --ddt=u8 --stag=nhwc:nhwc:nhwc \
               --dtag=nhwc --axis=2 8x8x3x5:8x8x7x5:8x8x0x5
```

Run a specific concat problem over the `c` axis, asking to deduce the output
with the f32 data type, iterating over source data types and physical memory
layouts, resulting in a 16x48x16x16 tensor:
``` sh
    ./benchdnn --concat --sdt=bf16,f32 --ddt=f32 \
               --stag=nChw16c:nChw16c,nchw:nchw --dtag=undef \
               16x16x16x16:16x32x16x16
```

More examples with different driver options can be found at
inputs/concat/test_concat_all. Examples with different benchdnn options can be
found at driver_conv.md.
