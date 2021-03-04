# Sum Driver

## Usage
``` sh
    ./benchdnn --sum [benchdnn-knobs] [sum-knobs] [sum-desc] ...
```

where *sum-knobs* are:

 - `--sdt={f32:f32 [default], ...}` -- src data type. Refer to ``Inputs`` below.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], s32, s8, u8, bf16, f16}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={nchw:nchw [default], ...}` -- physical src memory layout.
            Refer to ``Inputs`` below.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={undef [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--scales={N1:N2[:N3]...}` -- input scales. Refer to ``Scales`` below.
            The default is 0.25, 1, 4.

and *sum-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Inputs
The sum primitive accepts at least two input sources. That is why it has a
slightly different interface than almost every other driver. Specifying several
inputs is done via the special ':' delimiter, e.g. --sdt=f32:s32, which means
that the first source will be of type f32 and the second will be s32. `--stag`
option must either specify a single tag (this tag will be used for all input
tensors) or specify the same number of tags as the number of data types in
`--sdt` delimited by ':'.


## Scales
The same logic as in ``Inputs`` works for sum scales. `--scales` supports two
modes:
- Single value. In this case, a given value will be broadcasted for each input.
- Multiple values. In this case, the driver will require the amount of scales to
  coincide with the amount of the `--sdt` option. Each input will use its own
  scale value.


## Essence of Testing
Fill input data with integers so that an output will not overflow in the f16 or
bf16 data types.

Note that threshold is set to a small value, which means that not each scale may
pass it.


## Examples

Run the set of sum from sum/test_sum_all with the default settings:
``` sh
    ./benchdnn --sum --batch=inputs/sum/test_sum_all
```

Run a specific sum problem with three inputs of s8, s8, and u8 data types, with
each input in nhwc physical memory layout, providing scales for each input
individually and requesting the output in the s32 data type and nhwc layout:
``` sh
    ./benchdnn --sum --sdt=s8:s8:u8 --ddt=s32
               --stag=nhwc:nhwc:nhwc --dtag=nhwc
               --scales=2:4:1 16x16x3x5
```

Run a specific sum problem with the default scales, requesting to deduce the
output with the f32 data type, iterating over source data types and physical
memory layouts:
``` sh
    ./benchdnn --sum --sdt=bf16:bf16,f32:f32 --ddt=f32
               --stag=nChw16c:nChw16c,nchw:nchw --dtag=undef 16x16x16x16
```

More examples with different driver options can be found at
inputs/sum/test_sum_all. Examples with different benchdnn options can be
found at driver_conv.md.
