# Shuffle Driver

## Usage
``` sh
    ./benchdnn --shuffle [benchdnn-knobs] [shuffle-knobs] [shuffle-desc] ...
```

where *shuffle-knobs* are:

 - `--dir={FWD_D [default], BWD_D}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32 [default], s32, s8, u8, bf16, f16}` -- src and dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--axis=INT` -- dimension on which operation will be performed.
            Default is `1`, corresponds to channels in logical memory layout.
 - `--group=INT` -- number of elements to shuffle. The default is `1`.

and *shuffle-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Essence of Testing
Fill input data with integers so that an output will not overflow in the f16 or
bf16 data types. As no compute operations are executed, we set a zero threshold
and expect a precise answer with the reference implementation.


## Examples

Run the set of shuffles from an input file with the default settings:
``` sh
    ./benchdnn --shuffle --batch=inputs/shuffle/test_shuffle_all
```

Run a specific shuffle problem with forward prop_kind and plain physical memory
layout. Group elements by 4 and over `h` dimension, iterating by all listed
data types:
``` sh
    ./benchdnn --shuffle --dir=FWD_D --dt=f32,s32,s8,u8,bf16 \
               --tag=nchw --group=4 --axis=2 1x68x56x56
```

More examples with different driver options can be found at
inputs/shuffle/test_shuffle_all. Examples with different benchdnn options can be
found at driver_conv.md.
