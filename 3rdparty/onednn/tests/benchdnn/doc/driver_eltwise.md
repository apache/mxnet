# Element-wise Driver

## Usage
``` sh
    ./benchdnn --eltwise [benchdnn-knobs] [eltwise-knobs] [eltwise-desc] ...
```

where *eltwise-knobs* are:

 - `--dir={FWD_D [default], BWD_D}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32 [default], bf16, f16, s32, s8}` -- src and dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={RELU [default], ...}` -- dnnl_eltwise algorithm. Refer to
            [eltwise primitive](https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html)
            for details.
 - `--alpha=FLOAT` -- float value corresponding to algorithm operation.
            Refer to ``Floating point arguments`` below.
 - `--beta=FLOAT` -- float value corresponding to algorithm operation.
            Refer to ``Floating point arguments`` below.
 - `--attr-post-ops="STRING"` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            The default is `false`.

and *eltwise-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Floating point arguments
Some operations support `alpha` argument such as `BRELU`, `CLIP`, `ELU`,
`LINEAR`, `POW` and `RELU`. `CLIP`, `LINEAR` and `POW` also support `beta`
argument.

The `alpha` and `beta` parameters should meet algorithm requirements, otherwise
the problem will be silently skipped. For instance:
* An algorithm does not use `alpha`, but non-zero value is used;
* An algorithm does not use `beta`, but non-zero value is used;
* An algorithm expects `alpha` is non-negative, but negative value is used;
* An algorithm expects `alpha` is less than or equal to `beta`, but `alpha`
    value greater than `beta` is used;

This behavior allows using `,` (comma) operator to run multiple configurations
without dealing with the corner cases.

The default set for `alpha` and `beta` is {0, 0.25, -0.25}.


## Essence of Testing
Fill input data in four ranges: positive/negative integers up to 10 and
positive/negative fractions up to 1.0. This covers special areas of all
algorithm kinds. There is a general threshold; however, it cannot be applied
everywhere. That is why there are some special cases. For details, refer to
``eltwise/eltwise.cpp::compare()``.


## Examples

Run the eltwise set from an input file with the default settings:
``` sh
    ./benchdnn --eltwise --batch=inputs/eltwise/test_eltwise_all
```

Run a specific eltwise problem with the f32 data type and in-place memory mode,
iterating over memory layouts and forward and backward prop kinds:
``` sh
    ./benchdnn --eltwise --dir=FWD_D,BWD_D --dt=f32 --tag=nchw,nChw16c \
               --inplace=true 50x192x55x55
```

More examples with different driver options can be found at
inputs/eltwise/test_eltwise_all. Examples with different benchdnn options can be
found at driver_conv.md.
