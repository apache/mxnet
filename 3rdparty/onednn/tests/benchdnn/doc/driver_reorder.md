# Reorder Driver

## Usage
``` sh
    ./benchdnn --reorder [benchdnn-knobs] [reorder-knobs] [reorder-desc] ...
```

where *reorder-knobs* are:

 - `--sdt={f32 [default], s32, s8, u8, bf16, f16}` -- src data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--ddt={f32 [default], s32, s8, u8, bf16, f16}` -- dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--stag={nchw [default], ...}` -- physical src memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={nchw [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--attr-oscale="STRING"` -- output scale primitive attribute. No oscale is
            set by default. Refer to [attributes](knobs_attr.md) for details.
 - `--attr-zero-points="STRING"` -- zero points primitive attribute. No zero
            points are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--attr-post-ops="STRING"` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--def-scales={N1[,N2][,N3]...}` -- input scales, separated by ','.
            Example: 0.125, 0.25, 0.5, 1, 2, 4, 8
 - `--alg={reference [default], bootstrap}` -- reorder testing mode. `bootstrap`
            tests memory with weights compensation.
 - `--oflag={none [default], conv_s8s8, gconv_s8s8, conv_zp_comp,
            gconv_zp_comp}` -- memory descriptor extra field specifier. Also
            sets compensation mask based on the flag value. Only applicable
            when `--alg=bootstrap`.
            Multiple flags can be specified when separated by a ':'. Currently,
            the only accepted combination is '(g)conv_s8s8:(g)conv_zp_comp'.
            Note: the '(g)conv_zp_comp' flag is only meant for testing purposes.
            This is used to insert compensation data during optimized
            zero-point computation, and is transparent to the user.
 - `--cross-engine={none [default], cpu2gpu, gpu2cpu}` -- defines what kind of
            cross-engine reorder will be used. If `--engine` is set to `cpu`,
            `none` is the only supported value.

and *reorder-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Essence of Testing
TBA.


## Examples

Run the reorder set from an input file with the default settings:
``` sh
    ./benchdnn --reorder --batch=inputs/reorder/test_reorder_all
```

Run two specific reorders with s8 src and dst data type, bootstrap algorithm,
and specific input and output physical memory layouts. First problem without
a flag; second problem with the `conv_s8s8` flag:
``` sh
    ./benchdnn --reorder --alg=bootstrap --sdt=s8 --ddt=s8 \
               --stag=hwio --dtag=OIhw4i16o4i 32x32x3x3 \
               --oflag=conv_s8s8 16x32x7x5
```

More examples with different driver options can be found at
inputs/reorder/test_reorder_all. Examples with different benchdnn options can be
found at driver_conv.md.
