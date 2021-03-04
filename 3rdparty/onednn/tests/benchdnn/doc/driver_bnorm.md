# Batch Normalization Driver

## Usage
``` sh
    ./benchdnn --bnorm [benchdnn-knobs] [bnorm-knobs] [bnorm-desc] ...
```

where *bnorm-knobs* are:

 - `--dir={FWD_D [default], FWD_I, BWD_D, BWD_DW}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32 [default], s8, bf16, f16}` -- src and dst data types.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--flags=[|G|S|R]` -- batch normalization flags, default `none`; where
            multiple simultaneous flags are supported.
            `G` is dnnl_use_global_stats;
            `S` is dnnl_use_scaleshift;
            `R` is dnnl_fuse_norm_relu;
            Refer to [batch normalization primitive](https://oneapi-src.github.io/oneDNN/dev_guide_batch_normalization.html)
            for details.
 - `--attr-post-ops="STRING"` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            Default is `false`.
 - `--debug-check-ws=BOOL` -- checks if workspace has correct values. Feature is
            based on internal knowledge of a library implementation. The default
            is `false`.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--match=REGEXP` -- run only problems that match the regular expression
            `REGEXP`. By default there is no pattern applied. Note: Windows may
            interpret only string arguments surrounded by double quotation
            marks.

and *bnorm-desc* is a problem descriptor. The canonical form is:
```
    mbXicX_idXihXiwX_epsF_nS
```
Refer to [descriptor](knobs_desc.md) for details. `epsF` stands for Batch
Normalization epsilon value and accepts float F values. The default is `1.f/16`.

## Essence of Testing
TBA.


## Examples

Run a set of bnorms from an input file, using the default settings:
``` sh
    ./benchdnn --bnorm --batch=inputs/bnorm/bnorm_resnet_50
```

Run a named problem with single precision src/dst, skipping all problems that
use the reference implementation, iterating by:
1) blocked memory layouts, where channel blocking equals 8 and 16,
2) forward training, backward by data and weights prop_kinds,
3) all flag combinations:
``` sh
    ./benchdnn --bnorm --dt=f32 --skip-impl=ref --tag=nChw8c,nChw16c \
               --dir=FWD_D,BWD_D,BWD_DW --flags=SR,GS,S \
               mb96_ic192_ih71iw71_n"googlenet_v3:conv_4_4_batchnorm"
```

Run a set of 3D spatial bnorms with s8 src/dst data_type, inference prop_kind,
plain physical memory layout with dense channels, and all three flags specified:
``` sh
    ./benchdnn --bnorm --dt=s8 --tag=ndhwc --dir=FWD_I \
               --flags=GSR --batch=bnorm_3d
```

More examples with different driver options can be found at
inputs/bnorm/test_bnorm_all. Examples with different driver descriptors can be
found at inputs/bnorm/bnorm_***. Examples with different benchdnn options can be
found at driver_conv.md.
