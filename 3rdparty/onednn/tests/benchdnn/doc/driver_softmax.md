# Softmax Driver

## Usage
``` sh
    ./benchdnn --softmax [benchdnn-knobs] [softmax-knobs] [softmax-desc] ...
```

where *softmax-knobs* are:

 - `--dir={FWD_D [default], BWD_D}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--dt={f32 [default], bf16, f16}` -- src and dst data type.
            Refer to [data types](knobs_dt.md) for details.
 - `--tag={nchw [default], ...}` -- physical src and dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--alg={SOFTMAX [default], LOGSOFTMAX}` -- algorithm type.
            `SOFTMAX` enables softmax primitive;
            `LOGSOFTMAX` enables logsoftmax primitive;
            Refer to [softmax primitive](https://oneapi-src.github.io/oneDNN/dev_guide_softmax.html)
            and [logsoftmax primitive](https://oneapi-src.github.io/oneDNN/dev_guide_logsoftmax.html)
            for details.
 - `--axis=INT` -- dimension on which operation will be performed.
            Default is `1`, corresponds to channels in logical memory layout.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.
 - `--inplace=BOOL` -- memory mode for the primitive. If `true`, it uses input
            memory as output, otherwise, input and output are separate.
            The default is `false`.

and *softmax-desc* is a problem descriptor. The canonical form is:
```
    NxNxNxNxN
```
where N is an integer number. This represents a 3D spatial problem with the
following logical dimensions: N, C, D, H, W. Consider removing each `xN` from
the end to specify fewer dimensions.


## Essence of Testing
### Forward
Fill data the way it tests two modes: max_val < 0 and max_val >= 0;
Test max_val < 0 by using only negative numbers to check correct max_val
subtraction, mostly if library used signed value, not abs.
Test max_val >= 0 by exceeding `exp_overflow_arg` value to check answer does not
contain +infinity (nan) in the answer.

### Backward
Fill input data with negative integers, and expect positive output. This avoids
potential cancellation errors.


## Examples

Run the softmax set from an input file with the default settings:
``` sh
    ./benchdnn --softmax --batch=inputs/softmax/test_softmax_2d
```

Run a specific softmax problem with forward prop_kind, plain physical memory
layout, f32 data type, out-place memory mode, and axis size of 1000:
``` sh
    ./benchdnn --softmax --dir=FWD_D --dt=f32 --tag=nc \
               --inplace=false --axis=1 256x1000
```

Run a specific logsoftmax problem with backward prop_kind, default physical
memory layout, default data type, in-place memory mode, and axis size of 64:
``` sh
    ./benchdnn --softmax --dir=BWD_D --inplace=true \
               --alg=LOGSOFTMAX --axis=3 1x2x112x64
```

More examples with different driver options can be found at
inputs/softmax/test_softmax_all. Examples with different benchdnn options can be
found at driver_conv.md.
