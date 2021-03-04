# MatMul Driver

## Usage
``` sh
    ./benchdnn --matmul [benchdnn-knobs] [matmul-knobs] [matmul-desc] ...
```

where *matmul-knobs* are:

 - `--cfg={f32 [default], ...}` -- refer to ``Configurations`` in
            driver_conv.md.
 - `--stag={ab [default], any, ...}` -- memory format of the source memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--wtag={ab [default], any, ...}` -- memory format of the weights memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={ab [default], any, ...}` -- memory format of the destination memory.
            Refer to [tags](knobs_tag.md) for details.
 - `--runtime_mb=BOOL` -- specify whether `mb` dimension is a run-time
            parameter (will be deprecated soon. See `--runtime_dims_masks`).
 - `--runtime_m=BOOL` -- specify whether `m` dimension is a run-time parameter
            (will be deprecated soon. See `--runtime_dims_masks`).
 - `--runtime_n=BOOL` -- specify whether `n` dimension is a run-time parameter
            (will be deprecated soon. See `--runtime_dims_masks`).
 - `--runtime_k=BOOL` -- specify whether `k` dimension is a run-time parameter
            (will be deprecated soon. See `--runtime_dims_masks`).
 - `--attr-oscale="STRING"` -- output scale primitive attribute. No oscale is
            set by default. Refer to [attributes](knobs_attr.md) for details.
 - `--attr-zero-points="STRING"` -- zero points primitive attribute. No zero
            points are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--attr-post-ops="STRING"` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--bia_dt={undef [default], f32, s32, s8, u8}` -- bias data type.
            To run MatMul without bias, use `undef` data type (default).
            Refer to [data types](knobs_dt.md) for details.
 - `--bia_mask=INT` -- a bit-mask that indicates which bias dimensions are
            broadcasted. 0-bit means broadcast, 1-bit means full dimension.
- `--runtime_dims_masks=[INT][:INT]` -- a bit-mask values for `src` and
            `weights` that indicates if a dimension is `DNNL_RUNTIME_DIM_VAL`
            (indicated as 1-bit in corresponding dimension position). Default is
            `0` for all dimensions, meaning all tensor dimensions are fully
            defined at primitive creation.

and *matmul-desc* is a problem descriptor. The canonical form is:
```
    d0xd1xd2x..xMxK:d0xd1xd2x..xKxN[:d0xd1xd2x..xMxN][nS]
```
Here `x` is delimiter for dimensions within a tensor and `:` is delimiter for
tensors in the order `src`, `weights` and `dst`. The `dst` is optional and each
of its individual dimensions are computed as
`max(src_dimension, weights_dimension)` by the driver if not provided by user.
`d0`, `d1`, `d2` and so on are dimension values of the corresponding tensor,
where as `m`, `n` and `k` are inner dimensions for matrix multiplication.

**Deprecated desc (only supports up to 3D)**
```
    [mbX]mXnXkX_nS
```
Here `X` is an integer number and `S` is a string literal without spaces (`n`
stands for name). The special symbol `_` is ignored, so it may be used as a
delimiter for better readability.

The `mb` can be omitted, in which case the problem is treated as regular
2D matrix multiplication. With `mb` set to a non-zero value, batched matrix
multiplication is used.

## Examples

Run the default validation set of MatMul using `inputs/matmul/test_matmul_all`
file:
``` sh
    ./benchdnn --matmul --batch=inputs/matmul/test_matmul_all
```

Run single precision matrix multiplication with all sizes provided at run-time:
``` sh
    ./benchdnn --matmul \
               --runtime_dims_masks=3:3 \
               10x30:30x20
```

The same can be expressed with deprecated matmul desc as below:
``` sh
    ./benchdnn --matmul \
               --runtime_m=true --runtime_n=true --runtime_k=true \
               m10n20k30
```

Run reduced precision (int8) matrix multiplication with asymmetric quantization
for the source and destination memory (both use `uint8_t` data type) and
symmetric quantization for weights memory (with `int8_t` data type and allowing
the library to choose the proper memory format), with zero points provided at
runtime, but sizes specified at creation time:
``` sh
    ./benchdnn --matmul \
               --cfg=u8s8u8 \
               --wtag=any \
               --attr-zero-points=src:1*_dst:-2* \
               10x30:30x20 # or m10n20k3 with deprecated matmul desc
```

Run single precision batched matrix multiplication with bias, of which only the
full dimension is along the `n`-axis:
``` sh
    ./benchdnn --matmul \
               --bia_dt=f32 --bia_mask=4 \
               2x10x30:2x30x20 # or mb2m10n20k3 with deprecated matmul desc
```

More examples with different driver options can be found at
inputs/matmul/test_matmul_all.
