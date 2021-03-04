# Attributes

## Usage
```
    --attr-oscale=POLICY[:SCALE[*]]
    --attr-scales=ARG:POLICY[:SCALE][_...]
    --attr-zero-points=ARG:ZEROPOINT[*][_...]
    --attr-post-ops='SUM[:SCALE[:DATA_TYPE]];'
                    'ELTWISE[:ALPHA[:BETA[:SCALE]]];[...;]'
                    'DW_K3S1P1[:DST_DT[:OUTPUTSCALE]];'
                    'DW_K3S2P1[:DST_DT[:OUTPUTSCALE]];'
                    'BINARY:DT[:POLICY];'
```

`--attr-oscale` defines output scale primitive attribute. `POLICY` specifies the
way scale values will be applied to the output tensor. `SCALE` is optional
argument, parsed as a real number that specifies either a common output scale
(for `common` policy) or a starting point for a policy with non-zero mask
(e.g. `per_oc`), which uses many scales. The default scale is `1.0`. Asterisk
mark (`*`) is an optional addition to `SCALE` indicating the scales will be
passed to a primitive at run-time.

`POLICY` supported values are:
  - `none`       (the default) means no output scale is applied.
  - `common`     corresponds to `mask = 0` and means a whole tensor will be
                 multiplied by a single SCALE value.
  - `per_oc`     corresponds to `mask = 1 << 1` and means elements of dim1 will
                 be multiplied by scale factors different for each point. Number
                 of scale factors equals to dims[1].
  - `per_dim_0`  corresponds to `mask = 1 << 0` and means elements of dim0 will
                 be multiplied by scale factors different for each point. Number
                 of scale factors equals to dims[0].
  - `per_dim_1`  same as `per_oc`.
  - `per_dim_01` corresponds to `mask = (1 << 0) + (1 << 1)` and means elements
                 of dim0 and dim1 will be multiplied by scale factors different
                 for a pair of {dim0, dim1} points. Number of scale factors
                 equals to dims[0] * dims[1].

`--attr-scales` defines input scales per memory argument primitive attribute.
This attribute is supported only for integer data types as of now. `ARG`
specifies which memory argument will be modified with input scale. `POLICY` and
`SCALE` have the same semantics and meaning as for `--attr-oscale`. To specify
more than one memory argument, underscore (`_`) delimiter is used.

`ARG` supported values are:
  - `src` corresponds to `DNNL_ARG_SRC`
  - `src1` corresponds to `DNNL_ARG_SRC_1`

`POLICY` supported values are:
  - `none`
  - `common`

`--attr-zero-points` defines zero points per memory argument primitive
attribute. This attribute is supported only for integer data types as of now.
`ARG` specifies which memory argument will be modified with zero points.
`ZEROPOINT` is an integer value which will be subtracted from each tensor point.
Asterisk mark (`*`) is an optional addition to `ZEROPOINT` indicating the value
will be passed to a primitive at run-time. To specify more than one memory
argument, underscore (`_`) delimiter is used.

`ARG` supported values are:
  - `src` corresponds to `DNNL_ARG_SRC`
  - `wei` corresponds to `DNNL_ARG_WEIGHTS`
  - `dst` corresponds to `DNNL_ARG_DST`

`--attr-post-ops` defines post operations primitive attribute. Depending on
post operations kind, the syntax differs, but regardless the kind, single quotes
are used in the beginning and in the end in a string literal, even when empty
post operations are passed.

`SUM` post operation kind appends operation result to the output. It supports
optional arguments `SCALE` parsed as a real number, which scales the operation
result before appending, and `DATA_TYPE` argument which defines sum data type
parameter. No data type limitations are applied. Only single `SUM` operation
can be applied to the output tensor.

`ELTWISE` post operation kind applies one of supported element-wise algorithms
to the operation result and then stores it. It supports optional arguments
`ALPHA` and `BETA` parsed as real numbers. To specify `BETA`, `ALPHA` must be
specified. `SCALE` has same notation and semantics as for `SUM` kind, but
requires both `ALPHA` and `BETA` to be specified. `SCALE` is applicable only
when output tensor has integer data type.

`DW_K3S1P1` and `DW_K3S2P1` post operation kinds append depthwise convolution
with kernel size of 3, strides of 1 and 2 correspondently and paddings of 1.
These kinds are applicable only for convolution operation with kernel size of 1
as of now. They support optional argument `DST_DT`, which defines destination
tensor data type. Refer to [data types](knobs_dt.md) for details. Optional
argument `OUTPUTSCALE` defines the semantics of output scale as for
`--attr-oscale` with the same syntax. It requires `DST_DT` to be specified.

`BINARY` post operation kind applies one of supported binary algorithms to the
operation result and then stores it. It requires mandatory argument of `DT`
specifying data type of second memory operand. It supports optional argument of
`POLICY` giving a hint what are the dimensions for a second memory operand. Yet
so far only `COMMON` and `PER_OC` policy values are supported.

Operations may be called in any order, e.g. apply `SUM` at first and then apply
`ELTWISE`, or vice versa - apply `ELTWISE` and then `SUM` it with destination.

`ELTWISE` supported values are:
  - Eltwise operations that support no alpha or beta:
      - `abs`
      - `exp`
      - `exp_dst`
      - `gelu_erf`
      - `gelu_tanh`
      - `log`
      - `logistic`
      - `logistic_dst`
      - `round`
      - `sqrt`
      - `sqrt_dst`
      - `square`
      - `soft_relu`
      - `tanh`
      - `tanh_dst`
  - Eltwise operations that support only alpha:
      - `bounded_relu`
      - `elu`
      - `elu_dst`
      - `relu`
      - `relu_dst`
      - `swish`
  - Eltwise operations that support both alpha and beta:
      - `clip`
      - `linear`
      - `pow`

`BINARY` supported values are:
  - `add`
  - `max`
  - `min`
  - `mul`

## Examples:

Run a set of f32 forward convolutions without bias appending accumulation into
destination and perform relu on the output with scale set to 0.5:
``` sh
    ./benchdnn --conv --cfg=f32 --dir=FWD_D \
               --attr-post-ops="'sum;relu:0.5'" --batch=conv_tails
```

Run a 1D-spatial reorder problem with s8 input data and u8 output data in four
different physical memory layout combinations {ncw, ncw}, {ncw, nwc},
{nwc, ncw} and {nwc, nwc} applying output scale 2.5 for each output point:
``` sh
    ./benchdnn --reorder --sdt=s8 --ddt=u8 \
               --stag=ncw,nwc --dtag=ncw,nwc \
               --attr-oscale=common:2.5 2x8x8
```

Run a binary problem with s8 input data and u8 output data in nc layout
applying scales to both inputs without any post operations:
``` sh
    ./benchdnn --binary --sdt=u8:s8 --ddt=u8 --stag=nc:nc \
               --attr-scales=src:common:1.5_src1:common:2.5 \
               --attr-post-ops="''" 100x100:100x100
```

Run a 1x1 convolution fused with depthwise convolution where output scales set
to 0.5 for 1x1 convolution and 1.5 for depthwise post-op followed by a relu
post-op. The final dst datatype after the fusion in the example below is `s8`.
The weights datatype is inferred as `s8`, `f32` and `bf16` for int8, f32 and
bf16 convolutions respectively.
``` sh
  ./benchdnn --conv --cfg=u8s8u8 --attr-oscale=per_oc:0.5 \
             --attr-post-ops="'relu;dw_k3s1p1:s8:per_oc:1.5;relu'" \
             ic16oc16ih4oh4kh1ph0
```

Run a convolution problem with binary post operation:
``` sh
  ./benchdnn --conv --attr-post-ops="'add:s32:common'" ic16oc16ih4oh4kh1ph0
```
