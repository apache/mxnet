# Inner Product Driver

## Usage
``` sh
    ./benchdnn --ip [benchdnn-knobs] [ip-knobs] [ip-desc] ...
```

where *ip-knobs* are:

 - `--dir={FWD_B [default], FWD_D, FWD_I, BWD_D, BWD_W, BWD_WB}`
            -- dnnl_prop_kind_t. Refer to [direction](knobs_dir.md) for details.
 - `--cfg={f32 [default], ...}` -- refer to ``Configurations`` in
            driver_conv.md.
 - `--stag={any [default], ...}` -- physical src memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--wtag={any [default], ...}` -- physical wei memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--dtag={any [default], ...}` -- physical dst memory layout.
            Refer to [tags](knobs_tag.md) for details.
 - `--attr-oscale="STRING"` -- output scale primitive attribute. No oscale is
            set by default. Refer to [attributes](knobs_attr.md) for details.
 - `--attr-post-ops="STRING"` -- post operation primitive attribute. No post
            operations are set by default. Refer to [attributes](knobs_attr.md)
            for details.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.

and *ip-desc* is a problem descriptor. The canonical form is:
```
    mbX_icXidXihXiwX_ocX_nS
```
Refer to [descriptor](knobs_desc.md) for details.

## Essence of Testing
TBA.


## Examples

Run the set of ip from inputs/ip/ip_all file with default settings:
``` sh
    ./benchdnn --ip --batch=inputs/ip/ip_all
```

Run a named problem with single precision src and dst, backward by data
prop_kind, applying output scale of `2.25`, appending the result into dst with
output scale of `0.5`, and applying tanh as a post op:
``` sh
    ./benchdnn --ip --dir=BWD_D -attr-oscale="common:2.25" \
               --attr-post-ops="'sum:0.5;tanh'" \
               mb112ic2048_ih1iw1_oc1000_n"resnet:ip1"
```

More examples with different driver options can be found at
inputs/ip/test_ip_all. Examples with different driver descriptors can be found
at inputs/ip/ip_***. Examples with different benchdnn options can be found at
driver_conv.md.
