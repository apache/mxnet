# Direction

**Benchdnn** renames the library propagation kind abstraction into "direction".
The following direction values are supported:

| Prop kind     | Description
| :---          | :---
| FWD_B         | dnnl_forward_training w/ bias
| FWD_D         | dnnl_forward_training w/o bias
| FWD_I         | dnnl_forward_inference
| BWD_D         | dnnl_backward_data
| BWD_WB        | dnnl_backward_weights w/ bias
| BWD_W         | dnnl_backward_weights w/o bias
| BWD_DW        | dnnl_backward

