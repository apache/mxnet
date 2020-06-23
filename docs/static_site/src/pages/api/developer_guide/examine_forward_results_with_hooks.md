---
layout: page_category
title:  Examine forward results with hooks
category: Developer Guide
permalink: /api/dev-guide/examine_forward_results_with_hooks
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Examine forward results with hooks

There are currently three ways to register a function in an MXNet Gluon Block for execution:

* before `forward` via [register_forward_pre_hook]({{"/api/python/docs/api/gluon/block.html#mxnet.gluon.Block.register_forward_pre_hook" | relative_url }})
* after `forward` via [register_forward_hook]({{"/api/python/docs/api/gluon/block.html#mxnet.gluon.Block.register_forward_hook" | relative_url }})
* as a callback via [register_op_hook]({{"/api/python/docs/api/gluon/block.html#mxnet.gluon.Block.register_op_hook" | relative_url }})

## Pre-forward hook

To register a hook prior to forward execution, the requirement is that the registered operation **should not modify the input or output**. For example: `hook(block, input) -> None`. This is useful to get a summary before execution.

```
import mxnet as mx
from mxnet.gluon import nn

block = nn.Dense(10)
block.initialize()
print("{}".format(block))
# Dense(None -> 10, linear)

def pre_hook(block, input) -> None:  # notice it has two arguments, one block and one input
    print("{}".format(block))
    return
    
# register
pre_handle = block.register_forward_pre_hook(pre_hook)
input = mx.nd.ones((3, 5))
print(block(input))

# Dense(None -> 10, linear)
# [[ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]]
# <NDArray 3x10 @cpu(0)>
```

We can `detach` a hook from a block:


```
pre_handle.detach()
print(block(input))

# [[ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]]
# <NDArray 3x10 @cpu(0)>
```

Notice `Dense(None -> 10, linear)` is not displayed anymore.

## Post-forward hook

Registering a hook after forward execution is very similar to pre-forward hook (as explained above) with the difference that the hook signature should be `hook(block, input, output) -> None` where **hook should not modify the input and output.** Continuing from the above example:


```
def post_hook(block, intput, output) -> None:
    print("{}".format(block))
    return
    
post_handle = block.register_forward_hook(post_hook)
print(block(input))

# Dense(5 -> 10, linear)
# [[ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]]
# <NDArray 3x10 @cpu(0)>
```


Notice the difference between `pre_hook` and `post_hook` results due to shape inference after `forward` is done executing.

## Callback hook

We can register a callback monitor to monitor all operators that are called by the `HybridBlock` **after hybridization** with `register_op_hook(callback, monitor_all=False) ` where the callback signature should be:


```
callback(node_name: str,  opr_name: str, arr: NDArray) -> None
```

where `node_name` is the name of the tensor being inspected (str), `opr_name` is the name of the operator producing or consuming that tensor (str) and `arr` the tensor being inspected (NDArray).


```
import mxnet as mx
from mxnet.gluon import nn

def mon_callback(node_name, opr_name, arr):
    print("{}".format(node_name))
    print("{}".format(opr_name))
    return
    
model = nn.HybridSequential(prefix="dense_")
with model.name_scope():
     model.add(mx.gluon.nn.Dense(2))

model.initialize()
model.hybridize()
model.register_op_hook(mon_callback, monitor_all=True)
print(model(mx.nd.ones((2, 3, 4))))

# b'dense_dense0_fwd_data'
# b'FullyConnected'
# b'dense_dense0_fwd_weight'
# b'FullyConnected'
# b'dense_dense0_fwd_bias'
# b'FullyConnected'
# b'dense_dense0_fwd_output'
# b'FullyConnected'
# [[-0.05979988 -0.16349721]
#  [-0.05979988 -0.16349721]]
# <NDArray 2x2 @cpu(0)>
```


Setting `monitor_all=False` will print only the output:


```
`# b'dense_dense0_fwd_output'`
`# b'FullyConnected'``
# [[-0.05979988 -0.16349721]
#  [-0.05979988 -0.16349721]]
# <NDArray 2x2 @cpu(0)`
```

Note that to get the internal operator node names, one can use `model.collect_params().items()`.
