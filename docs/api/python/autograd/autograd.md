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

# Autograd Package

```eval_rst
.. currentmodule:: mxnet.autograd
```

## Overview

The `autograd` package enables automatic
differentiation of NDArray operations.
In machine learning applications,
`autograd` is often used to calculate the gradients
of loss functions with respect to parameters.


## Record vs Pause

`autograd` records computation history on the fly to calculate gradients later.
This is only enabled inside a `with autograd.record():` block.
A `with auto_grad.pause()` block can be used inside a `record()` block
to temporarily disable recording.

To compute gradient with respect to an `NDArray` `x`, first call `x.attach_grad()`
to allocate space for the gradient. Then, start a `with autograd.record()` block,
and do some computation. Finally, call `backward()` on the result:

```python
import mxnet as mx
x = mx.nd.array([1,2,3,4])
x.attach_grad()
with mx.autograd.record():
    y = x * x + 1
y.backward()
print(x.grad)
```

Which outputs:

```
[ 2.  4.  6.  8.]
<NDArray 4 @cpu(0)>
```

Gradient recording is enabled during the scope of the `with mx.autograd.record():` statement, then
disabled when we go out of that scope.

It can be also set manually by executing `mx.autograd.set_recording(True)`, and turning it off after
we no longer want to record operations with `mx.autograd.set_recording(False)`.


## Train mode and Predict Mode

Some operators (Dropout, BatchNorm, etc) behave differently in
training and making predictions.
This can be controlled with `train_mode` and `predict_mode` scope.

By default, MXNet is in `predict_mode`.
A `with autograd.record()` block by default turns on `train_mode`
(equivalent to ``with autograd.record(train_mode=True)``).
To compute a gradient in prediction mode (as when generating adversarial examples),
call record with `train_mode=False` and then call `backward(train_mode=False)`

Although training usually coincides with recording,
this isn't always the case.
To control *training* vs *predict_mode* without changing
*recording* vs *not recording*,
use a `with autograd.train_mode():`
or `with autograd.predict_mode():` block.

Detailed tutorials are available in Part 1 of
[the MXNet gluon book](http://gluon.mxnet.io/).


# Higher order gradient

Some operators support higher order gradients. Some operators support differentiating multiple
times, and others two, most just once.

For calculating higher order gradients, we can use the `mx.autograd.grad` function while recording
and then call backward, or call `mx.autograd.grad` two times. If we do the latter, is important that
the first call uses `create_graph=True` and `retain_graph=True` and the second call uses
`create_graph=False` and `retain_graph=True`. Otherwise we will not get the results that we want. If
we would be to recreate the graph in the second call, we would end up with a graph of just the
backward nodes, not the full initial graph that includes the forward nodes.

The pattern to calculate higher order gradients is the following:

```python
from mxnet import ndarray as nd
from mxnet import autograd as ag
x = nd.array([1,2,3])
x.attach_grad()
def f(x):
    # Any function which supports higher oder gradient
    return nd.log(x)
```

If the operators used in `f` don't support higher order gradients you will get an error like
`operator ... is non-differentiable because it didn't register FGradient attribute.`. This means
that it doesn't support getting the gradient of the gradient. Which is, running backward on
the backward graph.

Using mxnet.autograd.grad multiple times:

```python
with ag.record():
    y = f(x)
    x_grad = ag.grad(heads=y, variables=x, create_graph=True, retain_graph=True)[0]
    x_grad_grad = ag.grad(heads=x_grad, variables=x, create_graph=False, retain_graph=False)[0]
```

Running backward on the backward graph:

```python
with ag.record():
    y = f(x)
    x_grad = ag.grad(heads=y, variables=x, create_graph=True, retain_graph=True)[0]
x_grad.backward()
x_grad_grad = x.grad
```

Both methods are equivalent, except that in the second case, retain_graph on running backward is set
to False by default. But both calls are running a backward pass as on the graph as usual to get the
gradient of the first gradient `x_grad` with respect to `x` evaluated at the value of `x`.

For more examples, check the [higher order gradient unit tests](https://github.com/apache/incubator-mxnet/blob/master/tests/python/unittest/test_higher_order_grad.py).


<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

## Autograd

```eval_rst
.. autosummary::
    :nosignatures:

    record
    pause
    train_mode
    predict_mode
    backward
    set_training
    is_training
    set_recording
    is_recording
    mark_variables
    Function
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.autograd
    :members:
```

<script>auto_index("api-reference");</script>
