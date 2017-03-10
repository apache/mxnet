# Loss API

```eval_rst
.. currentmodule:: mxnet.loss
```

## Overview

The loss API is used to define training objectives. `mxnet.loss` contains
some standard objective functions such as mean squred error and softmax
cross entropy. For example, a simple linear regression can be defined as:

```python
# Assuming X and Y are (N, C) and (N, 1) matrices
train_iter = mx.io.NDArrayIter(X, Y)
data = mx.sym.Variable('data')
label = mx.sym.Variable('label')
output = mx.sym.FullyConnected(data, num_hidden=1)
loss = mx.loss.l2_loss(output, label)
model = mx.mod.Module(loss, data_names=['data'])
model.fit(train_iter, eval_metric=loss.metric, num_epochs=10)
```

You can also easily define their own objectives. For example, the
following code is equivalent to using `mxnet.loss.l2_loss`:

```python
data = mx.sym.Variable('data')
label = mx.sym.Variable('label')
output = mx.sym.FullyConnected(data, num_hidden=1)
L = mx.sym.square(output - label)/2
loss = mx.loss.custom_loss(L, output, label_names=['label'])
```

Multiple objectives can be combined to get a joint objective for multi-task
learning:

```python
loss1 = mx.sym.l1_loss(output1, label1, weight=0.2)
loss2 = mx.sym.l2_loss(output2, label2, weight=0.8)
loss = mx.sym.multi_loss([loss1, loss2])
```

## Predefined Objectives
```eval_rst
.. autosummary::
    :nosignatures:

    custom_loss
    multi_loss
    l2_loss
    l1_loss
    cross_entropy_loss
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.loss
    :members:
```

<script>auto_index("api-reference");</script>
