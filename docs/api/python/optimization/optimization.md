# Optimization: initialize and update weights

## Overview

This document summaries the APIs used to initialize and update the model weights
during training

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.initializer
    mxnet.optimizer
    mxnet.lr_scheduler
```

and how to develop a new optimization algorithm in MXNet.

Assume there there is a pre-defined ``Symbol`` and a ``Module`` is created for
it

```python
>>> data = mx.symbol.Variable('data')
>>> label = mx.symbol.Variable('softmax_label')
>>> fc = mx.symbol.FullyConnected(data, name='fc', num_hidden=10)
>>> loss = mx.symbol.SoftmaxOutput(fc, label, name='softmax')
>>> mod = mx.mod.Module(loss)
>>> mod.bind(data_shapes=[('data', (128,20))], label_shapes=[('softmax_label', (128,))])
```

Next we can initialize the weights with values sampled uniformly from
``[-1,1]``:

```python
>>> mod.init_params(mx.initializer.Uniform(scale=1.0))
```

Then we will train a model with standard SGD which decreases the learning rate
by multiplying 0.9 for each 100 batches.

```python
>>> lr_sch = mx.lr_scheduler.FactorScheduler(step=100, factor=0.9)
>>> mod.init_optimizer(
...     optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ('lr_scheduler', lr_sch)))
```

Finally run ``mod.fit(...)`` to start training.

## The ``mxnet.initializer`` package

```eval_rst
.. currentmodule:: mxnet.initializer
```

The base class ``Initializer`` defines the default behaviors to initialize
various parameters, such as set bias to 1, except for the weight. Other classes
then defines how to initialize the weight.

```eval_rst
.. autosummary::
    :nosignatures:

    Initializer
    Uniform
    Normal
    Load
    Mixed
    Zero
    One
    Constant
    Orthogonal
    Xavier
    MSRAPrelu
    Bilinear
    FusedRNN
```

## The ``mxnet.optimizer`` package

```eval_rst
.. currentmodule:: mxnet.optimizer
```

The base class ``Optimizer`` accepts commonly shared arguments such as
``learning_rate`` and defines the interface. Each other class in this package
implements one weight updating function.

```eval_rst
.. autosummary::
    :nosignatures:

    Optimizer
    SGD
    NAG
    RMSProp
    Adam
    AdaGrad
    AdaDelta
    DCASGD
    SGLD
```

## The ``mxnet.lr_scheduler`` package

```eval_rst
.. currentmodule:: mxnet.lr_scheduler
```

The base class ``LRScheduler`` defines the interface, while other classes
implement various schemes to change the learning rate during training.

```eval_rst
.. autosummary::
    :nosignatures:

    LRScheduler
    FactorScheduler
    MultiFactorScheduler
```

## Implement a new algorithm

Most classes listed in this document are implemented in Python by using ``NDArray``.
So implementing new weight updating or initialization functions is
straightforward.

For `initializer`, create a subclass of ``Initializer`` and define the
`_init_weight` method. We can also change the default behaviors to initialize
other parameters such as `_init_bias`. See
[`initializer.py`](https://github.com/dmlc/mxnet/blob/master/python/mxnet/initializer.py)
for examples.

For ``optimizer``, create a subclass of ``Optimizer``
and implement two methods ``create_state`` and ``update``. Also add
``@mx.optimizer.Optimizer.register`` before this class. See
[`optimizer.py`](https://github.com/dmlc/mxnet/blob/master/python/mxnet/optimizer.py)
for examples.

For `lr_scheduler`, create a subclass of `LRScheduler` and then implement the
`__call__` method. See
[`lr_scheduler.py`](https://github.com/dmlc/mxnet/blob/master/python/mxnet/lr_scheduler.py)
for examples.

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.optimizer
    :members:
.. automodule:: mxnet.lr_scheduler
    :members:
.. automodule:: mxnet.initializer
    :members:
```

<script>auto_index("api-reference");</script>
