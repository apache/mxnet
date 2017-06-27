# Foo Package

```eval_rst
.. currentmodule:: mxnet.foo
```

```eval_rst
.. warning:: This package is currently experimental and may change in the near future.
```

## Overview

Foo package is a high-level interface for MXNet designed to be easy to use while
keeping most of the flexibility of low level API. Foo supports both imperative
and symbolic programming, making it easy to train complex models imperatively
in Python and then deploy with symbolic graph in C++ and Scala.

## Parameter

```eval_rst
.. currentmodule:: mxnet.foo
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. autoclass:: mxnet.foo.Parameter
    :members:
.. autoclass:: mxnet.foo.ParameterDict
    :members:
```

<script>auto_index("api-reference");</script>


## Neural Network Layers

```eval_rst
.. currentmodule:: mxnet.foo.nn
```

### Containers

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. currentmodule:: mxnet.foo.nn
.. autoclass:: mxnet.foo.nn.Layer
    :members:

    .. automethod:: __call__
.. autoclass:: mxnet.foo.nn.HybridLayer
    :members:

    .. automethod:: __call__
.. autoclass:: mxnet.foo.nn.Sequential
    :members:
.. autoclass:: mxnet.foo.nn.HSequential
    :members:
```

<script>auto_index("api-reference");</script>

### Basic Layers

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. currentmodule:: mxnet.foo.nn  
.. autoclass:: mxnet.foo.nn.Dense
    :members:
.. autoclass:: mxnet.foo.nn.Activation
    :members:
.. autoclass:: mxnet.foo.nn.Dropout
    :members:
.. autoclass:: mxnet.foo.nn.BatchNorm
    :members:
.. autoclass:: mxnet.foo.nn.LeakyReLU
    :members:
.. autoclass:: mxnet.foo.nn.Embedding
    :members:
```

<script>auto_index("api-reference");</script>

### Convolutional Layers

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. currentmodule:: mxnet.foo.nn  
.. autoclass:: mxnet.foo.nn.Conv1D
    :members:
.. autoclass:: mxnet.foo.nn.Conv2D
    :members:
.. autoclass:: mxnet.foo.nn.Conv3D
    :members:
.. autoclass:: mxnet.foo.nn.Conv1DTranspose
    :members:
.. autoclass:: mxnet.foo.nn.Conv2DTranspose
    :members:
.. autoclass:: mxnet.foo.nn.Conv3DTranspose
    :members:
```

<script>auto_index("api-reference");</script>


### Pooling Layers

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. currentmodule:: mxnet.foo.nn
.. autoclass:: mxnet.foo.nn.MaxPool1D
    :members:
.. autoclass:: mxnet.foo.nn.MaxPool2D
    :members:
.. autoclass:: mxnet.foo.nn.MaxPool3D
    :members:
.. autoclass:: mxnet.foo.nn.AvgPool1D
    :members:
.. autoclass:: mxnet.foo.nn.AvgPool2D
    :members:
.. autoclass:: mxnet.foo.nn.AvgPool3D
    :members:
.. autoclass:: mxnet.foo.nn.GlobalMaxPool1D
    :members:
.. autoclass:: mxnet.foo.nn.GlobalMaxPool2D
    :members:
.. autoclass:: mxnet.foo.nn.GlobalMaxPool3D
    :members:
.. autoclass:: mxnet.foo.nn.GlobalAvgPool1D
    :members:
.. autoclass:: mxnet.foo.nn.GlobalAvgPool2D
    :members:
.. autoclass:: mxnet.foo.nn.GlobalAvgPool3D
    :members:
```

<script>auto_index("api-reference");</script>


## Recurrent Layers

```eval_rst
.. currentmodule:: mxnet.foo.rnn
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. autoclass:: mxnet.foo.rnn.RecurrentCell
    :members:

    .. automethod:: __call__
.. autoclass:: mxnet.foo.rnn.RNN
    :members:
.. autoclass:: mxnet.foo.rnn.LSTM
    :members:
.. autoclass:: mxnet.foo.rnn.GRU
    :members:
.. autoclass:: mxnet.foo.rnn.RNNCell
    :members:
.. autoclass:: mxnet.foo.rnn.LSTMCell
    :members:
.. autoclass:: mxnet.foo.rnn.GRUCell
    :members:
.. autoclass:: mxnet.foo.rnn.SequentialRNNCell
    :members:
.. autoclass:: mxnet.foo.rnn.BidirectionalCell
    :members:
.. autoclass:: mxnet.foo.rnn.DropoutCell
    :members:
.. autoclass:: mxnet.foo.rnn.ZoneoutCell
    :members:
.. autoclass:: mxnet.foo.rnn.ResidualCell
    :members:
```

<script>auto_index("api-reference");</script>

## Trainer

```eval_rst
.. currentmodule:: mxnet.foo
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. autoclass:: mxnet.foo.Trainer
    :members:
```

<script>auto_index("api-reference");</script>

## Loss functions

```eval_rst
.. currentmodule:: mxnet.foo.loss
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automethod:: mxnet.foo.loss.custom_loss
.. automethod:: mxnet.foo.loss.multitask_loss
.. automethod:: mxnet.foo.loss.l1_loss
.. automethod:: mxnet.foo.loss.l2_loss
.. automethod:: mxnet.foo.loss.softmax_cross_entropy_loss
```

<script>auto_index("api-reference");</script>

## Utilities

```eval_rst
.. currentmodule:: mxnet.foo.utils
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automethod:: mxnet.foo.utils.split_data
.. automethod:: mxnet.foo.utils.load_data
```

<script>auto_index("api-reference");</script>
