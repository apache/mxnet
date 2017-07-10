# Foo Package


```eval_rst
.. currentmodule:: mxnet.foo
```

```eval_rst
.. warning:: This package is currently experimental and may change in the near future.
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

## Overview

Foo package is a high-level interface for MXNet designed to be easy to use while
keeping most of the flexibility of low level API. Foo supports both imperative
and symbolic programming, making it easy to train complex models imperatively
in Python and then deploy with symbolic graph in C++ and Scala.

## Parameter

```eval_rst
.. currentmodule:: mxnet.foo
```


```eval_rst
.. currentmodule:: mxnet.foo
.. autoclass:: mxnet.foo.Parameter
    :members:
.. autoclass:: mxnet.foo.ParameterDict
    :members:
```


## Containers

```eval_rst
.. currentmodule:: mxnet.foo
.. autoclass:: mxnet.foo.Block
    :members:

    .. automethod:: forward
.. autoclass:: mxnet.foo.HybridBlock
    :members:

    .. automethod:: hybrid_forward
```

## Neural Network Layers

```eval_rst
.. currentmodule:: mxnet.foo.nn
```

### Containers


```eval_rst
.. currentmodule:: mxnet.foo.nn

    .. automethod:: __call__
.. autoclass:: mxnet.foo.nn.Sequential
    :members:
.. autoclass:: mxnet.foo.nn.HSequential
    :members:
```


### Basic Layers


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


### Convolutional Layers


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



### Pooling Layers


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



## Recurrent Layers

```eval_rst
.. currentmodule:: mxnet.foo.rnn
```


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


## Trainer

```eval_rst
.. currentmodule:: mxnet.foo
```


```eval_rst
.. autoclass:: mxnet.foo.Trainer
    :members:
```


## Loss functions

```eval_rst
.. currentmodule:: mxnet.foo.loss
```


```eval_rst
.. automethod:: mxnet.foo.loss.custom_loss
.. automethod:: mxnet.foo.loss.multitask_loss
.. automethod:: mxnet.foo.loss.l1_loss
.. automethod:: mxnet.foo.loss.l2_loss
.. automethod:: mxnet.foo.loss.softmax_cross_entropy_loss
```


## Utilities

```eval_rst
.. currentmodule:: mxnet.foo.utils
```


```eval_rst
.. automethod:: mxnet.foo.utils.split_data
.. automethod:: mxnet.foo.utils.split_and_load
.. automethod:: mxnet.foo.utils.clip_global_norm
```

<script>auto_index("api-reference");</script>
