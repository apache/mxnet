# Gluon Package


```eval_rst
.. currentmodule:: mxnet.gluon
```

```eval_rst
.. warning:: This package is currently experimental and may change in the near future.
```

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

## Overview

Gluon package is a high-level interface for MXNet designed to be easy to use while
keeping most of the flexibility of low level API. Gluon supports both imperative
and symbolic programming, making it easy to train complex models imperatively
in Python and then deploy with symbolic graph in C++ and Scala.

## Parameter

```eval_rst
.. currentmodule:: mxnet.gluon
```


```eval_rst
.. currentmodule:: mxnet.gluon
.. autoclass:: mxnet.gluon.Parameter
    :members:
.. autoclass:: mxnet.gluon.ParameterDict
    :members:
```


## Containers

```eval_rst
.. currentmodule:: mxnet.gluon
.. autoclass:: mxnet.gluon.Block
    :members:

    .. automethod:: forward
.. autoclass:: mxnet.gluon.HybridBlock
    :members:

    .. automethod:: hybrid_forward
```

## Neural Network Layers

```eval_rst
.. currentmodule:: mxnet.gluon.nn
```

### Containers


```eval_rst
.. currentmodule:: mxnet.gluon.nn

    .. automethod:: __call__
.. autoclass:: mxnet.gluon.nn.Sequential
    :members:
.. autoclass:: mxnet.gluon.nn.HSequential
    :members:
```


### Basic Layers


```eval_rst
.. currentmodule:: mxnet.gluon.nn  
.. autoclass:: mxnet.gluon.nn.Dense
    :members:
.. autoclass:: mxnet.gluon.nn.Activation
    :members:
.. autoclass:: mxnet.gluon.nn.Dropout
    :members:
.. autoclass:: mxnet.gluon.nn.BatchNorm
    :members:
.. autoclass:: mxnet.gluon.nn.LeakyReLU
    :members:
.. autoclass:: mxnet.gluon.nn.Embedding
    :members:
```


### Convolutional Layers


```eval_rst
.. currentmodule:: mxnet.gluon.nn  
.. autoclass:: mxnet.gluon.nn.Conv1D
    :members:
.. autoclass:: mxnet.gluon.nn.Conv2D
    :members:
.. autoclass:: mxnet.gluon.nn.Conv3D
    :members:
.. autoclass:: mxnet.gluon.nn.Conv1DTranspose
    :members:
.. autoclass:: mxnet.gluon.nn.Conv2DTranspose
    :members:
.. autoclass:: mxnet.gluon.nn.Conv3DTranspose
    :members:
```



### Pooling Layers


```eval_rst
.. currentmodule:: mxnet.gluon.nn
.. autoclass:: mxnet.gluon.nn.MaxPool1D
    :members:
.. autoclass:: mxnet.gluon.nn.MaxPool2D
    :members:
.. autoclass:: mxnet.gluon.nn.MaxPool3D
    :members:
.. autoclass:: mxnet.gluon.nn.AvgPool1D
    :members:
.. autoclass:: mxnet.gluon.nn.AvgPool2D
    :members:
.. autoclass:: mxnet.gluon.nn.AvgPool3D
    :members:
.. autoclass:: mxnet.gluon.nn.GlobalMaxPool1D
    :members:
.. autoclass:: mxnet.gluon.nn.GlobalMaxPool2D
    :members:
.. autoclass:: mxnet.gluon.nn.GlobalMaxPool3D
    :members:
.. autoclass:: mxnet.gluon.nn.GlobalAvgPool1D
    :members:
.. autoclass:: mxnet.gluon.nn.GlobalAvgPool2D
    :members:
.. autoclass:: mxnet.gluon.nn.GlobalAvgPool3D
    :members:
```



## Recurrent Layers

```eval_rst
.. currentmodule:: mxnet.gluon.rnn
```


```eval_rst
.. autoclass:: mxnet.gluon.rnn.RecurrentCell
    :members:

    .. automethod:: __call__
.. autoclass:: mxnet.gluon.rnn.RNN
    :members:
.. autoclass:: mxnet.gluon.rnn.LSTM
    :members:
.. autoclass:: mxnet.gluon.rnn.GRU
    :members:
.. autoclass:: mxnet.gluon.rnn.RNNCell
    :members:
.. autoclass:: mxnet.gluon.rnn.LSTMCell
    :members:
.. autoclass:: mxnet.gluon.rnn.GRUCell
    :members:
.. autoclass:: mxnet.gluon.rnn.SequentialRNNCell
    :members:
.. autoclass:: mxnet.gluon.rnn.BidirectionalCell
    :members:
.. autoclass:: mxnet.gluon.rnn.DropoutCell
    :members:
.. autoclass:: mxnet.gluon.rnn.ZoneoutCell
    :members:
.. autoclass:: mxnet.gluon.rnn.ResidualCell
    :members:
```


## Trainer

```eval_rst
.. currentmodule:: mxnet.gluon
```


```eval_rst
.. autoclass:: mxnet.gluon.Trainer
    :members:
```


## Loss functions

```eval_rst
.. currentmodule:: mxnet.gluon.loss
```


```eval_rst
.. automethod:: mxnet.gluon.loss.custom_loss
.. automethod:: mxnet.gluon.loss.multitask_loss
.. automethod:: mxnet.gluon.loss.l1_loss
.. automethod:: mxnet.gluon.loss.l2_loss
.. automethod:: mxnet.gluon.loss.softmax_cross_entropy_loss
```


## Utilities

```eval_rst
.. currentmodule:: mxnet.gluon.utils
```


```eval_rst
.. automethod:: mxnet.gluon.utils.split_data
.. automethod:: mxnet.gluon.utils.split_and_load
.. automethod:: mxnet.gluon.utils.clip_global_norm
```

<script>auto_index("api-reference");</script>
