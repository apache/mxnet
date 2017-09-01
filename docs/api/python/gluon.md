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
.. autosummary::
    :nosignatures:

    Parameter
    ParameterDict
```


## Containers

```eval_rst
.. autosummary::
    :nosignatures:

    Block
    HybridBlock
    SymbolBlock
```

## Neural Network Layers

```eval_rst
.. currentmodule:: mxnet.gluon.nn
```

### Containers

```eval_rst
.. autosummary::
    :nosignatures:

    Sequential
    HybridSequential
```


### Basic Layers


```eval_rst
.. autosummary::
    :nosignatures:

    Dense
    Activation
    Dropout
    BatchNorm
    LeakyReLU
    Embedding
```


### Convolutional Layers


```eval_rst
.. autosummary::
    :nosignatures:

    Conv1D
    Conv2D
    Conv3D
    Conv1DTranspose
    Conv2DTranspose
    Conv3DTranspose
```



### Pooling Layers


```eval_rst
.. autosummary::
    :nosignatures:

    MaxPool1D
    MaxPool2D
    MaxPool3D
    AvgPool1D
    AvgPool2D
    AvgPool3D
    GlobalMaxPool1D
    GlobalMaxPool2D
    GlobalMaxPool3D
    GlobalAvgPool1D
    GlobalAvgPool2D
    GlobalAvgPool3D
```



## Recurrent Layers

```eval_rst
.. currentmodule:: mxnet.gluon.rnn
```


```eval_rst
.. autosummary::
    :nosignatures:

    RecurrentCell
    RNN
    LSTM
    GRU
    RNNCell
    LSTMCell
    GRUCell
    SequentialRNNCell
    BidirectionalCell
    DropoutCell
    ZoneoutCell
    ResidualCell
```


## Trainer

```eval_rst
.. currentmodule:: mxnet.gluon

.. autosummary::
    :nosignatures:

    Trainer
```


## Loss functions

```eval_rst
.. currentmodule:: mxnet.gluon.loss
```


```eval_rst
.. autosummary::
    :nosignatures:

    L2Loss
    L1Loss
    SoftmaxCrossEntropyLoss
    KLDivLoss
```

## Utilities

```eval_rst
.. currentmodule:: mxnet.gluon.utils
```


```eval_rst
.. autosummary::
    :nosignatures:

    split_data
    split_and_load
    clip_global_norm
```

## Data

```eval_rst
.. currentmodule:: mxnet.gluon.data
```

```eval_rst
.. autosummary::
    :nosignatures:

    Dataset
    ArrayDataset
    RecordFileDataset
    ImageRecordDataset
```

```eval_rst
.. autosummary::
    :nosignatures:

    Sampler
    SequentialSampler
    RandomSampler
    BatchSampler
```

```eval_rst
.. autosummary::
    :nosignatures:

    DataLoader
```

### Vision

```eval_rst
.. currentmodule:: mxnet.gluon.data.vision
```

```eval_rst
.. autosummary::
    :nosignatures:

    MNIST
    FashionMNIST
    CIFAR10
```

## Model Zoo

Model zoo provides pre-defined and pre-trained models to help bootstrap machine learning applications.

### Vision

```eval_rst
.. currentmodule:: mxnet.gluon.model_zoo.vision
.. automodule:: mxnet.gluon.model_zoo.vision
```

```eval_rst
.. autosummary::
    :nosignatures:

    get_model
```

#### ResNet

```eval_rst
.. autosummary::
    :nosignatures:

    resnet18_v1
    resnet34_v1
    resnet50_v1
    resnet101_v1
    resnet152_v1
    resnet18_v2
    resnet34_v2
    resnet50_v2
    resnet101_v2
    resnet152_v2
```

```eval_rst
.. autosummary::
    :nosignatures:

    ResNetV1
    ResNetV2
    BasicBlockV1
    BasicBlockV2
    BottleneckV1
    BottleneckV2
    get_resnet
```

#### VGG

```eval_rst
.. autosummary::
    :nosignatures:

    vgg11
    vgg13
    vgg16
    vgg19
    vgg11_bn
    vgg13_bn
    vgg16_bn
    vgg19_bn
```

```eval_rst
.. autosummary::
    :nosignatures:

    VGG
    get_vgg
```

#### Alexnet

```eval_rst
.. autosummary::
    :nosignatures:

    alexnet
```

```eval_rst
.. autosummary::
    :nosignatures:

    AlexNet
```

#### DenseNet

```eval_rst
.. autosummary::
    :nosignatures:

    densenet121
    densenet161
    densenet169
    densenet201
```

```eval_rst
.. autosummary::
    :nosignatures:

    DenseNet
```

#### SqueezeNet

```eval_rst
.. autosummary::
    :nosignatures:

    squeezenet1_0
    squeezenet1_1
```

```eval_rst
.. autosummary::
    :nosignatures:

    SqueezeNet
```

#### Inception

```eval_rst
.. autosummary::
    :nosignatures:

    inception_v3
```

```eval_rst
.. autosummary::
    :nosignatures:

    Inception3
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. autoclass:: mxnet.gluon.Parameter
    :members:
.. autoclass:: mxnet.gluon.ParameterDict
    :members:

.. autoclass:: mxnet.gluon.Block
    :members:

    .. automethod:: __call__
.. autoclass:: mxnet.gluon.HybridBlock
    :members:
.. autoclass:: mxnet.gluon.SymbolBlock
    :members:

.. autoclass:: mxnet.gluon.nn.Sequential
    :members:
.. autoclass:: mxnet.gluon.nn.HybridSequential
    :members:
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

.. autoclass:: mxnet.gluon.Trainer
    :members:

.. autoclass:: mxnet.gluon.loss.L2Loss
    :members:
.. autoclass:: mxnet.gluon.loss.L1Loss
    :members:
.. autoclass:: mxnet.gluon.loss.SoftmaxCrossEntropyLoss
    :members:
.. autoclass:: mxnet.gluon.loss.KLDivLoss
    :members:
.. automethod:: mxnet.gluon.utils.split_data

.. automethod:: mxnet.gluon.utils.split_and_load

.. automethod:: mxnet.gluon.utils.clip_global_norm

.. autoclass:: mxnet.gluon.data.Dataset
    :members:
.. autoclass:: mxnet.gluon.data.ArrayDataset
    :members:
.. autoclass:: mxnet.gluon.data.RecordFileDataset
    :members:
.. autoclass:: mxnet.gluon.data.ImageRecordDataset
    :members:
.. autoclass:: mxnet.gluon.data.Sampler
    :members:
.. autoclass:: mxnet.gluon.data.SequentialSampler
    :members:
.. autoclass:: mxnet.gluon.data.RandomSampler
    :members:
.. autoclass:: mxnet.gluon.data.BatchSampler
    :members:
.. autoclass:: mxnet.gluon.data.DataLoader
    :members:
.. automodule:: mxnet.gluon.data.vision
    :members:

.. automethod:: mxnet.gluon.model_zoo.vision.get_model
.. automethod:: mxnet.gluon.model_zoo.vision.resnet18_v1
.. automethod:: mxnet.gluon.model_zoo.vision.resnet34_v1
.. automethod:: mxnet.gluon.model_zoo.vision.resnet50_v1
.. automethod:: mxnet.gluon.model_zoo.vision.resnet101_v1
.. automethod:: mxnet.gluon.model_zoo.vision.resnet152_v1
.. automethod:: mxnet.gluon.model_zoo.vision.resnet18_v2
.. automethod:: mxnet.gluon.model_zoo.vision.resnet34_v2
.. automethod:: mxnet.gluon.model_zoo.vision.resnet50_v2
.. automethod:: mxnet.gluon.model_zoo.vision.resnet101_v2
.. automethod:: mxnet.gluon.model_zoo.vision.resnet152_v2
.. automethod:: mxnet.gluon.model_zoo.vision.get_resnet
.. autoclass:: mxnet.gluon.model_zoo.vision.ResNetV1
    :members:
.. autoclass:: mxnet.gluon.model_zoo.vision.BasicBlockV1
    :members:
.. autoclass:: mxnet.gluon.model_zoo.vision.BottleneckV1
    :members:
.. autoclass:: mxnet.gluon.model_zoo.vision.ResNetV2
    :members:
.. autoclass:: mxnet.gluon.model_zoo.vision.BasicBlockV2
    :members:
.. autoclass:: mxnet.gluon.model_zoo.vision.BottleneckV2
    :members:
.. automethod:: mxnet.gluon.model_zoo.vision.vgg11
.. automethod:: mxnet.gluon.model_zoo.vision.vgg13
.. automethod:: mxnet.gluon.model_zoo.vision.vgg16
.. automethod:: mxnet.gluon.model_zoo.vision.vgg19
.. automethod:: mxnet.gluon.model_zoo.vision.vgg11_bn
.. automethod:: mxnet.gluon.model_zoo.vision.vgg13_bn
.. automethod:: mxnet.gluon.model_zoo.vision.vgg16_bn
.. automethod:: mxnet.gluon.model_zoo.vision.vgg19_bn
.. automethod:: mxnet.gluon.model_zoo.vision.get_vgg
.. autoclass:: mxnet.gluon.model_zoo.vision.VGG
    :members:
.. automethod:: mxnet.gluon.model_zoo.vision.alexnet
.. autoclass:: mxnet.gluon.model_zoo.vision.AlexNet
    :members:
.. automethod:: mxnet.gluon.model_zoo.vision.densenet121
.. automethod:: mxnet.gluon.model_zoo.vision.densenet161
.. automethod:: mxnet.gluon.model_zoo.vision.densenet169
.. automethod:: mxnet.gluon.model_zoo.vision.densenet201
.. autoclass:: mxnet.gluon.model_zoo.vision.DenseNet
    :members:
.. automethod:: mxnet.gluon.model_zoo.vision.squeezenet1_0
.. automethod:: mxnet.gluon.model_zoo.vision.squeezenet1_1
.. autoclass:: mxnet.gluon.model_zoo.vision.SqueezeNet
    :members:
.. automethod:: mxnet.gluon.model_zoo.vision.inception_v3
.. autoclass:: mxnet.gluon.model_zoo.vision.Inception3
    :members:
```

<script>auto_index("api-reference");</script>
