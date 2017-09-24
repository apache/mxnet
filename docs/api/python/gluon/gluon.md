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
    VariationalDropoutCell
```



## Conrib

### Recurrent Layers

```eval_rst
.. currentmodule:: mxnet.gluon.contrib.rnn
```


```eval_rst
.. autosummary::
    :nosignatures:

    Conv1DRNNCell
    Conv2DRNNCell
    Conv3DRNNCell
    Conv1DLSTMCell
    Conv2DLSTMCell
    Conv3DLSTMCell
    Conv1DGRUCell
    Conv2DGRUCell
    Conv3DGRUCell
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
    CTCLoss
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
    CIFAR100
    ImageRecordDataset
    ImageFolderDataset
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

#### MobileNet

```eval_rst
.. autosummary::
    :nosignatures:

    mobilenet1_0
    mobilenet0_75
    mobilenet0_5
    mobilenet0_25
```

```eval_rst
.. autosummary::
    :nosignatures:

    MobileNet
```


## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.gluon
    :members:
    :imported-members:
    :special-members:

.. automodule:: mxnet.gluon.nn
    :members:
    :imported-members:
    :special-members:

.. automodule:: mxnet.gluon.rnn
    :members:
    :imported-members:
    :special-members:

.. automodule:: mxnet.gluon.loss
    :members:

.. automodule:: mxnet.gluon.utils
    :members:

.. automodule:: mxnet.gluon.data
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.data.vision
    :members:

.. automodule:: mxnet.gluon.model_zoo.vision
    :members:
    :imported-members:
```

<script>auto_index("api-reference");</script>
