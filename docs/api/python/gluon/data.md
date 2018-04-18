# Gluon Data API

## Overview

This document lists the data APIs in Gluon:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.gluon.data
    mxnet.gluon.data.vision
```

The `Gluon Data` API, defined in the `gluon.data` package, provides useful dataset loading
and processing tools, as well as common public datasets.

In the rest of this document, we list routines provided by the `gluon.data` package.

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

#### Vision Datasets

```eval_rst
.. currentmodule:: mxnet.gluon.data.vision.datasets
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

#### Vision Transforms

```eval_rst
.. currentmodule:: mxnet.gluon.data.vision.transforms
```

Transforms can be used to augment input data during training. You
can compose multiple transforms sequentially, for example:

```python
from mxnet.gluon.data.vision import MNIST, transforms
from mxnet import gluon
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomResizedCrop(224),
    transforms.RandomBrightness(0.1),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)])
data = MNIST(train=True).transform_first(transform)
data_loader = gluon.data.DataLoader(data, batch_size=32, num_workers=1)
for data, label in data_loader:
    # do something with data and label
```

```eval_rst
.. autosummary::
    :nosignatures:

    Compose
    Cast
    ToTensor
    Normalize
    RandomResizedCrop
    CenterCrop
    Resize
    RandomFlipLeftRight
    RandomFlipTopBottom
    RandomBrightness
    RandomContrast
    RandomSaturation
    RandomHue
    RandomColorJitter
    RandomLighting
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.gluon.data
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.data.vision
    :members:

.. automodule:: mxnet.gluon.data.vision.datasets
    :members:
    
.. automodule:: mxnet.gluon.data.vision.transforms
    :members:

```

<script>auto_index("api-reference");</script>
