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
can compose multiple transforms sequentially (taking note of which functions should be applied before and after `ToTensor`).

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
