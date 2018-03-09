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

```

<script>auto_index("api-reference");</script>
