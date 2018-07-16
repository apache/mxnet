# Gluon Contrib API

## Overview

This document lists the contrib APIs in Gluon:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.gluon.contrib
```

The `Gluon Contrib` API, defined in the `gluon.contrib` package, provides
many useful experimental APIs for new features.
This is a place for the community to try out the new features,
so that feature contributors can receive feedback.

```eval_rst
.. warning:: This package contains experimental APIs and may change in the near future.
```

In the rest of this document, we list routines provided by the `gluon.contrib` package.

## Contrib

### Neural network

```eval_rst
.. currentmodule:: mxnet.gluon.contrib.nn

.. autosummary::
    :nosignatures:

    Concurrent
    HybridConcurrent
    Identity
    SparseEmbedding
    SyncBatchNorm
```

### Recurrent neural network

```eval_rst
.. currentmodule:: mxnet.gluon.contrib.rnn

.. autosummary::
    :nosignatures:

    VariationalDropoutCell
    Conv1DRNNCell
    Conv2DRNNCell
    Conv3DRNNCell
    Conv1DLSTMCell
    Conv2DLSTMCell
    Conv3DLSTMCell
    Conv1DGRUCell
    Conv2DGRUCell
    Conv3DGRUCell
    LSTMPCell
```

### Data

```eval_rst
.. currentmodule:: mxnet.gluon.contrib.data

.. autosummary::
    :nosignatures:

    IntervalSampler
```

#### Text dataset

```eval_rst
.. currentmodule:: mxnet.gluon.contrib.data.text

.. autosummary::
    :nosignatures:

    WikiText2
    WikiText103
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst

.. automodule:: mxnet.gluon.contrib
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.contrib.nn
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.contrib.rnn
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.contrib.data
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.contrib.data.text
    :members:
    :imported-members:

```

<script>auto_index("api-reference");</script>
