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
    PixelShuffle1D
    PixelShuffle2D
    PixelShuffle3D
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
