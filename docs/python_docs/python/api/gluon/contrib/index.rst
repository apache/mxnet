.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

gluon.contrib
=============

This document lists the contrib APIs in Gluon:


.. autosummary::
    :nosignatures:

    mxnet.gluon.contrib


The `Gluon Contrib` API, defined in the `gluon.contrib` package, provides
many useful experimental APIs for new features.
This is a place for the community to try out the new features,
so that feature contributors can receive feedback.


.. warning:: This package contains experimental APIs and may change in the near future.


In the rest of this document, we list routines provided by the `gluon.contrib` package.

Neural Network
--------------


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


Convolutional Neural Network
----------------------------

.. currentmodule:: mxnet.gluon.contrib.cnn

.. autosummary::
    :nosignatures:

    DeformableConvolution


Recurrent Neural Network
------------------------


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


Data
----


.. currentmodule:: mxnet.gluon.contrib.data.sampler

.. autosummary::
    :nosignatures:

    IntervalSampler


Text Dataset
------------

.. currentmodule:: mxnet.gluon.contrib.data.text

.. autosummary::
    :nosignatures:

    WikiText2
    WikiText103


Estimator
---------

.. currentmodule:: mxnet.gluon.contrib.estimator

.. autosummary::
    :nosignatures:

    Estimator


Event Handler
-------------

.. currentmodule:: mxnet.gluon.contrib.estimator

.. autosummary::
    :nosignatures:

    StoppingHandler
    MetricHandler
    ValidationHandler
    LoggingHandler
    CheckpointHandler
    EarlyStoppingHandler


API Reference
-------------


.. automodule:: mxnet.gluon.contrib
    :members:

.. automodule:: mxnet.gluon.contrib.nn
    :members:

.. automodule:: mxnet.gluon.contrib.cnn
    :members:

.. automodule:: mxnet.gluon.contrib.rnn
    :members:

.. automodule:: mxnet.gluon.contrib.data.sampler
    :members:

.. automodule:: mxnet.gluon.contrib.data.text
    :members:

.. automodule:: mxnet.gluon.contrib.estimator
    :members:
    :imported-members: