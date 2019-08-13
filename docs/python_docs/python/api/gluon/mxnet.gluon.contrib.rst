contrib
=======

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


.. currentmodule:: mxnet.gluon.contrib.data

.. autosummary::
    :nosignatures:
    IntervalSampler
```

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
    :imported-members:

.. automodule:: mxnet.gluon.contrib.nn
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.contrib.cnn
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

.. automodule:: mxnet.gluon.contrib.estimator
    :members:
    :imported-members: