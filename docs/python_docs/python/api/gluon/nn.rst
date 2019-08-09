nn and contrib.nn
=========================

Gluon provides a large number of build-in neural network layers in the following
two modules:

..
   Don't add toctree to these two modules, otherwise it will generate two pages in
   the global TOC

.. autosummary::
    :nosignatures:

    mxnet.gluon.nn
    mxnet.gluon.contrib.nn


We group all layers in these two modules according to their categories.

.. currentmodule:: mxnet.gluon

Blocks
------

.. autosummary::
   :nosignatures:
   :toctree: .

   nn.Block
   nn.HybridBlock
   nn.SymbolBlock


Sequential containers
---------------------


.. autosummary::
    :toctree: _autogen
    :nosignatures:

    nn.Sequential
    nn.HybridSequential

Concurrent containers
---------------------

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    contrib.nn.Concurrent
    contrib.nn.HybridConcurrent


Basic Layers
------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.Dense
    nn.Activation
    nn.Dropout
    nn.Flatten
    nn.Lambda
    nn.HybridLambda

Convolutional Layers
--------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.Conv1D
    nn.Conv2D
    nn.Conv3D
    nn.Conv1DTranspose
    nn.Conv2DTranspose
    nn.Conv3DTranspose

Pooling Layers
--------------

.. autosummary::
   :nosignatures:
   :toctree: _autogen

    nn.MaxPool1D
    nn.MaxPool2D
    nn.MaxPool3D
    nn.AvgPool1D
    nn.AvgPool2D
    nn.AvgPool3D
    nn.GlobalMaxPool1D
    nn.GlobalMaxPool2D
    nn.GlobalMaxPool3D
    nn.GlobalAvgPool1D
    nn.GlobalAvgPool2D
    nn.GlobalAvgPool3D
    nn.ReflectionPad2D

Normalization Layers
--------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.BatchNorm
    nn.InstanceNorm
    nn.LayerNorm
    contrib.nn.SyncBatchNorm

Embedding Layers
----------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.Embedding
    contrib.nn.SparseEmbedding


Advanced Activation Layers
--------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    nn.LeakyReLU
    nn.PReLU
    nn.ELU
    nn.SELU
    nn.Swish
