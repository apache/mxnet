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
   
gluon.nn
========

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



Sequential containers
---------------------


.. autosummary::
    :nosignatures:

    nn.Sequential
    nn.HybridSequential


Basic Layers
------------

.. autosummary::
    :nosignatures:

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

    nn.BatchNorm
    nn.InstanceNorm
    nn.LayerNorm

Embedding Layers
----------------

.. autosummary::
    :nosignatures:

    nn.Embedding


Advanced Activation Layers
--------------------------

.. autosummary::
    :nosignatures:

    nn.LeakyReLU
    nn.PReLU
    nn.ELU
    nn.SELU
    nn.Swish

API Reference
-------------
.. automodule:: mxnet.gluon.nn
    :members:
    :imported-members:
    :autosummary:
