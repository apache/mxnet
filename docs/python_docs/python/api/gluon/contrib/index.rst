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

.. currentmodule:: mxnet.gluon.contrib

.. autosummary::
    :nosignatures:

    mxnet.gluon.contrib


The `Gluon Contrib` API, defined in the `gluon.contrib` package, provides
many useful experimental APIs for new features.
This is a place for the community to try out the new features,
so that feature contributors can receive feedback.


.. warning:: This package contains experimental APIs and may change in the near future.


In the rest of this document, we list routines provided by the `gluon.contrib` package.

Vision Data
-----------

.. autosummary::
    :nosignatures:

    data.vision.create_image_augment
    data.vision.ImageDataLoader
    data.vision.ImageBboxDataLoader
    data.vision.ImageBboxRandomFlipLeftRight
    data.vision.ImageBboxCrop
    data.vision.ImageBboxRandomCropWithConstraints
    data.vision.ImageBboxResize


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

.. automodule:: mxnet.gluon.contrib.estimator
    :members:
    :imported-members:
