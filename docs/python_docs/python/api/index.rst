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

Python API
==========

Overview
--------

This API section details functions, modules, and objects included in MXNet,
describing what they are and what they do. The APIs are grouped into the
following categories:


Imperative API
---------------
.. container:: cards

   .. card::
      :title: mxnet.ndarray
      :link: ndarray/index.html

      Imperative APIs to manipulate multi-dimensional arrays.

   .. card::
      :title: mxnet.gluon
      :link: gluon/index.html

      Imperative APIs to load data, construct and train neural networks.



Gluon related modules
---------------------

.. container:: cards

   .. card::
      :title: mxnet.autograd
      :link: autograd/index.html

      Functions for Automatic differentiation.

   .. card::
      :title: mxnet.optimizer
      :link: optimizer/index.html

      Functions for applying an optimizer on weights.

   .. card::
      :title: mxnet.initializer
      :link: initializer/index.html

      Default behaviors to initialize parameters.

   .. card::
      :title: mxnet.lr_scheduler
      :link: lr_scheduler/index.html

      Scheduling the learning rate.

   .. card::
      :title: mxnet.metric
      :link: metric/index.html

      Metrics to evaluate the performance of a learned model.

   .. card::
      :title: mxnet.kvstore
      :link: kvstore/index.html

      Key value store interface of MXNet for parameter synchronization.

   .. card::
      :title: mxnet.context
      :link: mxnet/context/index.html

      CPU and GPU context information.

   .. card::
      :title: mxnet.profiler
      :link: mxnet/profiler/index.html

      Profiler setting methods.

   .. card::
      :title: mxnet.random
      :link: mxnet/random/index.html

      Imperative random distribution generator functions.


Symbolic API
------------

.. container:: cards

   .. card::
      :title: mxnet.sym
      :link: symbol/index.html

      Symbolic APIs for multi-dimensional arrays and neural network layers

   .. card::
      :title: mxnet.module
      :link: module/index.html

      Intermediate and high-level interface for performing computation with Symbols.


Symbol related modules
----------------------

.. container:: cards

   .. card::
      :title: mxnet.callback
      :link: mxnet/callback/index.html

      Functions to track various statuses during an epoch.

   .. card::
      :title: mxnet.monitor
      :link: mxnet/monitor/index.html

      Outputs, weights, and gradients for debugging


   .. card::
      :title: mxnet.image
      :link: mxnet/image/index.html

      Image iterators and image augmentation functions.

   .. card::
      :title: mxnet.io
      :link: mxnet/io/index.html

      Data iterators for common data formats and utility functions.

   .. card::
      :title: mxnet.recordio
      :link: mxnet/recordio/index.html

      Read and write for the RecordIO data format.

   .. card::
      :title: mxnet.visualization
      :link: mxnet/visualization/index.html

      Functions for Symbol visualization.

Advanced modules
----------------

.. container:: cards

   .. card::
      :title: mxnet.executor
      :link: mxnet/executor/index.html

      Managing symbolic graph execution.

   .. card::
      :title: mxnet.kvstore_server
      :link: mxnet/kvstore_server/index.html

      Server node for the key value store.

   .. card::
      :title: mxnet.engine
      :link: mxnet/engine/index.html

      Engine properties management.


   .. card::
      :title: mxnet.executor_manager
      :link: mxnet/executor_manager/index.html

      Executor manager


   .. card::
      :title: mxnet.rtc
      :link: mxnet/rtc/index.html

      Tools for compiling and running CUDA code from the python frontend.

   .. card::
      :title: mxnet.test_utils
      :link: mxnet/test_utils/index.html

      Tools for using and testing MXNet.

   .. card::
      :title: mxnet.util
      :link: mxnet/util/index.html

      General utility functions


.. toctree::
   :maxdepth: 1
   :hidden:

   ndarray/index
   gluon/index
   autograd/index
   initializer/index
   optimizer/index
   lr_scheduler/index
   metric/index
   kvstore/index
   symbol/index
   module/index
   contrib/index
   image/index
   mxnet/index
