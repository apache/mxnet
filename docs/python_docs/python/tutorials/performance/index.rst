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

Performance
===========
The following tutorials will help you learn how to tune MXNet or use tools that will improve training and inference performance.

Essential
---------

.. container:: cards

   .. card::
      :title: Improving Performance
      :link: https://mxnet.apache.org/api/faq/perf

      How to get the best performance from MXNet.

   .. card::
      :title: Profiler
      :link: backend/profiler.html

      How to profile MXNet models.

   .. card::
      :title: Tuning NumPy Operations
      :link: https://mxnet.apache.org/versions/master/tutorials/gluon/gotchas_numpy_in_mxnet.html

      Gotchas using NumPy in MXNet.

Compression
-----------

.. container:: cards

   .. card::
      :title: Compression: float16
      :link: compression/float16.html

      How to use float16 in your model to boost training speed.

   .. card::
      :title: Gradient Compression
      :link: compression/gradient_compression.html

      How to use gradient compression to reduce communication bandwidth and increase speed.
   ..
      .. card::
         :title: Compression: int8
         :link: compression/int8.html

         How to use int8 in your model to boost training speed.
   ..


Accelerated Backend
-------------------

.. container:: cards

   .. card::
      :title: TensorRT
      :link: backend/tensorRt.html

      How to use NVIDIA's TensorRT to boost inference performance.

   ..
      TBD Content
      .. card::
         :title: MKL-DNN
         :link: backend/mkl-dnn.html

         How to get the most from your CPU by using Intel's MKL-DNN.

      .. card::
         :title: TVM
         :link: backend/tvm.html

         How to use TVM to boost performance.
   ..

Distributed Training
--------------------

.. container:: cards

   .. card::
      :title: Distributed Training Using the KVStore API
      :link: https://mxnet.apache.org/versions/master/faq/distributed_training.html

      How to use the KVStore API to use multiple GPUs when training a model.

   .. card::
      :title: Training with Multiple GPUs Using Model Parallelism
      :link: https://mxnet.apache.org/versions/master/faq/model_parallel_lstm.html

      An overview of using multiple GPUs when training an LSTM.

   .. card::
      :title: Data Parallelism in MXNet
      :link: https://mxnet.apache.org/versions/master/faq/multi_devices.html

      An overview of distributed training strategies.

   .. card::
      :title: MXNet with Horovod
      :link: https://github.com/apache/incubator-mxnet/tree/master/example/distributed_training-horovod

      A set of example scripts demonstrating MNIST and ImageNet training with Horovod as the distributed training backend.

.. toctree::
   :hidden:
   :maxdepth: 1

   compression/index
   backend/index
