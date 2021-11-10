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

NPX: NumPy Neural Network Extension
===================================

.. currentmodule:: mxnet.npx

Compatibility
-------------

.. autosummary::
   :toctree: generated/

   set_np
   reset_np

.. code::

   is_np_array
   use_np_array
   is_np_shape
   use_np_shape
   np_array
   np_shape


Devices
---------


.. autosummary::
   :toctree: generated/

   cpu
   cpu_pinned
   gpu
   gpu_memory_info
   current_device
   num_gpus

Nerual networks
-----------------------

.. autosummary::
   :toctree: generated/

   activation
   batch_norm
   convolution
   dropout
   embedding
   fully_connected
   layer_norm
   pooling
   rnn
   leaky_relu
   multibox_detection
   multibox_prior
   multibox_target
   roi_pooling


More operators
------------------

.. autosummary::
   :toctree: generated/

   sigmoid
   relu
   smooth_l1
   softmax
   log_softmax
   topk
   waitall
   load
   save
   one_hot
   pick
   reshape_like
   batch_flatten
   batch_dot
   gamma
   sequence_mask

.. code::

   seed
