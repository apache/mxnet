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

Parameter
=========

.. currentmodule:: mxnet.gluon

.. autoclass:: Parameter


Get and set parameters
-----------------------

.. autosummary::
   :toctree: _autogen

   Parameter.initialize
   Parameter.data
   Parameter.list_data
   Parameter.list_row_sparse_data
   Parameter.row_sparse_data
   Parameter.set_data
   Parameter.shape


Get and set gradients associated with parameters
-------------------------------------------------

.. autosummary::
   :toctree: _autogen

   Parameter.grad
   Parameter.list_grad
   Parameter.zero_grad
   Parameter.grad_req

Handle device contexts
------------------------

.. autosummary::
   :toctree: _autogen

   Parameter.cast
   Parameter.list_ctx
   Parameter.reset_ctx

Convert to symbol
--------------------

.. autosummary::
   :toctree: _autogen

   Parameter.var
