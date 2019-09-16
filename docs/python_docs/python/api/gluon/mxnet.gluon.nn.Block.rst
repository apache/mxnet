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

Block
=====

.. currentmodule:: mxnet.gluon.nn

.. autoclass:: Block
   :members:
   :inherited-members:

   ..
      .. automethod:: __init__


   .. rubric:: Handle model parameters:

   .. autosummary::
      :toctree: _autogen

      Block.initialize
      Block.save_parameters
      Block.load_parameters
      Block.collect_params
      Block.cast
      Block.apply

   .. rubric:: Run computation

   .. autosummary::
      :toctree: _autogen

      Block.forward

   .. rubric:: Debugging

   .. autosummary::
      :toctree: _autogen

      Block.summary

   .. rubric:: Advanced API for customization


   .. autosummary::
      :toctree: _autogen

      Block.name_scope
      Block.register_child
      Block.register_forward_hook
      Block.register_forward_pre_hook

   .. rubric:: Attributes

   .. autosummary::

      Block.name
      Block.params
      Block.prefix


   .. warning::

      The following two APIs are deprecated since `v1.2.1
      <https://github.com/apache/incubator-mxnet/releases/tag/1.2.1>`_.

      .. autosummary::
          :toctree: _autogen

          Block.save_params
          Block.load_params
