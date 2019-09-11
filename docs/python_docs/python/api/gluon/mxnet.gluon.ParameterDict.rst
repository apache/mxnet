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

ParameterDict
=============

.. currentmodule:: mxnet.gluon

.. autoclass:: ParameterDict

Load and save parameters
--------------------------

.. autosummary::
   :toctree: _autogen

   ParameterDict.load
   ParameterDict.save

Get a particular parameter
--------------------------

.. autosummary::
   :toctree: _autogen

   ParameterDict.get
   ParameterDict.get_constant

Get (name, paramter) pairs
--------------------------

.. autosummary::
   :toctree: _autogen

   ParameterDict.items
   ParameterDict.keys
   ParameterDict.values

Update parameters
--------------------------

.. autosummary::
   :toctree: _autogen

   ParameterDict.initialize
   ParameterDict.setattr
   ParameterDict.update


Set devices contexts and gradients
---------------------------------------

.. autosummary::
   :toctree: _autogen

   ParameterDict.reset_ctx
   ParameterDict.zero_grad

Attributes
---------------------------------------


.. autosummary::

   ParameterDict.prefix
