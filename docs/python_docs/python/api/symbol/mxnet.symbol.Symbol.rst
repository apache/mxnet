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

Symbol
======

Composition
------------------

Composite multiple symbols into a new one by an operator.

.. currentmodule:: mxnet.symbol

.. autoclass:: Symbol

.. autosummary::
   :toctree: _autogen

    Symbol.__call__


Arithmetic operations
------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.__add__
    Symbol.__sub__
    Symbol.__rsub__
    Symbol.__neg__
    Symbol.__mul__
    Symbol.__div__
    Symbol.__rdiv__
    Symbol.__mod__
    Symbol.__rmod__
    Symbol.__pow__


Trigonometric functions
------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.sin
    Symbol.cos
    Symbol.tan
    Symbol.arcsin
    Symbol.arccos
    Symbol.arctan
    Symbol.degrees
    Symbol.radians


Hyperbolic functions
------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.sinh
    Symbol.cosh
    Symbol.tanh
    Symbol.arcsinh
    Symbol.arccosh
    Symbol.arctanh


Exponents and logarithms
------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.exp
    Symbol.expm1
    Symbol.log
    Symbol.log10
    Symbol.log2
    Symbol.log1p


Powers
------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.sqrt
    Symbol.rsqrt
    Symbol.cbrt
    Symbol.rcbrt
    Symbol.square


Basic neural network functions
----------------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.relu
    Symbol.sigmoid
    Symbol.softmax
    Symbol.log_softmax


Comparison operators
----------------------


.. autosummary::
   :toctree: _autogen

    Symbol.__lt__
    Symbol.__le__
    Symbol.__gt__
    Symbol.__ge__
    Symbol.__eq__
    Symbol.__ne__


Symbol creation
---------------------


.. autosummary::
   :toctree: _autogen

    Symbol.zeros_like
    Symbol.ones_like
    Symbol.diag


Changing shape and type
---------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.astype
    Symbol.shape_array
    Symbol.size_array
    Symbol.reshape
    Symbol.reshape_like
    Symbol.flatten
    Symbol.expand_dims


Expanding elements
-----------------------


.. autosummary::
   :toctree: _autogen

    Symbol.broadcast_to
    Symbol.broadcast_axes
    Symbol.broadcast_like
    Symbol.tile
    Symbol.pad


Rearranging elements
----------------------


.. autosummary::
   :toctree: _autogen

    Symbol.transpose
    Symbol.swapaxes
    Symbol.flip
    Symbol.depth_to_space
    Symbol.space_to_depth


Reduce functions
---------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.sum
    Symbol.nansum
    Symbol.prod
    Symbol.nanprod
    Symbol.mean
    Symbol.max
    Symbol.min
    Symbol.norm


Rounding
---------------------


.. autosummary::
   :toctree: _autogen

    Symbol.round
    Symbol.rint
    Symbol.fix
    Symbol.floor
    Symbol.ceil
    Symbol.trunc


Sorting and searching
-----------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.sort
    Symbol.argsort
    Symbol.topk
    Symbol.argmax
    Symbol.argmin
    Symbol.argmax_channel


Query information
--------------------


.. autosummary::
   :toctree: _autogen

    Symbol.name
    Symbol.list_arguments
    Symbol.list_outputs
    Symbol.list_auxiliary_states
    Symbol.list_attr
    Symbol.attr
    Symbol.attr_dict


Indexing
-----------------------


.. autosummary::
   :toctree: _autogen

    Symbol.slice
    Symbol.slice_axis
    Symbol.slice_like
    Symbol.take
    Symbol.one_hot
    Symbol.pick


Get internal and output symbol
----------------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.__getitem__
    Symbol.__iter__
    Symbol.get_internals
    Symbol.get_children


Inference type and shape
----------------------------------


.. autosummary::
   :toctree: _autogen

    Symbol.infer_type
    Symbol.infer_shape
    Symbol.infer_shape_partial



Bind
------------------


.. autosummary::
   :toctree: _autogen

    Symbol.bind
    Symbol.simple_bind


Save
------------------


.. autosummary::
   :toctree: _autogen

    Symbol.save
    Symbol.tojson
    Symbol.debug_str


Miscellaneous
-----------------------


.. autosummary::
   :toctree: _autogen

    Symbol.clip
    Symbol.sign
