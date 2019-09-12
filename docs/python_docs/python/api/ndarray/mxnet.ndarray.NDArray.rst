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

NDArray
=================

.. currentmodule:: mxnet.ndarray

.. autoclass:: NDArray


Attributes
----------

.. autosummary::
   :toctree: _autogen

   NDArray.context
   NDArray.dtype
   NDArray.grad
   NDArray.handle
   NDArray.ndim
   NDArray.shape
   NDArray.size
   NDArray.stype
   NDArray.writable


Array creation
--------------

.. autosummary::
   :toctree: _autogen

   NDArray.ones_like
   NDArray.zeros_like


Manipulation
-------------

Array conversion
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.as_in_context
   NDArray.asnumpy
   NDArray.asscalar
   NDArray.astype
   NDArray.copy
   NDArray.copyto
   NDArray.tostype


Changing shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.diag
   NDArray.expand_dims
   NDArray.flatten
   NDArray.reshape
   NDArray.reshape_like
   NDArray.shape_array
   NDArray.size_array
   NDArray.split


Expanding elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.broadcast_to
   NDArray.broadcast_axes
   NDArray.broadcast_like
   NDArray.pad
   NDArray.repeat
   NDArray.tile


Rearrange elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.depth_to_space
   NDArray.flip
   NDArray.swapaxes
   NDArray.space_to_depth
   NDArray.T
   NDArray.transpose


Sorting and searching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.argmax
   NDArray.argmax_channel
   NDArray.argmin
   NDArray.argsort
   NDArray.sort
   NDArray.topk


Indexing
^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__getitem__
   NDArray.__setitem__
   NDArray.one_hot
   NDArray.pick
   NDArray.slice
   NDArray.slice_axis
   NDArray.slice_like
   NDArray.take


Lazy evaluation
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.wait_to_read


Math
----

Arithmetic operations
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__add__
   NDArray.__div__
   NDArray.__neg__
   NDArray.__mod__
   NDArray.__mul__
   NDArray.__pow__
   NDArray.__rdiv__
   NDArray.__rmod__
   NDArray.__rsub__
   NDArray.__sub__


Rounding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.ceil
   NDArray.fix
   NDArray.floor
   NDArray.rint
   NDArray.round
   NDArray.trunc


Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.max
   NDArray.mean
   NDArray.min
   NDArray.nanprod
   NDArray.nansum
   NDArray.norm
   NDArray.prod
   NDArray.sum


In^place arithmetic operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__iadd__
   NDArray.__idiv__
   NDArray.__imod__
   NDArray.__imul__
   NDArray.__isub__


Comparison operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.__eq__
   NDArray.__ge__
   NDArray.__gt__
   NDArray.__le__
   NDArray.__lt__
   NDArray.__ne__


Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.arccos
   NDArray.arcsin
   NDArray.arctan
   NDArray.cos
   NDArray.degrees
   NDArray.radians
   NDArray.sin
   NDArray.tan


Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.arccosh
   NDArray.arcsinh
   NDArray.arctanh
   NDArray.cosh
   NDArray.sinh
   NDArray.tanh


Exponents and logarithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.exp
   NDArray.expm1
   NDArray.log
   NDArray.log1p
   NDArray.log10
   NDArray.log2


Powers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.cbrt
   NDArray.rcbrt
   NDArray.reciprocal
   NDArray.rsqrt
   NDArray.square
   NDArray.sqrt


Miscellaneous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autogen

   NDArray.clip
   NDArray.sign


Neural network
------------------

.. autosummary::
   :toctree: _autogen

   NDArray.log_softmax
   NDArray.relu
   NDArray.sigmoid
   NDArray.softmax

