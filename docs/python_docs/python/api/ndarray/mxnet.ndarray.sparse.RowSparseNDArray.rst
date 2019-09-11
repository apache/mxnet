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

RowSparseNDArray
==========================

.. currentmodule:: mxnet.ndarray.sparse

.. autoclass:: RowSparseNDArray


Array attributes
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.context
    RowSparseNDArray.data
    RowSparseNDArray.dtype
    RowSparseNDArray.indices
    RowSparseNDArray.shape
    RowSparseNDArray.stype


Array conversion
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.as_in_context
    RowSparseNDArray.asnumpy
    RowSparseNDArray.asscalar
    RowSparseNDArray.astype
    RowSparseNDArray.copy
    RowSparseNDArray.copyto
    RowSparseNDArray.tostype


Array inspection
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.check_format


Array creation
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.zeros_like


Array reduction
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.norm


Array rounding
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.ceil
    RowSparseNDArray.fix
    RowSparseNDArray.floor
    RowSparseNDArray.rint
    RowSparseNDArray.round
    RowSparseNDArray.trunc


Trigonometric functions
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.arcsin
    RowSparseNDArray.arctan
    RowSparseNDArray.degrees
    RowSparseNDArray.radians
    RowSparseNDArray.sin
    RowSparseNDArray.tan


Hyperbolic functions
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.arcsinh
    RowSparseNDArray.arctanh
    RowSparseNDArray.sinh
    RowSparseNDArray.tanh


Exponents and logarithms
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.expm1
    RowSparseNDArray.log1p


Powers
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.sqrt
    RowSparseNDArray.square


Indexing
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.__getitem__
    RowSparseNDArray.__setitem__
    RowSparseNDArray.retain


Lazy evaluation
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.wait_to_read


Miscellaneous
-------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    RowSparseNDArray.abs
    RowSparseNDArray.clip
    RowSparseNDArray.sign
