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

Sparse routines
=====================================

.. currentmodule:: mxnet.ndarray.sparse

Create Arrays
--------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    array
    csr_matrix
    empty
    row_sparse_array
    zeros
    zeros_like


Manipulate
------------

Change shape and type
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    cast_storage


Index
^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    retain
    slice
    where


Math
----

Arithmetic
^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    add_n
    broadcast_add
    broadcast_div
    broadcast_mul
    broadcast_sub
    dot
    elemwise_add
    elemwise_mul
    elemwise_sub
    negative


Trigonometric
^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    arcsin
    arctan
    degrees
    radians
    sin
    tan


Hyperbolic
^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    arcsinh
    arctanh
    sinh
    tanh


Reduce
^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    mean
    norm
    sum


Round
^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    ceil
    fix
    floor
    rint
    round
    trunc


Exponents and logarithms
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    expm1
    log1p


Powers
^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    sqrt
    square


Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    abs
    sign


Neural network
---------------

Updater
^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    adam_update
    adagrad_update
    sgd_mom_update
    sgd_update


More
^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    Embedding
    LinearRegressionOutput
    LogisticRegressionOutput
    make_loss
    stop_gradient