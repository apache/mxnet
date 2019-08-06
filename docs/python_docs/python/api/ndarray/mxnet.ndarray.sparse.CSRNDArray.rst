CSRNDArray
====================

.. currentmodule:: mxnet.ndarray.sparse

.. autoclass:: CSRNDArray


Array attributes
---------------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.context
    CSRNDArray.data
    CSRNDArray.dtype
    CSRNDArray.indices
    CSRNDArray.indptr
    CSRNDArray.shape
    CSRNDArray.stype


Array creation
--------------

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.zeros_like


Manipulation
-------------

Array conversion
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.as_in_context
    CSRNDArray.asnumpy
    CSRNDArray.asscalar
    CSRNDArray.asscipy
    CSRNDArray.astype
    CSRNDArray.copy
    CSRNDArray.copyto
    CSRNDArray.tostype


Array inspection
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.check_format


Array reduction
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.mean
    CSRNDArray.norm
    CSRNDArray.sum


Indexing
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.__getitem__
    CSRNDArray.__setitem__
    CSRNDArray.slice


Joining arrays
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    concat


Lazy evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.wait_to_read


Math
----

Array rounding
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.ceil
    CSRNDArray.fix
    CSRNDArray.floor
    CSRNDArray.round
    CSRNDArray.rint
    CSRNDArray.trunc


Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.arcsin
    CSRNDArray.arctan
    CSRNDArray.degrees
    CSRNDArray.radians
    CSRNDArray.sin
    CSRNDArray.tan


Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.arcsinh
    CSRNDArray.arctanh
    CSRNDArray.sinh
    CSRNDArray.tanh


Exponents and logarithms
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.expm1
    CSRNDArray.log1p


Powers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.sqrt
    CSRNDArray.square


Miscellaneous
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :toctree: _autogen

    CSRNDArray.abs
    CSRNDArray.clip
    CSRNDArray.sign
