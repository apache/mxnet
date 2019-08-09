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
