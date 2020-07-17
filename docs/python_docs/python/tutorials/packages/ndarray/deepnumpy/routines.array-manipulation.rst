Array manipulation routines
***************************

.. currentmodule:: mxnet.np

Basic operations
================
.. autosummary::
   :toctree: generated/

::

    copyto

Changing array shape
====================
.. autosummary::
   :toctree: generated/


   reshape
   ravel
   ndarray.flatten

::

   ndarray.flat

Transpose-like operations
=========================
.. autosummary::
   :toctree: generated/

   swapaxes
   ndarray.T
   transpose
   moveaxis

::

   rollaxis

Changing number of dimensions
=============================
.. autosummary::
   :toctree: generated/

   expand_dims
   squeeze
   broadcast_to
   broadcast_arrays

::

   atleast_1d
   atleast_2d
   atleast_3d
   broadcast

Changing kind of array
======================
.. autosummary::
   :toctree: generated/

::

   asarray
   asanyarray
   asmatrix
   asfarray
   asfortranarray
   ascontiguousarray
   asarray_chkfinite
   asscalar
   require

Joining arrays
==============
.. autosummary::
   :toctree: generated/

   concatenate
   stack
   dstack
   vstack

::

   column_stack
   hstack
   block

Splitting arrays
================
.. autosummary::
   :toctree: generated/

   split
   hsplit
   vsplit

::

   array_split
   dsplit

Tiling arrays
=============
.. autosummary::
   :toctree: generated/

   tile
   repeat

Adding and removing elements
============================
.. autosummary::
   :toctree: generated/

   unique

::

   delete
   insert
   append
   resize
   trim_zeros

Rearranging elements
====================
.. autosummary::
   :toctree: generated/

   reshape
   flip
   roll
   rot90

::

   fliplr
   flipud
