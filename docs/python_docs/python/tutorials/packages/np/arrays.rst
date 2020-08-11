.. _arrays:

*************
Array objects
*************

.. currentmodule:: mxnet.np

``np`` provides an N-dimensional array type, the :ref:`ndarray
<arrays.ndarray>`, which describes a collection of "items" of the same
type. The items can be :ref:`indexed <arrays.indexing>` using for
example N integers.

All ndarrays are :term:`homogenous`: every item takes up the same size
block of memory, and all blocks are interpreted in exactly the same
way. How each item in the array is to be interpreted is specified by a
separate :ref:`data-type object <arrays.dtypes>`, one of which is associated
with every array. In addition to basic types (integers, floats,
*etc.*), the data type objects can also represent data structures.

An item extracted from an array, *e.g.*, by indexing, is represented
by a Python object whose type is one of the :ref:`array scalar types
<arrays.scalars>` built in NumPy. The array scalars allow easy manipulation
of also more complicated arrangements of data.

.. note::

   A major difference to ``numpy.ndarray`` is that ``mxnet.np.ndarray``'s scalar
   is a 0-dim ndarray instead of a scalar object (``numpy.generic``).

.. toctree::
   :maxdepth: 2

   arrays.ndarray
   arrays.scalars
   arrays.dtypes
   arrays.indexing
   arrays.nditer
   arrays.classes
   maskedarray
   arrays.interface
   arrays.datetime
