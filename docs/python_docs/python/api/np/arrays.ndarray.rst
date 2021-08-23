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

.. _arrays.ndarray:

******************************************
The N-dimensional array (:class:`ndarray`)
******************************************

.. currentmodule:: mxnet.np

An :class:`ndarray` is a (usually fixed-size) multidimensional
container of items of the same type and size. The number of dimensions
and items in an array is defined by its :attr:`shape <ndarray.shape>`,
which is a :class:`tuple` of *N* non-negative integers that specify the
sizes of each dimension. The type of items in the array is specified by
a separate data-type object (dtype), one of which
is associated with each ndarray.

As with other container objects in Python, the contents of an
:class:`ndarray` can be accessed and modified by :ref:`indexing or
slicing <arrays.indexing>` the array (using, for example, *N* integers),
and via the methods and attributes of the :class:`ndarray`.

.. index:: view, base

Different :class:`ndarrays <ndarray>` can share the same data, so that
changes made in one :class:`ndarray` may be visible in another. That
is, an ndarray can be a *"view"* to another ndarray, and the data it
is referring to is taken care of by the *"base"* ndarray.


.. admonition:: Example

   A 2-dimensional array of size 2 x 3, composed of 4-byte integer
   elements:

   >>> x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
   >>> type(x)
   <class 'mxnet.numpy.ndarray'>
   >>> x.shape
   (2, 3)
   >>> x.dtype
   dtype('int32')

   The array can be indexed using Python container-like syntax:

   >>> # The element of x in the *second* row, *third* column, namely, 6.
   >>> x[1, 2]
   array(6, dtype=int32)  # this is different than the official NumPy which returns a np.int32 object

   For example :ref:`slicing <arrays.indexing>` can produce views of
   the array if the elements to be sliced is continguous in memory:

   >>> y = x[1,:]
   >>> y
   array([9, 5, 6], dtype=int32)  # this also changes the corresponding element in x
   >>> x
   array([[1, 2, 3],
           [9, 5, 6]], dtype=int32)


Constructing arrays
===================

New arrays can be constructed using the routines detailed in
:ref:`routines.array-creation`, and also by using the low-level
:class:`ndarray` constructor:

.. autosummary::

   ndarray


Indexing arrays
===============

Arrays can be indexed using an extended Python slicing syntax,
``array[selection]``.

.. seealso:: :ref:`Array Indexing <arrays.indexing>`.

.. _memory-layout:

Internal memory layout of an ndarray
====================================

An instance of class :class:`ndarray` consists of a contiguous
one-dimensional segment of computer memory (owned by the array, or by
some other object), combined with an indexing scheme that maps *N*
integers into the location of an item in the block.  The ranges in
which the indices can vary is specified by the :obj:`shape
<ndarray.shape>` of the array. How many bytes each item takes and how
the bytes are interpreted is defined by the data-type object
associated with the array.

.. index:: C-order, Fortran-order, row-major, column-major, stride,
  offset

.. note::

    `mxnet.numpy.ndarray` currently only supports storing elements in
    C-order/row-major and contiguous memory space. The following content
    on explaining a variety of memory layouts of an ndarray
    are copied from the official NumPy documentation as a comprehensive reference.

A segment of memory is inherently 1-dimensional, and there are many
different schemes for arranging the items of an *N*-dimensional array
in a 1-dimensional block. NumPy is flexible, and :class:`ndarray`
objects can accommodate any *strided indexing scheme*. In a strided
scheme, the N-dimensional index :math:`(n_0, n_1, ..., n_{N-1})`
corresponds to the offset (in bytes):

.. math:: n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k

from the beginning of the memory block associated with the
array. Here, :math:`s_k` are integers which specify the :obj:`strides
<ndarray.strides>` of the array. The column-major order (used,
for example, in the Fortran language and in *Matlab*) and
row-major order (used in C) schemes are just specific kinds of
strided scheme, and correspond to memory that can be *addressed* by the strides:

.. math::

   s_k^{\mathrm{column}} = \mathrm{itemsize} \prod_{j=0}^{k-1} d_j ,
   \quad  s_k^{\mathrm{row}} = \mathrm{itemsize} \prod_{j=k+1}^{N-1} d_j .

.. index:: single-segment, contiguous, non-contiguous

where :math:`d_j` `= self.shape[j]`.

Both the C and Fortran orders are contiguous, *i.e.,*
single-segment, memory layouts, in which every part of the
memory block can be accessed by some combination of the indices.

While a C-style and Fortran-style contiguous array, which has the corresponding
flags set, can be addressed with the above strides, the actual strides may be
different. This can happen in two cases:

    1. If ``self.shape[k] == 1`` then for any legal index ``index[k] == 0``.
       This means that in the formula for the offset :math:`n_k = 0` and thus
       :math:`s_k n_k = 0` and the value of :math:`s_k` `= self.strides[k]` is
       arbitrary.
    2. If an array has no elements (``self.size == 0``) there is no legal
       index and the strides are never used. Any array with no elements may be
       considered C-style and Fortran-style contiguous.

Point 1. means that ``self`` and ``self.squeeze()`` always have the same
contiguity and ``aligned`` flags value. This also means
that even a high dimensional array could be C-style and Fortran-style
contiguous at the same time.

.. index:: aligned

An array is considered aligned if the memory offsets for all elements and the
base offset itself is a multiple of `self.itemsize`. Understanding
`memory-alignment` leads to better performance on most hardware.

.. note::

    Points (1) and (2) are not yet applied by default. Beginning with
    NumPy 1.8.0, they are applied consistently only if the environment
    variable ``NPY_RELAXED_STRIDES_CHECKING=1`` was defined when NumPy
    was built. Eventually this will become the default.

    You can check whether this option was enabled when your NumPy was
    built by looking at the value of ``np.ones((10,1),
    order='C').flags.f_contiguous``. If this is ``True``, then your
    NumPy has relaxed strides checking enabled.

.. warning::

    It does *not* generally hold that ``self.strides[-1] == self.itemsize``
    for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
    Fortran-style contiguous arrays is true.

Data in new :class:`ndarrays <ndarray>` is in the row-major
(C) order, unless otherwise specified, but, for example, :ref:`basic
array slicing <arrays.indexing>` often produces views
in a different scheme.

.. seealso: :ref:`Indexing <arrays.ndarray.indexing>`_

.. note::

   Several algorithms in NumPy work on arbitrarily strided arrays.
   However, some algorithms require single-segment arrays. When an
   irregularly strided array is passed in to such algorithms, a copy
   is automatically made.

.. _arrays.ndarray.attributes:

Array attributes
================

Array attributes reflect information that is intrinsic to the array
itself. Generally, accessing an array through its attributes allows
you to get and sometimes set intrinsic properties of the array without
creating a new array. The exposed attributes are the core parts of an
array and only some of them can be reset meaningfully without creating
a new array. Information on each attribute is given below.

Memory layout
-------------

The following attributes contain information about the memory layout
of the array:

.. autosummary::

   ndarray.shape
   ndarray.ndim
   ndarray.size

Data type
---------

The data type object associated with the array can be found in the
:attr:`dtype <ndarray.dtype>` attribute:

.. autosummary::

   ndarray.dtype

.. _array.ndarray.methods:

Array methods
=============

An :class:`ndarray` object has many methods which operate on or with
the array in some fashion, typically returning an array result. These
methods are briefly explained below. (Each method's docstring has a
more complete description.)

For the following methods there are also corresponding functions in
:mod:`numpy`: :func:`all`, :func:`any`, :func:`argmax`,
:func:`argmin`, :func:`argpartition`, :func:`argsort`, :func:`choose`,
:func:`clip`, :func:`compress`, :func:`copy`, :func:`cumprod`,
:func:`cumsum`, :func:`diagonal`, :func:`imag`, :func:`max <amax>`,
:func:`mean`, :func:`min <amin>`, :func:`nonzero`, :func:`partition`,
:func:`prod`, :func:`ptp`, :func:`put`, :func:`ravel`, :func:`real`,
:func:`repeat`, :func:`reshape`, :func:`round <around>`,
:func:`searchsorted`, :func:`sort`, :func:`squeeze`, :func:`std`,
:func:`sum`, :func:`swapaxes`, :func:`take`, :func:`trace`,
:func:`transpose`, :func:`var`.

Array conversion
----------------

.. autosummary::

   ndarray.item
   ndarray.copy
   ndarray.tolist
   ndarray.astype


Shape manipulation
------------------

For reshape, resize, and transpose, the single tuple argument may be
replaced with ``n`` integers which will be interpreted as an n-tuple.

.. autosummary::

   ndarray.reshape
   ndarray.transpose
   ndarray.swapaxes
   ndarray.flatten
   ndarray.squeeze

Item selection and manipulation
-------------------------------

For array methods that take an *axis* keyword, it defaults to
:const:`None`. If axis is *None*, then the array is treated as a 1-D
array. Any other value for *axis* represents the dimension along which
the operation should proceed.

.. autosummary::

   ndarray.nonzero
   ndarray.take
   ndarray.repeat
   ndarray.argsort
   ndarray.sort

Calculation
-----------

.. index:: axis

Many of these methods take an argument named *axis*. In such cases,

- If *axis* is *None* (the default), the array is treated as a 1-D
  array and the operation is performed over the entire array. This
  behavior is also the default if self is a 0-dimensional array or
  array scalar. (An array scalar is an instance of the types/classes
  float32, float64, etc., whereas a 0-dimensional array is an ndarray
  instance containing precisely one array scalar.)

- If *axis* is an integer, then the operation is done over the given
  axis (for each 1-D subarray that can be created along the given axis).

.. admonition:: Example of the *axis* argument

   A 3-dimensional array of size 3 x 3 x 3, summed over each of its
   three axes

   >>> x
   array([[[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8]],
          [[ 9, 10, 11],
           [12, 13, 14],
           [15, 16, 17]],
          [[18, 19, 20],
           [21, 22, 23],
           [24, 25, 26]]])
   >>> x.sum(axis=0)
   array([[27, 30, 33],
          [36, 39, 42],
          [45, 48, 51]])
   >>> # for sum, axis is the first keyword, so we may omit it,
   >>> # specifying only its value
   >>> x.sum(0), x.sum(1), x.sum(2)
   (array([[27, 30, 33],
           [36, 39, 42],
           [45, 48, 51]]),
    array([[ 9, 12, 15],
           [36, 39, 42],
           [63, 66, 69]]),
    array([[ 3, 12, 21],
           [30, 39, 48],
           [57, 66, 75]]))

The parameter *dtype* specifies the data type over which a reduction
operation (like summing) should take place. The default reduce data
type is the same as the data type of *self*. To avoid overflow, it can
be useful to perform the reduction using a larger data type.

For several methods, an optional *out* argument can also be provided
and the result will be placed into the output array given. The *out*
argument must be an :class:`ndarray` and have the same number of
elements. It can have a different data type in which case casting will
be performed.

.. autosummary::

   ndarray.max
   ndarray.argmax
   ndarray.min
   ndarray.argmin
   ndarray.clip
   ndarray.sum
   ndarray.mean
   ndarray.prod
   ndarray.cumsum
   ndarray.var
   ndarray.std
   ndarray.round
   ndarray.all
   ndarray.any

Arithmetic, matrix multiplication, and comparison operations
============================================================

.. index:: comparison, arithmetic, matrix, operation, operator

Arithmetic and comparison operations on :class:`ndarrays <ndarray>`
are defined as element-wise operations, and generally yield
:class:`ndarray` objects as results.

Each of the arithmetic operations (``+``, ``-``, ``*``, ``/``, ``//``,
``%``, ``divmod()``, ``**`` or ``pow()``, ``<<``, ``>>``, ``&``,
``^``, ``|``, ``~``) and the comparisons (``==``, ``<``, ``>``,
``<=``, ``>=``, ``!=``) is equivalent to the corresponding
universal function (or ufunc for short) in NumPy.

Comparison operators:

.. autosummary::

   ndarray.__lt__
   ndarray.__le__
   ndarray.__gt__
   ndarray.__ge__
   ndarray.__eq__
   ndarray.__ne__

Truth value of an array (:func:`bool()`):

.. autosummary::

   ndarray.__bool__

.. note::

   Truth-value testing of an array invokes
   :meth:`ndarray.__bool__`, which raises an error if the number of
   elements in the array is larger than 1, because the truth value
   of such arrays is ambiguous.


Unary operations:

.. autosummary::

   ndarray.__neg__
   ndarray.__abs__
   ndarray.__invert__

Arithmetic:

.. autosummary::

   ndarray.__add__
   ndarray.__sub__
   ndarray.__mul__
   ndarray.__truediv__
   ndarray.__mod__
   ndarray.__pow__
   ndarray.__and__
   ndarray.__or__
   ndarray.__xor__

.. note::

   - Any third argument to :func:`pow()` is silently ignored,
     as the underlying :func:`ufunc <power>` takes only two arguments.

   - The three division operators are all defined; :obj:`div` is active
     by default, :obj:`truediv` is active when
     :obj:`__future__` division is in effect.

   - Because :class:`ndarray` is a built-in type (written in C), the
     ``__r{op}__`` special methods are not directly defined.

   - The functions called to implement many arithmetic special methods
     for arrays can be modified using :class:`__array_ufunc__ <numpy.class.__array_ufunc__>`.

Arithmetic, in-place:

.. autosummary::

   ndarray.__iadd__
   ndarray.__isub__
   ndarray.__imul__
   ndarray.__itruediv__
   ndarray.__imod__
   ndarray.__iand__
   ndarray.__ior__
   ndarray.__ixor__


.. warning::

   In place operations will perform the calculation using the
   precision decided by the data type of the two operands, but will
   silently downcast the result (if necessary) so it can fit back into
   the array.  Therefore, for mixed precision calculations,
   ``A {op}= B`` can be different than ``A = A {op} B``. For example, suppose
   ``a = ones((3,3))``. Then, ``a += 3j`` is different than ``a = a + 3j``:
   while they both perform the same computation, ``a += 3``
   casts the result to fit back in ``a``, whereas ``a = a + 3j``
   re-binds the name ``a`` to the result.


Matrix Multiplication:


.. autosummary::

   ndarray.__matmul__


Special methods
===============

For standard library functions:

.. autosummary::

   ndarray.__reduce__
   ndarray.__setstate__

Basic customization:

.. autosummary::

   ndarray.__new__

Container customization: (see :ref:`Indexing <arrays.indexing>`)

.. autosummary::

   ndarray.__len__
   ndarray.__getitem__
   ndarray.__setitem__

Conversion; the operations :func:`int()` and :func:`float()`.
They work only on arrays that have one element in them
and return the appropriate scalar.

.. autosummary::

   ndarray.__int__
   ndarray.__float__

String representations:

.. autosummary::

   ndarray.__str__
   ndarray.__repr__
