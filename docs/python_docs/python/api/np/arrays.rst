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

.. _arrays:

*************
Array objects
*************

.. currentmodule:: mxnet.np

``np`` provides an N-dimensional array type, the :ref:`ndarray
<arrays.ndarray>`, which describes a collection of "items" of the same
type. The items can be :ref:`indexed <arrays.indexing>` using for
example N integers.

All ndarrays are homogenous: every item takes up the same size
block of memory, and all blocks are interpreted in exactly the same
way. How each item in the array is to be interpreted is specified by a
separate data-type object, one of which is associated
with every array. In addition to basic types (integers, floats,
*etc.*), the data type objects can also represent data structures.

An item extracted from an array, *e.g.*, by indexing, is represented
by a Python object whose type is one of the array scalar types
built in NumPy. The array scalars allow easy manipulation
of also more complicated arrangements of data.

.. note::

   A major difference to ``numpy.ndarray`` is that ``mxnet.np.ndarray``'s scalar
   is a 0-dim ndarray instead of a scalar object (``numpy.generic``).

.. toctree::
   :maxdepth: 2

   arrays.ndarray
   arrays.indexing
