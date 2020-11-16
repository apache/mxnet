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

Array manipulation routines
***************************

.. currentmodule:: mxnet.np

Changing array shape
====================
.. autosummary::
   :toctree: generated/


   reshape
   ravel
   ndarray.flatten

Transpose-like operations
=========================
.. autosummary::
   :toctree: generated/

   swapaxes
   ndarray.T
   transpose
   moveaxis
   rollaxis

Changing number of dimensions
=============================
.. autosummary::
   :toctree: generated/

   expand_dims
   squeeze
   broadcast_to
   broadcast_arrays
   atleast_1d
   atleast_2d
   atleast_3d

Joining arrays
==============
.. autosummary::
   :toctree: generated/

   concatenate
   stack
   dstack
   vstack
   column_stack
   hstack

Splitting arrays
================
.. autosummary::
   :toctree: generated/

   split
   hsplit
   vsplit
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
   fliplr
   flipud
