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

Routines
========

In this chapter routine docstrings are presented, grouped by functionality.
Many docstrings contain example code, which demonstrates basic usage
of the routine. The examples assume that the `np` module is imported with::

  >>> from mxnet import np, npx
  >>> npx.set_np()

A convenient way to execute examples is the ``%doctest_mode`` mode of
IPython, which allows for pasting of multi-line examples and preserves
indentation.

.. toctree::
   :maxdepth: 2

   routines.array-creation
   routines.array-manipulation
   routines.io
   routines.linalg
   routines.math
   random/index
   routines.sort
   routines.statistics
