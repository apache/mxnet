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

mxnet.ndarray
=============

The NDArray library in Apache MXNet defines the core data structure for all mathematical computations. NDArray supports fast execution on a wide range of hardware configurations and automatically parallelizes multiple operations across the available hardware.

Example
-------

The following example shows how you can create an NDArray from a regular Python list using the 'array' function. 

.. code-block:: python

	import mxnet as mx
	# create a 1-dimensional array with a python list
	a = mx.nd.array([1,2,3])
	# create a 2-dimensional array with a nested python list
	b = mx.nd.array([[1,2,3], [2,3,4]])
	{'a.shape':a.shape, 'b.shape':b.shape}


.. note:: ``mxnet.ndarray`` is similar to ``numpy.ndarray`` in some aspects. But the differences are not negligible. For instance:

   - ``mxnet.ndarray.NDArray.T`` does real data transpose to return new a copied
     array, instead of returning a view of the input array.
   - ``mxnet.ndarray.dot`` performs dot product between the last axis of the
     first input array and the first axis of the second input, while `numpy.dot`
     uses the second last axis of the input array.

   In addition, ``mxnet.ndarray.NDArray`` supports GPU computation and various neural
   network layers.

.. note:: ``ndarray`` provides almost the same routines as ``symbol``. Most
  routines between these two packages share the source code. But ``ndarray``
  differs from ``symbol`` in few aspects:

  - ``ndarray`` adopts imperative programming, namely sentences are executed
    step-by-step so that the results can be obtained immediately whereas
    ``symbol`` adopts declarative programming.

  - Most binary operators in ``ndarray`` such as ``+`` and ``>`` have
    broadcasting enabled by default.

Tutorials
---------

.. container:: cards


   .. card::
      :title: NDArray Guide
      :link: ../../tutorials/packages/ndarray/

      The NDArray guide. Start here!



NDArray API of MXNet
--------------------

.. container:: cards

   .. card::
      :title: NDArray
      :link: ndarray.html

      Imperative tensor operations using the NDArray API.


Sparse NDArray API of MXNet
---------------------------

.. container:: cards

   .. card::
      :title: Sparse routines
      :link: sparse/index.html

      Representing and manipulating sparse arrays.



.. toctree::
   :hidden:
   :maxdepth: 2
   :glob:

   ndarray
   */index
