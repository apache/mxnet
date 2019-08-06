mxnet.ndarray
==========================

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


Tutorials
-----------------

.. container:: cards


   .. card::
      :title: NDArray Guide
      :link: ../../guide/packages/ndarray/

      The NDArray guide. Start here!



NDArray API of MXNet
-----------------

.. container:: cards

   .. card::
      :title: NDArray
      :link: mxnet.ndarray.NDArray.html

      Imperative tensor operations using the NDArray API.

   .. card::
      :title: Routines
      :link: routines.html

      Manipulating multi-dimensional, fixed-size arrays.

Sparse NDArray API of MXNet
-----------------

.. container:: cards

   .. card::
      :title: CSRNDArray
      :link: mxnet.ndarray.sparse.CSRNDArray.html

      Representing two-dimensional, fixed-size arrays in compressed sparse row format.

   .. card::
      :title: RowSparseNDArray
      :link: mxnet.ndarray.sparse.RowSparseNDArray.html

      Representing multi-dimensional, fixed-size arrays in row sparse format.

   .. card::
      :title: Sparse routines
      :link: sparse_routines.html

      Manipulating sparse arrays.



.. toctree::
   :hidden:
   :maxdepth: 2

   mxnet.ndarray.NDArray
   routines
   mxnet.ndarray.sparse.CSRNDArray
   mxnet.ndarray.sparse.RowSparseNDArray
   sparse_routines
