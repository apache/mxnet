
NDArray
=======




.. class:: NDArray

   Wrapper of the ``NDArray`` type in ``libmxnet``. This is the basic building block
   of tensor-based computation.

   .. _ndarray-shape-note:

   .. note::

      since C/C++ use row-major ordering for arrays while Julia follows a
      column-major ordering. To keep things consistent, we keep the underlying data
      in their original layout, but use *language-native* convention when we talk
      about shapes. For example, a mini-batch of 100 MNIST images is a tensor of
      C/C++/Python shape (100,1,28,28), while in Julia, the same piece of memory
      have shape (28,28,1,100).




.. function:: context(arr :: NDArray)

   Get the context that this :class:`NDArray` lives on.




.. function::
   empty(shape :: Tuple, ctx :: Context)
   empty(shape :: Tuple)
   empty(dim1, dim2, ...)

   Allocate memory for an uninitialized :class:`NDArray` with specific shape.




Interface functions similar to Julia Arrays
-------------------------------------------




.. function::
   zeros(shape :: Tuple, ctx :: Context)
   zeros(shape :: Tuple)
   zeros(dim1, dim2, ...)

   Create zero-ed :class:`NDArray` with specific shape.




.. function::
   ones(shape :: Tuple, ctx :: Context)
   ones(shape :: Tuple)
   ones(dim1, dim2, ...)

   Create an :class:`NDArray` with specific shape and initialize with 1.




.. function::
   size(arr :: NDArray)
   size(arr :: NDArray, dim :: Int)

   Get the shape of an :class:`NDArray`. The shape is in Julia's column-major convention. See
   also the :ref:`notes on NDArray shapes <ndarray-shape-note>`.




.. function:: length(arr :: NDArray)

   Get the number of elements in an :class:`NDArray`.




.. function:: ndims(arr :: NDArray)

   Get the number of dimensions of an :class:`NDArray`. Is equivalent to ``length(size(arr))``.




.. function:: eltype(arr :: NDArray)

   Get the element type of an :class:`NDArray`. Currently the element type is always ``mx.MX_float``.



