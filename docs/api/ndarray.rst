
NDArray API
===========




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




.. function:: slice(arr :: NDArray, start:stop)

   Create a view into a sub-slice of an :class:`NDArray`. Note only slicing at the slowest
   changing dimension is supported. In Julia's column-major perspective, this is the last
   dimension. For example, given an :class:`NDArray` of shape (2,3,4), ``slice(array, 2:3)`` will create
   a :class:`NDArray` of shape (2,3,2), sharing the data with the original array. This operation is
   used in data parallelization to split mini-batch into sub-batches for different devices.




.. function:: setindex!(arr :: NDArray, val, idx)

   Assign values to an :class:`NDArray`. Elementwise assignment is not implemented, only the following
   scenarios are supported

   - ``arr[:] = val``: whole array assignment, ``val`` could be a scalar or an array (Julia ``Array``
     or :class:`NDArray`) of the same shape.
   - ``arr[start:stop] = val``: assignment to a *slice*, ``val`` could be a scalar or an array of
     the same shape to the slice. See also :func:`slice`.




.. function:: getindex(arr :: NDArray, idx)

   Shortcut for :func:`slice`. A typical use is to write

   .. code-block:: julia

      arr[:] += 5

   which translates into

   .. code-block:: julia

      arr[:] = arr[:] + 5

   which furthur translates into

   .. code-block:: julia

      setindex!(getindex(arr, Colon()), 5, Colon())

   .. note::

      The behavior is quite different from indexing into Julia's ``Array``. For example, ``arr[2:5]``
      create a **copy** of the sub-array for Julia ``Array``, while for :class:`NDArray`, this is
      a *slice* that shares the memory.




Copying functions
-----------------




.. function::
   copy!(dst :: Union{NDArray, Array}, src :: Union{NDArray, Array})

   Copy contents of ``src`` into ``dst``.




.. function::
   copy(arr :: NDArray)
   copy(arr :: NDArray, ctx :: Context)
   copy(arr :: Array, ctx :: Context)

   Create a copy of an array. When no :class:`Context` is given, create a Julia ``Array``.
   Otherwise, create an :class:`NDArray` on the specified context.




.. function:: convert(::Type{Array{T}}, arr :: NDArray)

   Convert an :class:`NDArray` into a Julia ``Array`` of specific type. Data will be copied.




Basic arithmetics
-----------------




.. function:: @inplace

   Julia does not support re-definiton of ``+=`` operator (like ``__iadd__`` in python),
   When one write ``a += b``, it gets translated to ``a = a+b``. ``a+b`` will allocate new
   memory for the results, and the newly allocated :class:`NDArray` object is then assigned
   back to a, while the original contents in a is discarded. This is very inefficient
   when we want to do inplace update.

   This macro is a simple utility to implement this behavior. Write

   .. code-block:: julia

      @mx.inplace a += b

   will translate into

   .. code-block:: julia

      mx.add_to!(a, b)

   which will do inplace adding of the contents of ``b`` into ``a``.




.. function:: add_to!(dst :: NDArray, args :: Union{Real, NDArray}...)

   Add a bunch of arguments into ``dst``. Inplace updating.




.. function::
   +(args...)
   .+(args...)

   Summation. Multiple arguments of either scalar or :class:`NDArray` could be
   added together. Note at least the first or second argument needs to be an :class:`NDArray` to
   avoid ambiguity of built-in summation.




.. function:: sub_from!(dst :: NDArray, args :: Union{Real, NDArray}...)

   Subtract a bunch of arguments from ``dst``. Inplace updating.




.. function::
   -(arg0, arg1)
   -(arg0)
   .-(arg0, arg1)

   Subtraction ``arg0 - arg1``, of scalar types or :class:`NDArray`. Or create
   the negative of ``arg0``.




.. function:: mul_to!(dst :: NDArray, arg :: Union{Real, NDArray})

   Elementwise multiplication into ``dst`` of either a scalar or an :class:`NDArray` of the same shape.
   Inplace updating.




.. function::
   .*(arg0, arg1)

   Elementwise multiplication of ``arg0`` and ``arg``, could be either scalar or :class:`NDArray`.




.. function::
   *(arg0, arg1)

   Currently only multiplication a scalar with an :class:`NDArray` is implemented. Matrix multiplication
   is to be added soon.




.. function:: div_from!(dst :: NDArray, arg :: Union{Real, NDArray})

   Elementwise divide a scalar or an :class:`NDArray` of the same shape from ``dst``. Inplace updating.




.. function:: ./(arg0 :: NDArray, arg :: Union{Real, NDArray})

   Elementwise dividing an :class:`NDArray` by a scalar or another :class:`NDArray` of the same shape.




.. function:: /(arg0 :: NDArray, arg :: Real)

   Divide an :class:`NDArray` by a scalar. Matrix division (solving linear systems) is not implemented yet.




Manipulating as Julia Arrays
----------------------------

.. function:: @nd_as_jl(captures..., statement)

   A convenient macro that allows to operate :class:`NDArray` as Julia Arrays. For example,

   .. code-block:: julia

      x = mx.zeros(3,4)
      y = mx.ones(3,4)
      z = mx.zeros((3,4), mx.gpu())

      @mx.nd_as_jl ro=(x,y) rw=z begin
        # now x, y, z are just ordinary Julia Arrays
        z[:,1] = y[:,2]
        z[:,2] = 5
      end

   Under the hood, the macro convert all the declared captures from :class:`NDArray` into Julia
   Arrays, by using :func:`try_get_shared`. And automatically commit the modifications back into
   the :class:`NDArray` that is declared as ``rw``. This is useful for fast prototyping and when
   implement non-critical computations, such as :class:`AbstractEvalMetric`.

   .. note::

      - Multiple ``rw`` and / or ``ro`` capture declaration could be made.
      - The macro does **not** check to make sure that ``ro`` captures are not modified. If the
        original :class:`NDArray` lives in CPU memory, then it is very likely the corresponding
        Julia Array shares data with the :class:`NDArray`, so modifying the Julia Array will also
        modify the underlying :class:`NDArray`.
      - More importantly, since the :class:`NDArray` is
        asynchronized, we will wait for *writing* for ``rw`` variables but wait only for *reading*
        in ``ro`` variables. If we write into those ``ro`` variables, **and** if the memory is
        shared, racing condition might happen, and the behavior is undefined.
      - When an :class:`NDArray` is declared to be captured as ``rw``, its contents is always sync
        back in the end.
      - The execution results of the expanded macro is always ``nothing``.
      - The statements are wrapped in a ``let``, thus locally introduced new variables will not be
        available after the statements. So you will need to declare the variables before calling the
        macro if needed.




.. function:: try_get_shared(arr)

   Try to create a Julia array by sharing the data with the underlying :class:`NDArray`.

   :param NDArray arr: the array to be shared.

   .. warning::

      The returned array does not guarantee to share data with the underlying :class:`NDArray`.
      In particular, data sharing is possible only when the :class:`NDArray` lives on CPU.




.. function:: is_shared(j_arr, arr)

   Test whether ``j_arr`` is sharing data with ``arr``.

   :param Array j_arr: the Julia Array.
   :param NDArray arr: the :class:`NDArray`.




IO
--




.. function:: load(filename, ::Type{NDArray})

   Load NDArrays from binary file.

   :param AbstractString filename: the path of the file to load. It could be S3 or HDFS address.
   :return: Either ``Dict{Base.Symbol, NDArray}`` or ``Vector{NDArray}``.

   If the ``libmxnet`` is built with the corresponding component enabled. Examples

   * ``s3://my-bucket/path/my-s3-ndarray``
   * ``hdfs://my-bucket/path/my-hdfs-ndarray``
   * ``/path-to/my-local-ndarray``




.. function:: save(filename :: AbstractString, data)

   Save NDarrays to binary file. Filename could be S3 or HDFS address, if ``libmxnet`` is built
   with corresponding support.

   :param AbstractString filename: path to the binary file to write to.
   :param data: data to save to file.
   :type data: :class:`NDArray`, or a ``Vector{NDArray}`` or a ``Dict{Base.Symbol, NDArray}``.




libmxnet APIs
-------------




The libxmnet APIs are automatically imported from ``libmxnet.so``. The functions listed
here operate on :class:`NDArray` objects. The arguments to the functions are typically ordered
as

.. code-block:: julia

   func_name(arg_in1, arg_in2, ..., scalar1, scalar2, ..., arg_out1, arg_out2, ...)

unless ``NDARRAY_ARG_BEFORE_SCALAR`` is not set. In this case, the scalars are put before the input arguments:

.. code-block:: julia

   func_name(scalar1, scalar2, ..., arg_in1, arg_in2, ..., arg_out1, arg_out2, ...)


If ``ACCEPT_EMPTY_MUTATE_TARGET`` is set. An overloaded function without the output arguments will also be defined:

.. code-block:: julia

   func_name(arg_in1, arg_in2, ..., scalar1, scalar2, ...)

Upon calling, the output arguments will be automatically initialized with empty NDArrays.

Those functions always return the output arguments. If there is only one output (the typical situation), that
object (:class:`NDArray`) is returned. Otherwise, a tuple containing all the outputs will be returned.

Public APIs
^^^^^^^^^^^
.. function:: abs(...)

   Take absolute value of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: argmax_channel(...)

   Take argmax indices of each channel of the src.The result will be ndarray of shape (num_channel,) on the same device.
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: ceil(...)

   Take ceil value of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: choose_element_0index(...)

   Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: NDArray
   




.. function:: clip(...)

   Clip ndarray elements to range (a_min, a_max)
   
   :param src: Source input
   :type src: NDArray
   
   
   :param a_min: Minimum value
   :type a_min: real_t
   
   
   :param a_max: Maximum value
   :type a_max: real_t
   




.. function:: cos(...)

   Take cos of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: dot(...)

   Calculate 2D matrix multiplication
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: NDArray
   




.. function:: exp(...)

   Take exp of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: fill_element_0index(...)

   Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param mhs: Middle operand to the function.
   :type mhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: NDArray
   




.. function:: floor(...)

   Take floor value of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: log(...)

   Take log of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: max(...)

   Take max of the src.The result will be ndarray of shape (1,) on the same device.
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: min(...)

   Take min of the src.The result will be ndarray of shape (1,) on the same device.
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: norm(...)

   Take L2 norm of the src.The result will be ndarray of shape (1,) on the same device.
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: round(...)

   Take round value of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: rsqrt(...)

   Take rsqrt of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: sign(...)

   Take sign value of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: sin(...)

   Take sin of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: sqrt(...)

   Take sqrt of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: square(...)

   Take square of the src
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: sum(...)

   Take sum of the src.The result will be ndarray of shape (1,) on the same device.
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: sum_mid_internal(...)

   Take sum on medium dimension of the 3D src.
   
   :param src: Source input to the function
   :type src: NDArray
   




.. function:: transpose(...)

   Transpose the input matrix and return a new one
   
   :param src: Source input to the function
   :type src: NDArray
   



Internal APIs
^^^^^^^^^^^^^

.. note::

   Document and signatures for internal API functions might be incomplete.

.. function:: _broadcast(...)

   Broadcast array in the given axis to the given size
   
   :param src: source ndarray
   :type src: NDArray
   
   
   :param axis: axis to broadcast
   :type axis: int
   
   
   :param size: size of broadcast
   :type size: int
   




.. function:: _copyto(...)

   
   
   :param src: Source input to the function.
   :type src: NDArray
   




.. function:: _div(...)

   Multiply lhs by rhs
   
   :param lhs: Left operand  to the function
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function
   :type rhs: NDArray
   




.. function:: _div_scalar(...)

   
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: real_t
   




.. function:: _imdecode(...)

   Decode an image, clip to (x0, y0, x1, y1), substract mean, and write to buffer
   
   :param mean: image mean
   :type mean: NDArray
   
   
   :param index: buffer position for output
   :type index: int
   
   
   :param x0: x0
   :type x0: int
   
   
   :param y0: y0
   :type y0: int
   
   
   :param x1: x1
   :type x1: int
   
   
   :param y1: y1
   :type y1: int
   
   
   :param c: channel
   :type c: int
   
   
   :param size: length of str_img
   :type size: int
   




.. function:: _maximum(...)

   Elementwise max of lhs by rhs
   
   :param lhs: Left operand  to the function
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function
   :type rhs: NDArray
   




.. function:: _minimum(...)

   Elementwise min of lhs by rhs
   
   :param lhs: Left operand  to the function
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function
   :type rhs: NDArray
   




.. function:: _minus(...)

   Minus lhs and rhs
   
   :param lhs: Left operand  to the function
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function
   :type rhs: NDArray
   




.. function:: _minus_scalar(...)

   
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: real_t
   




.. function:: _mul(...)

   Multiply lhs and rhs
   
   :param lhs: Left operand  to the function
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function
   :type rhs: NDArray
   




.. function:: _mul_scalar(...)

   
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: real_t
   




.. function:: _onehot_encode(...)

   
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: NDArray
   




.. function:: _plus(...)

   Add lhs and rhs
   
   :param lhs: Left operand  to the function
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function
   :type rhs: NDArray
   




.. function:: _plus_scalar(...)

   
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: real_t
   




.. function:: _power(...)

   Elementwise power(lhs, rhs)
   
   :param lhs: Left operand  to the function
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function
   :type rhs: NDArray
   




.. function:: _random_gaussian(...)

   
   




.. function:: _random_uniform(...)

   
   




.. function:: _rdiv_scalar(...)

   
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: real_t
   




.. function:: _rminus_scalar(...)

   
   
   :param lhs: Left operand to the function.
   :type lhs: NDArray
   
   
   :param rhs: Right operand to the function.
   :type rhs: real_t
   




.. function:: _set_value(...)

   
   
   :param src: Source input to the function.
   :type src: real_t
   







