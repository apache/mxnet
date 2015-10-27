
Symbol
======




.. class:: Symbol

   Symbol is the basic building block of the symbolic graph in MXNet.jl.

   .. note::

      Throughout this documentation, ``Symbol`` always refer to this :class:`Symbol` type.
      When we refer to the Julia's build-in symbol type (e.g. ``typeof(:foo)``), we always
      say ``Base.Symbol``.




libmxnet APIs
-------------

Public APIs
^^^^^^^^^^^
.. function:: Activation(...)

   Apply activation function to input.
   
   :param data: ``Symbol``. Input data to activation function.
   
   
   :param act_type: ``{'relu', 'sigmoid', 'tanh'}, required``. Activation function to be applied.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: BatchNorm(...)

   Apply batch normalization to input.
   
   :param data: ``Symbol``. Input data to batch normalization
   
   
   :param eps: ``float, optional, default=1e-10``. Epsilon to prevent div 0
   
   
   :param momentum: ``float, optional, default=0.1``. Momentum for moving average
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: BlockGrad(...)

   Get output from a symbol and pass 0 gradient back
   
   :param data: ``Symbol``. Input data.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Concat(...)

   Perform an feature concat on channel dim (dim 1) over all the inputs.
   
   This function support variable length positional :class:`Symbol` inputs.
   
   :param num_args: ``int, required``. Number of inputs to be concated.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Convolution(...)

   Apply convolution to input then add a bias.
   
   :param data: ``Symbol``. Input data to the ConvolutionOp.
   
   
   :param weight: ``Symbol``. Weight matrix.
   
   
   :param bias: ``Symbol``. Bias parameter.
   
   
   :param kernel: ``Shape(tuple), required``. convolution kernel size: (y, x)
   
   
   :param stride: ``Shape(tuple), optional, default=(1, 1)``. convolution stride: (y, x)
   
   
   :param pad: ``Shape(tuple), optional, default=(0, 0)``. pad for convolution: (y, x)
   
   
   :param num_filter: ``int (non-negative), required``. convolution filter(channel) number
   
   
   :param num_group: ``int (non-negative), optional, default=1``. number of groups partition
   
   
   :param workspace: ``long (non-negative), optional, default=512``. Tmp workspace for convolution (MB)
   
   
   :param no_bias: ``boolean, optional, default=False``. Whether to disable bias parameter.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Dropout(...)

   Apply dropout to input
   
   :param data: ``Symbol``. Input data to dropout.
   
   
   :param p: ``float, optional, default=0.5``. Fraction of the input that gets dropped out at training time
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: ElementWiseSum(...)

   Perform an elementwise sum over all the inputs.
   
   This function support variable length positional :class:`Symbol` inputs.
   
   :param num_args: ``int, required``. Number of inputs to be sumed.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Flatten(...)

   Flatten input
   
   :param data: ``Symbol``. Input data to  flatten.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: FullyConnected(...)

   Apply matrix multiplication to input then add a bias.
   
   :param data: ``Symbol``. Input data to the FullyConnectedOp.
   
   
   :param weight: ``Symbol``. Weight matrix.
   
   
   :param bias: ``Symbol``. Bias parameter.
   
   
   :param num_hidden: ``int, required``. Number of hidden nodes of the output.
   
   
   :param no_bias: ``boolean, optional, default=False``. Whether to disable bias parameter.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LRN(...)

   Apply convolution to input then add a bias.
   
   :param data: ``Symbol``. Input data to the ConvolutionOp.
   
   
   :param alpha: ``float, optional, default=0.0001``. value of the alpha variance scaling parameter in the normalization formula
   
   
   :param beta: ``float, optional, default=0.75``. value of the beta power parameter in the normalization formula
   
   
   :param knorm: ``float, optional, default=2``. value of the k parameter in normalization formula
   
   
   :param nsize: ``int (non-negative), required``. normalization window width in elements.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LeakyReLU(...)

   Apply activation function to input.
   
   :param data: ``Symbol``. Input data to activation function.
   
   
   :param act_type: ``{'leaky', 'prelu', 'rrelu'},optional, default='leaky'``. Activation function to be applied.
   
   
   :param slope: ``float, optional, default=0.25``. Init slope for the activation. (For leaky only)
   
   
   :param lower_bound: ``float, optional, default=0.125``. Lower bound of random slope. (For rrelu only)
   
   
   :param upper_bound: ``float, optional, default=0.334``. Upper bound of random slope. (For rrelu only)
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LinearRegressionOutput(...)

   Use linear regression for final output, this is used on final output of a net.
   
   :param data: ``Symbol``. Input data to function.
   
   
   :param label: ``Symbol``. Input label to function.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LogisticRegressionOutput(...)

   Use Logistic regression for final output, this is used on final output of a net.
   Logistic regression is suitable for binary classification or probability prediction tasks.
   
   :param data: ``Symbol``. Input data to function.
   
   
   :param label: ``Symbol``. Input label to function.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Pooling(...)

   Perform spatial pooling on inputs.
   
   :param data: ``Symbol``. Input data to the pooling operator.
   
   
   :param kernel: ``Shape(tuple), required``. pooling kernel size: (y, x)
   
   
   :param pool_type: ``{'avg', 'max', 'sum'}, required``. Pooling type to be applied.
   
   
   :param stride: ``Shape(tuple), optional, default=(1, 1)``. stride: for pooling (y, x)
   
   
   :param pad: ``Shape(tuple), optional, default=(0, 0)``. pad for pooling: (y, x)
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Reshape(...)

   Reshape input to target shape
   
   :param data: ``Symbol``. Input data to  reshape.
   
   
   :param target_shape: ``Shape(tuple), required``. Target new shape
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: SliceChannel(...)

   Slice channel into many outputs with equally divided channel
   
   :param num_outputs: ``int, required``. Number of outputs to be sliced.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Softmax(...)

   Perform a softmax transformation on input.
   
   :param data: ``Symbol``. Input data to softmax.
   
   
   :param grad_scale: ``float, optional, default=1``. Scale the gradient by a float factor
   
   
   :param multi_output: ``boolean, optional, default=False``. If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: sqrt(...)

   Take square root of the src
   
   :param src: ``Symbol``. Source symbolic input to the function
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: square(...)

   Take square of the src
   
   :param src: ``Symbol``. Source symbolic input to the function
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   



Internal APIs
^^^^^^^^^^^^^

.. note::

   Document and signatures for internal API functions might be incomplete.

.. function:: _Div(...)

   Perform an elementwise div.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: _Minus(...)

   Perform an elementwise minus.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: _Mul(...)

   Perform an elementwise mul.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: _Plus(...)

   Perform an elementwise plus.
   
   :param name: ``Base.Symbol``. The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   







