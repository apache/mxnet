
libmxnet APIs
-------------

Public APIs
^^^^^^^^^^^
.. function:: Activation(...)

   Apply activation function to input.
   
   :param Symbol data: Input data to activation function.
   
   
   :param {'relu', 'sigmoid', 'tanh'}, required act_type: Activation function to be applied.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: BatchNorm(...)

   Apply batch normalization to input.
   
   :param Symbol data: Input data to batch normalization
   
   
   :param float, optional, default=1e-10 eps: Epsilon to prevent div 0
   
   
   :param float, optional, default=0.1 momentum: Momentum for moving average
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: BlockGrad(...)

   Get output from a symbol and pass 0 gradient back
   
   :param Symbol data: Input data.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Concat(...)

   Perform an feature concat on channel dim (dim 1) over all the inputs.
   
   This function support variable length positional :class:`Symbol` inputs.
   
   :param int, required num_args: Number of inputs to be concated.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Convolution(...)

   Apply convolution to input then add a bias.
   
   :param Symbol data: Input data to the ConvolutionOp.
   
   
   :param Symbol weight: Weight matrix.
   
   
   :param Symbol bias: Bias parameter.
   
   
   :param Shape(tuple), required kernel: convolution kernel size: (y, x)
   
   
   :param Shape(tuple), optional, default=(1, 1) stride: convolution stride: (y, x)
   
   
   :param Shape(tuple), optional, default=(0, 0) pad: pad for convolution: (y, x)
   
   
   :param int (non-negative), required num_filter: convolution filter(channel) number
   
   
   :param int (non-negative), optional, default=1 num_group: number of groups partition
   
   
   :param long (non-negative), optional, default=512 workspace: Tmp workspace for convolution (MB)
   
   
   :param boolean, optional, default=False no_bias: Whether to disable bias parameter.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Dropout(...)

   Apply dropout to input
   
   :param Symbol data: Input data to dropout.
   
   
   :param float, optional, default=0.5 p: Fraction of the input that gets dropped out at training time
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: ElementWiseSum(...)

   Perform an elementwise sum over all the inputs.
   
   This function support variable length positional :class:`Symbol` inputs.
   
   :param int, required num_args: Number of inputs to be sumed.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Flatten(...)

   Flatten input
   
   :param Symbol data: Input data to  flatten.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: FullyConnected(...)

   Apply matrix multiplication to input then add a bias.
   
   :param Symbol data: Input data to the FullyConnectedOp.
   
   
   :param Symbol weight: Weight matrix.
   
   
   :param Symbol bias: Bias parameter.
   
   
   :param int, required num_hidden: Number of hidden nodes of the output.
   
   
   :param boolean, optional, default=False no_bias: Whether to disable bias parameter.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LRN(...)

   Apply convolution to input then add a bias.
   
   :param Symbol data: Input data to the ConvolutionOp.
   
   
   :param float, optional, default=0.0001 alpha: value of the alpha variance scaling parameter in the normalization formula
   
   
   :param float, optional, default=0.75 beta: value of the beta power parameter in the normalization formula
   
   
   :param float, optional, default=2 knorm: value of the k parameter in normalization formula
   
   
   :param int (non-negative), required nsize: normalization window width in elements.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LeakyReLU(...)

   Apply activation function to input.
   
   :param Symbol data: Input data to activation function.
   
   
   :param {'leaky', 'prelu', 'rrelu'},optional, default='leaky' act_type: Activation function to be applied.
   
   
   :param float, optional, default=0.25 slope: Init slope for the activation. (For leaky only)
   
   
   :param float, optional, default=0.125 lower_bound: Lower bound of random slope. (For rrelu only)
   
   
   :param float, optional, default=0.334 upper_bound: Upper bound of random slope. (For rrelu only)
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LinearRegressionOutput(...)

   Use linear regression for final output, this is used on final output of a net.
   
   :param Symbol data: Input data to function.
   
   
   :param Symbol label: Input label to function.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: LogisticRegressionOutput(...)

   Use Logistic regression for final output, this is used on final output of a net.
   Logistic regression is suitable for binary classification or probability prediction tasks.
   
   :param Symbol data: Input data to function.
   
   
   :param Symbol label: Input label to function.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Pooling(...)

   Perform spatial pooling on inputs.
   
   :param Symbol data: Input data to the pooling operator.
   
   
   :param Shape(tuple), required kernel: pooling kernel size: (y, x)
   
   
   :param {'avg', 'max', 'sum'}, required pool_type: Pooling type to be applied.
   
   
   :param Shape(tuple), optional, default=(1, 1) stride: stride: for pooling (y, x)
   
   
   :param Shape(tuple), optional, default=(0, 0) pad: pad for pooling: (y, x)
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Reshape(...)

   Reshape input to target shape
   
   :param Symbol data: Input data to  reshape.
   
   
   :param Shape(tuple), required target_shape: Target new shape
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: SliceChannel(...)

   Slice channel into many outputs with equally divided channel
   
   :param int, required num_outputs: Number of outputs to be sliced.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: Softmax(...)

   Perform a softmax transformation on input.
   
   :param Symbol data: Input data to softmax.
   
   
   :param float, optional, default=1 grad_scale: Scale the gradient by a float factor
   
   
   :param boolean, optional, default=False multi_output: If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: sqrt(...)

   Take square root of the src
   
   :param Symbol src: Source symbolic input to the function
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: square(...)

   Take square of the src
   
   :param Symbol src: Source symbolic input to the function
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   



Internal APIs
^^^^^^^^^^^^^

.. note::

   Document and signatures for internal API functions might be incomplete.

.. function:: _Div(...)

   Perform an elementwise div.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: _Minus(...)

   Perform an elementwise minus.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: _Mul(...)

   Perform an elementwise mul.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   




.. function:: _Plus(...)

   Perform an elementwise plus.
   
   :param Base.Symbol name: The name of the symbol. (e.g. `:my_symbol`), optional.
   
   :return: The constructed :class:`Symbol`.
   







