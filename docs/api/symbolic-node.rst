
Symbolic API
============




.. class:: SymbolicNode

   SymbolicNode is the basic building block of the symbolic graph in MXNet.jl.




.. function:: deepcopy(self :: SymbolicNode)

   Make a deep copy of a SymbolicNode.




.. function:: copy(self :: SymbolicNode)

   Make a copy of a SymbolicNode. The same as making a deep copy.




libmxnet APIs
-------------

Public APIs
^^^^^^^^^^^
.. function:: Activation(...)

   Apply activation function to input.
   
   :param data: Input data to activation function.
   :type data: SymbolicNode
   
   
   :param act_type: Activation function to be applied.
   :type act_type: {'relu', 'sigmoid', 'tanh'}, required
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: BatchNorm(...)

   Apply batch normalization to input.
   
   :param data: Input data to batch normalization
   :type data: SymbolicNode
   
   
   :param eps: Epsilon to prevent div 0
   :type eps: float, optional, default=1e-10
   
   
   :param momentum: Momentum for moving average
   :type momentum: float, optional, default=0.1
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: BlockGrad(...)

   Get output from a symbol and pass 0 gradient back
   
   :param data: Input data.
   :type data: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Concat(...)

   Perform an feature concat on channel dim (dim 1) over all the inputs.
   
   This function support variable length positional :class:`SymbolicNode` inputs.
   
   :param num_args: Number of inputs to be concated.
   :type num_args: int, required
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Convolution(...)

   Apply convolution to input then add a bias.
   
   :param data: Input data to the ConvolutionOp.
   :type data: SymbolicNode
   
   
   :param weight: Weight matrix.
   :type weight: SymbolicNode
   
   
   :param bias: Bias parameter.
   :type bias: SymbolicNode
   
   
   :param kernel: convolution kernel size: (y, x)
   :type kernel: Shape(tuple), required
   
   
   :param stride: convolution stride: (y, x)
   :type stride: Shape(tuple), optional, default=(1, 1)
   
   
   :param pad: pad for convolution: (y, x)
   :type pad: Shape(tuple), optional, default=(0, 0)
   
   
   :param num_filter: convolution filter(channel) number
   :type num_filter: int (non-negative), required
   
   
   :param num_group: Number of groups partition. This option is not supported by CuDNN, you can use SliceChannel to num_group,apply convolution and concat instead to achieve the same need.
   :type num_group: int (non-negative), optional, default=1
   
   
   :param workspace: Tmp workspace for convolution (MB)
   :type workspace: long (non-negative), optional, default=512
   
   
   :param no_bias: Whether to disable bias parameter.
   :type no_bias: boolean, optional, default=False
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Deconvolution(...)

   Apply deconvolution to input then add a bias.
   
   :param data: Input data to the DeconvolutionOp.
   :type data: SymbolicNode
   
   
   :param weight: Weight matrix.
   :type weight: SymbolicNode
   
   
   :param bias: Bias parameter.
   :type bias: SymbolicNode
   
   
   :param kernel: deconvolution kernel size: (y, x)
   :type kernel: Shape(tuple), required
   
   
   :param stride: deconvolution stride: (y, x)
   :type stride: Shape(tuple), optional, default=(1, 1)
   
   
   :param pad: pad for deconvolution: (y, x)
   :type pad: Shape(tuple), optional, default=(0, 0)
   
   
   :param num_filter: deconvolution filter(channel) number
   :type num_filter: int (non-negative), required
   
   
   :param num_group: number of groups partition
   :type num_group: int (non-negative), optional, default=1
   
   
   :param workspace: Tmp workspace for deconvolution (MB)
   :type workspace: long (non-negative), optional, default=512
   
   
   :param no_bias: Whether to disable bias parameter.
   :type no_bias: boolean, optional, default=True
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Dropout(...)

   Apply dropout to input
   
   :param data: Input data to dropout.
   :type data: SymbolicNode
   
   
   :param p: Fraction of the input that gets dropped out at training time
   :type p: float, optional, default=0.5
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: ElementWiseSum(...)

   Perform an elementwise sum over all the inputs.
   
   This function support variable length positional :class:`SymbolicNode` inputs.
   
   :param num_args: Number of inputs to be sumed.
   :type num_args: int, required
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Flatten(...)

   Flatten input
   
   :param data: Input data to  flatten.
   :type data: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: FullyConnected(...)

   Apply matrix multiplication to input then add a bias.
   
   :param data: Input data to the FullyConnectedOp.
   :type data: SymbolicNode
   
   
   :param weight: Weight matrix.
   :type weight: SymbolicNode
   
   
   :param bias: Bias parameter.
   :type bias: SymbolicNode
   
   
   :param num_hidden: Number of hidden nodes of the output.
   :type num_hidden: int, required
   
   
   :param no_bias: Whether to disable bias parameter.
   :type no_bias: boolean, optional, default=False
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: LRN(...)

   Apply convolution to input then add a bias.
   
   :param data: Input data to the ConvolutionOp.
   :type data: SymbolicNode
   
   
   :param alpha: value of the alpha variance scaling parameter in the normalization formula
   :type alpha: float, optional, default=0.0001
   
   
   :param beta: value of the beta power parameter in the normalization formula
   :type beta: float, optional, default=0.75
   
   
   :param knorm: value of the k parameter in normalization formula
   :type knorm: float, optional, default=2
   
   
   :param nsize: normalization window width in elements.
   :type nsize: int (non-negative), required
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: LeakyReLU(...)

   Apply activation function to input.
   
   :param data: Input data to activation function.
   :type data: SymbolicNode
   
   
   :param act_type: Activation function to be applied.
   :type act_type: {'leaky', 'prelu', 'rrelu'},optional, default='leaky'
   
   
   :param slope: Init slope for the activation. (For leaky only)
   :type slope: float, optional, default=0.25
   
   
   :param lower_bound: Lower bound of random slope. (For rrelu only)
   :type lower_bound: float, optional, default=0.125
   
   
   :param upper_bound: Upper bound of random slope. (For rrelu only)
   :type upper_bound: float, optional, default=0.334
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: LinearRegressionOutput(...)

   Use linear regression for final output, this is used on final output of a net.
   
   :param data: Input data to function.
   :type data: SymbolicNode
   
   
   :param label: Input label to function.
   :type label: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: LogisticRegressionOutput(...)

   Use Logistic regression for final output, this is used on final output of a net.
   Logistic regression is suitable for binary classification or probability prediction tasks.
   
   :param data: Input data to function.
   :type data: SymbolicNode
   
   
   :param label: Input label to function.
   :type label: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Pooling(...)

   Perform spatial pooling on inputs.
   
   :param data: Input data to the pooling operator.
   :type data: SymbolicNode
   
   
   :param kernel: pooling kernel size: (y, x)
   :type kernel: Shape(tuple), required
   
   
   :param pool_type: Pooling type to be applied.
   :type pool_type: {'avg', 'max', 'sum'}, required
   
   
   :param stride: stride: for pooling (y, x)
   :type stride: Shape(tuple), optional, default=(1, 1)
   
   
   :param pad: pad for pooling: (y, x)
   :type pad: Shape(tuple), optional, default=(0, 0)
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Reshape(...)

   Reshape input to target shape
   
   :param data: Input data to  reshape.
   :type data: SymbolicNode
   
   
   :param target_shape: Target new shape
   :type target_shape: Shape(tuple), required
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: SliceChannel(...)

   Slice channel into many outputs with equally divided channel
   
   :param num_outputs: Number of outputs to be sliced.
   :type num_outputs: int, required
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Softmax(...)

   DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
   
   :param data: Input data to softmax.
   :type data: SymbolicNode
   
   
   :param grad_scale: Scale the gradient by a float factor
   :type grad_scale: float, optional, default=1
   
   
   :param multi_output: If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
   :type multi_output: boolean, optional, default=False
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: SoftmaxOutput(...)

   Perform a softmax transformation on input, backprop with logloss.
   
   :param data: Input data to softmax.
   :type data: SymbolicNode
   
   
   :param grad_scale: Scale the gradient by a float factor
   :type grad_scale: float, optional, default=1
   
   
   :param multi_output: If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
   :type multi_output: boolean, optional, default=False
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: SwapAxis(...)

   Apply swapaxis to input.
   
   :param data: Input data to the SwapAxisOp.
   :type data: SymbolicNode
   
   
   :param dim1: the first axis to be swapped.
   :type dim1: int (non-negative), optional, default=0
   
   
   :param dim2: the second axis to be swapped.
   :type dim2: int (non-negative), optional, default=0
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: exp(...)

   Take exp of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: log(...)

   Take log of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: sqrt(...)

   Take sqrt of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: square(...)

   Take square of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   



Internal APIs
^^^^^^^^^^^^^

.. note::

   Document and signatures for internal API functions might be incomplete.

.. function:: _Div(...)

   Perform an elementwise div.
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Minus(...)

   Perform an elementwise minus.
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Mul(...)

   Perform an elementwise mul.
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Native(...)

   Stub for implementing an operator implemented in native frontend language.
   
   :param info: 
   :type info: , required
   
   
   :param need_top_grad: Whether this layer needs out grad for backward. Should be false for loss layers.
   :type need_top_grad: boolean, optional, default=True
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Plus(...)

   Perform an elementwise plus.
   
   :param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   
   :return: the constructed :class:`SymbolicNode`.
   







