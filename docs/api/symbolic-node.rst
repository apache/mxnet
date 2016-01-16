
Symbolic API
============




.. class:: SymbolicNode

   SymbolicNode is the basic building block of the symbolic graph in MXNet.jl.




.. function:: deepcopy(self :: SymbolicNode)

   Make a deep copy of a SymbolicNode.




.. function:: copy(self :: SymbolicNode)

   Make a copy of a SymbolicNode. The same as making a deep copy.




.. function::
   call(self :: SymbolicNode, args :: SymbolicNode...)
   call(self :: SymbolicNode; kwargs...)

   Make a new node by composing ``self`` with ``args``. Or the arguments
   can be specified using keyword arguments.




.. function:: list_arguments(self :: SymbolicNode)

   List all the arguments of this node. The argument for a node contains both
   the inputs and parameters. For example, a :class:`FullyConnected` node will
   have both data and weights in its arguments. A composed node (e.g. a MLP) will
   list all the arguments for intermediate nodes.

   :return: A list of symbols indicating the names of the arguments.




.. function:: list_outputs(self :: SymbolicNode)

   List all the outputs of this node.

   :return: A list of symbols indicating the names of the outputs.




.. function:: list_auxiliary_states(self :: SymbolicNode)


   List all auxiliary states in the symbool.

   Auxiliary states are special states of symbols that do not corresponds to an argument,
   and do not have gradient. But still be useful for the specific operations.
   A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   Most operators do not have Auxiliary states.

   :return: A list of symbols indicating the names of the auxiliary states.




.. function:: get_internals(self :: SymbolicNode)

   Get a new grouped :class:`SymbolicNode` whose output contains all the internal outputs of
   this :class:`SymbolicNode`.




.. function:: get_attr(self :: SymbolicNode, key :: Symbol)

   Get attribute attached to this :class:`SymbolicNode` belonging to key.
   :return: The value belonging to key as a :class:`Nullable`.




.. function:: set_attr(self:: SymbolicNode, key :: Symbol, value :: AbstractString)

   Set the attribute key to value for this :class:`SymbolicNode`.

   .. warning::

      It is encouraged not to call this function directly, unless you know exactly what you are doing. The
      recommended way of setting attributes is when creating the :class:`SymbolicNode`. Changing
      the attributes of a :class:`SymbolicNode` that is already been used somewhere else might
      cause unexpected behavior and inconsistency.




.. function:: Variable(name :: Union{Symbol, AbstractString})

   Create a symbolic variable with the given name. This is typically used as a placeholder.
   For example, the data node, acting as the starting point of a network architecture.

   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`Variable`.




.. function:: Group(nodes :: SymbolicNode...)

   Create a :class:`SymbolicNode` by grouping nodes together.




.. function::
   infer_shape(self :: SymbolicNode; args...)
   infer_shape(self :: SymbolicNode; kwargs...)

   Do shape inference according to the input shapes. The input shapes could be provided
   as a list of shapes, which should specify the shapes of inputs in the same order as
   the arguments returned by :func:`list_arguments`. Alternatively, the shape information
   could be specified via keyword arguments.

   :return: A 3-tuple containing shapes of all the arguments, shapes of all the outputs and
            shapes of all the auxiliary variables. If shape inference failed due to incomplete
            or incompatible inputs, the return value will be ``(nothing, nothing, nothing)``.




.. function::
   getindex(self :: SymbolicNode, idx :: Union{Int, Base.Symbol, AbstractString})

   Get a node representing the specified output of this node. The index could be
   a symbol or string indicating the name of the output, or a 1-based integer
   indicating the index, as in the list of :func:`list_outputs`.




.. function:: to_json(self :: SymbolicNode)

   Convert a :class:`SymbolicNode` into a JSON string.




.. function:: from_json(repr :: AbstractString, ::Type{SymbolicNode})

   Load a :class:`SymbolicNode` from a JSON string representation.




.. function:: load(filename :: AbstractString, ::Type{SymbolicNode})

   Load a :class:`SymbolicNode` from a JSON file.




.. function:: save(filename :: AbstractString, node :: SymbolicNode)

   Save a :class:`SymbolicNode` to a JSON file.




libmxnet APIs
-------------

Public APIs
^^^^^^^^^^^
.. function:: Activation(...)

   Apply activation function to input.Softmax Activation is only available with CUDNN on GPUand will be computed at each location across channel if input is 4D.
   
   :param data: Input data to activation function.
   :type data: SymbolicNode
   
   
   :param act_type: Activation function to be applied.
   :type act_type: {'relu', 'sigmoid', 'softrelu', 'tanh'}, required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: BatchNorm(...)

   Apply batch normalization to input.
   
   :param data: Input data to batch normalization
   :type data: SymbolicNode
   
   
   :param eps: Epsilon to prevent div 0
   :type eps: float, optional, default=0.001
   
   
   :param momentum: Momentum for moving average
   :type momentum: float, optional, default=0.9
   
   
   :param fix_gamma: Fix gamma while training
   :type fix_gamma: boolean, optional, default=True
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: BlockGrad(...)

   Get output from a symbol and pass 0 gradient back
   
   :param data: Input data.
   :type data: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Cast(...)

   Cast array to a different data type.
   
   :param data: Input data to cast function.
   :type data: SymbolicNode
   
   
   :param dtype: Target data type.
   :type dtype: {'float16', 'float32', 'float64', 'int32', 'uint8'}, required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Concat(...)

   Perform an feature concat on channel dim (dim 1) over all the inputs.
   
   This function support variable length positional :class:`SymbolicNode` inputs.
   
   :param num_args: Number of inputs to be concated.
   :type num_args: int, required
   
   
   :param dim: the dimension to be concated.
   :type dim: int, optional, default='1'
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
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
   
   
   :param dilate: convolution dilate: (y, x)
   :type dilate: Shape(tuple), optional, default=(1, 1)
   
   
   :param pad: pad for convolution: (y, x)
   :type pad: Shape(tuple), optional, default=(0, 0)
   
   
   :param num_filter: convolution filter(channel) number
   :type num_filter: int (non-negative), required
   
   
   :param num_group: Number of groups partition. This option is not supported by CuDNN, you can use SliceChannel to num_group,apply convolution and concat instead to achieve the same need.
   :type num_group: int (non-negative), optional, default=1
   
   
   :param workspace: Tmp workspace for convolution (MB).
   :type workspace: long (non-negative), optional, default=512
   
   
   :param no_bias: Whether to disable bias parameter.
   :type no_bias: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Crop(...)

   Crop the 2th and 3th dim of input data, with the corresponding size of w_h orwith widht and height of the second input symbol
   
   This function support variable length positional :class:`SymbolicNode` inputs.
   
   :param num_args: Number of inputs for crop, if equals one, then we will use the h_wfor crop heihgt and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here
   :type num_args: int, required
   
   
   :param offset: corp offset coordinate: (y, x)
   :type offset: Shape(tuple), optional, default=(0, 0)
   
   
   :param h_w: corp height and weight: (h, w)
   :type h_w: Shape(tuple), optional, default=(0, 0)
   
   
   :param center_crop: If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like
   :type center_crop: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
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
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Dropout(...)

   Apply dropout to input
   
   :param data: Input data to dropout.
   :type data: SymbolicNode
   
   
   :param p: Fraction of the input that gets dropped out at training time
   :type p: float, optional, default=0.5
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: ElementWiseSum(...)

   Perform an elementwise sum over all the inputs.
   
   This function support variable length positional :class:`SymbolicNode` inputs.
   
   :param num_args: Number of inputs to be sumed.
   :type num_args: int, required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Embedding(...)

   Get embedding for one-hot input
   
   :param data: Input data to the EmbeddingOp.
   :type data: SymbolicNode
   
   
   :param weight: Enbedding weight matrix.
   :type weight: SymbolicNode
   
   
   :param input_dim: input dim of one-hot encoding
   :type input_dim: int, required
   
   
   :param output_dim: output dim of embedding
   :type output_dim: int, required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Flatten(...)

   Flatten input
   
   :param data: Input data to  flatten.
   :type data: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
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
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: IdentityAttachKLSparseReg(...)

   Apply a sparse regularization to the output a sigmoid activation function.
   
   :param data: Input data.
   :type data: SymbolicNode
   
   
   :param sparseness_target: The sparseness target
   :type sparseness_target: float, optional, default=0.1
   
   
   :param penalty: The tradeoff parameter for the sparseness penalty
   :type penalty: float, optional, default=0.001
   
   
   :param momentum: The momentum for running average
   :type momentum: float, optional, default=0.9
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
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
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: LeakyReLU(...)

   Apply activation function to input.
   
   :param data: Input data to activation function.
   :type data: SymbolicNode
   
   
   :param act_type: Activation function to be applied.
   :type act_type: {'elu', 'leaky', 'prelu', 'rrelu'},optional, default='leaky'
   
   
   :param slope: Init slope for the activation. (For leaky and elu only)
   :type slope: float, optional, default=0.25
   
   
   :param lower_bound: Lower bound of random slope. (For rrelu only)
   :type lower_bound: float, optional, default=0.125
   
   
   :param upper_bound: Upper bound of random slope. (For rrelu only)
   :type upper_bound: float, optional, default=0.334
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: LinearRegressionOutput(...)

   Use linear regression for final output, this is used on final output of a net.
   
   :param data: Input data to function.
   :type data: SymbolicNode
   
   
   :param label: Input label to function.
   :type label: SymbolicNode
   
   
   :param grad_scale: Scale the gradient by a float factor
   :type grad_scale: float, optional, default=1
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: LogisticRegressionOutput(...)

   Use Logistic regression for final output, this is used on final output of a net.
   Logistic regression is suitable for binary classification or probability prediction tasks.
   
   :param data: Input data to function.
   :type data: SymbolicNode
   
   
   :param label: Input label to function.
   :type label: SymbolicNode
   
   
   :param grad_scale: Scale the gradient by a float factor
   :type grad_scale: float, optional, default=1
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: MAERegressionOutput(...)

   Use mean absolute error regression for final output, this is used on final output of a net.
   
   :param data: Input data to function.
   :type data: SymbolicNode
   
   
   :param label: Input label to function.
   :type label: SymbolicNode
   
   
   :param grad_scale: Scale the gradient by a float factor
   :type grad_scale: float, optional, default=1
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
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
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Reshape(...)

   Reshape input to target shape
   
   :param data: Input data to  reshape.
   :type data: SymbolicNode
   
   
   :param target_shape: Target new shape. One and only one dim can be 0, in which case it will be infered from the rest of dims
   :type target_shape: Shape(tuple), required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: SliceChannel(...)

   Slice channel into many outputs with equally divided channel
   
   :param num_outputs: Number of outputs to be sliced.
   :type num_outputs: int, required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: Softmax(...)

   DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput
   
   :param data: Input data to softmax.
   :type data: SymbolicNode
   
   
   :param grad_scale: Scale the gradient by a float factor
   :type grad_scale: float, optional, default=1
   
   
   :param ignore_label: the ignore_label will not work in backward, and this onlybe used when multi_output=true
   :type ignore_label: float, optional, default=-1
   
   
   :param multi_output: If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
   :type multi_output: boolean, optional, default=False
   
   
   :param use_ignore: If set to true, the ignore_label value will not contributorto the backward gradient
   :type use_ignore: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: SoftmaxActivation(...)

   Apply softmax activation to input. This is intended for internal layers. For output (loss layer) please use SoftmaxOutput. If type=instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If type=channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
   
   :param data: Input data to activation function.
   :type data: SymbolicNode
   
   
   :param type: Softmax Mode. If set to instance, this operator will compute a softmax for each instance in the batch; this is the default mode. If set to channel, this operator will compute a num_channel-class softmax at each position of each instance; this can be used for fully convolutional network, image segmentation, etc.
   :type type: {'channel', 'instance'},optional, default='instance'
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: SoftmaxOutput(...)

   Perform a softmax transformation on input, backprop with logloss.
   
   :param data: Input data to softmax.
   :type data: SymbolicNode
   
   
   :param grad_scale: Scale the gradient by a float factor
   :type grad_scale: float, optional, default=1
   
   
   :param ignore_label: the ignore_label will not work in backward, and this onlybe used when multi_output=true
   :type ignore_label: float, optional, default=-1
   
   
   :param multi_output: If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor, softmax will generate n*x_1*...*x_n output, eachhas k classes
   :type multi_output: boolean, optional, default=False
   
   
   :param use_ignore: If set to true, the ignore_label value will not contributorto the backward gradient
   :type use_ignore: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: SwapAxis(...)

   Apply swapaxis to input.
   
   :param data: Input data to the SwapAxisOp.
   :type data: SymbolicNode
   
   
   :param dim1: the first axis to be swapped.
   :type dim1: int (non-negative), optional, default=0
   
   
   :param dim2: the second axis to be swapped.
   :type dim2: int (non-negative), optional, default=0
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: UpSampling(...)

   Perform nearest neighboor/bilinear up sampling to inputs
   
   This function support variable length positional :class:`SymbolicNode` inputs.
   
   :param scale: Up sampling scale
   :type scale: int (non-negative), required
   
   
   :param num_filter: Input filter. Only used by nearest sample_type.
   :type num_filter: int (non-negative), optional, default=0
   
   
   :param sample_type: upsampling method
   :type sample_type: {'bilinear', 'nearest'}, required
   
   
   :param multi_input_mode: How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
   :type multi_input_mode: {'concat', 'sum'},optional, default='concat'
   
   
   :param num_args: Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.
   :type num_args: int, required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: abs(...)

   Take absolute value of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: ceil(...)

   Take ceil value of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: cos(...)

   Take cos of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: exp(...)

   Take exp of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: floor(...)

   Take floor value of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: log(...)

   Take log of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: round(...)

   Take round value of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: rsqrt(...)

   Take rsqrt of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: sign(...)

   Take sign value of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: sin(...)

   Take sin of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: sqrt(...)

   Take sqrt of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: square(...)

   Take square of the src
   
   :param src: Source symbolic input to the function
   :type src: SymbolicNode
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   



Internal APIs
^^^^^^^^^^^^^

.. note::

   Document and signatures for internal API functions might be incomplete.

.. function:: _CrossDeviceCopy(...)

   Special op to copy data cross device
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Div(...)

   Perform an elementwise div.
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _DivScalar(...)

   Perform an elementwise div.
   
   :param array: Input array operand to the operation.
   :type array: SymbolicNode
   
   
   :param scalar: scalar value.
   :type scalar: float, required
   
   
   :param scalar_on_left: scalar operand is on the left.
   :type scalar_on_left: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Maximum(...)

   Perform an elementwise power.
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _MaximumScalar(...)

   Perform an elementwise maximum.
   
   :param array: Input array operand to the operation.
   :type array: SymbolicNode
   
   
   :param scalar: scalar value.
   :type scalar: float, required
   
   
   :param scalar_on_left: scalar operand is on the left.
   :type scalar_on_left: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Minimum(...)

   Perform an elementwise power.
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _MinimumScalar(...)

   Perform an elementwise minimum.
   
   :param array: Input array operand to the operation.
   :type array: SymbolicNode
   
   
   :param scalar: scalar value.
   :type scalar: float, required
   
   
   :param scalar_on_left: scalar operand is on the left.
   :type scalar_on_left: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Minus(...)

   Perform an elementwise minus.
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _MinusScalar(...)

   Perform an elementwise minus.
   
   :param array: Input array operand to the operation.
   :type array: SymbolicNode
   
   
   :param scalar: scalar value.
   :type scalar: float, required
   
   
   :param scalar_on_left: scalar operand is on the left.
   :type scalar_on_left: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Mul(...)

   Perform an elementwise mul.
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _MulScalar(...)

   Perform an elementwise mul.
   
   :param array: Input array operand to the operation.
   :type array: SymbolicNode
   
   
   :param scalar: scalar value.
   :type scalar: float, required
   
   
   :param scalar_on_left: scalar operand is on the left.
   :type scalar_on_left: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _NDArray(...)

   Stub for implementing an operator implemented in native frontend language with ndarray.
   
   :param info: 
   :type info: , required
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Native(...)

   Stub for implementing an operator implemented in native frontend language.
   
   :param info: 
   :type info: , required
   
   
   :param need_top_grad: Whether this layer needs out grad for backward. Should be false for loss layers.
   :type need_top_grad: boolean, optional, default=True
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Plus(...)

   Perform an elementwise plus.
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _PlusScalar(...)

   Perform an elementwise plus.
   
   :param array: Input array operand to the operation.
   :type array: SymbolicNode
   
   
   :param scalar: scalar value.
   :type scalar: float, required
   
   
   :param scalar_on_left: scalar operand is on the left.
   :type scalar_on_left: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _Power(...)

   Perform an elementwise power.
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   




.. function:: _PowerScalar(...)

   Perform an elementwise power.
   
   :param array: Input array operand to the operation.
   :type array: SymbolicNode
   
   
   :param scalar: scalar value.
   :type scalar: float, required
   
   
   :param scalar_on_left: scalar operand is on the left.
   :type scalar_on_left: boolean, optional, default=False
   
   :param Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.
   :param Dict{Symbol, AbstractString} attrs: The attributes associated with this :class:`SymbolicNode`.
   
   :return: the constructed :class:`SymbolicNode`.
   







