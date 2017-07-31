import numpy as np


def _get_input_output_name(net, node, index=0):
    name = node['name']
    inputs = node['inputs']

    if index == 'all':
        input_name = [_get_node_name(net, inputs[id][0]) for id in range(len(inputs))]
    elif type(index) == int:
        input_name = _get_node_name(net, inputs[0][0])
    else:
        input_name = [_get_node_name(net, inputs[id][0]) for id in index]
    return input_name, name


def _get_node_name(net, node_id):
    return net['nodes'][node_id]['name']


def _get_node_shape(net, node_id):
    return net['nodes'][node_id]['shape']


# FYI for images in CoreML channel, H, W in CoreML for image

# TODO LOWER PRIORITY (BUT STILL IMPORTANT)
# mxnet.symbol.repeat -> builder.add_repeat to flatten and repeat the NDArray sequence
# mxnet.symbol.Crop -> builder.add_crop to crop image along spacial dimensions
# mxnet.symbol.Pad -> builder.add_padding putting 0's on height and width for tensor


# TODO EVENTUALLY 

# depthwise seperable convolution support through groups in builder.add_convolution
# add_optional -> for all RNNs defining what goes in and out (to define beam search or if input is streaming)
# mx.symbol.Embedding -> add_embedding takes indicies, word ids from dict that is outside coreml or 
# in pipeline only if we have text mapping to indicies
# FusedRNNCell -> add_bidirlstm
#  add_unilstm -> reverse_input param true as second and concat on outputs
# Do vanilla (0.9 mxnet) lstm, gru, vanilla_rnn


def convert_reshape(net, node, module, builder):
    """Converts a reshape layer from mxnet to coreml.

    This doesn't currently handle the deprecated parameters for the reshape layer.

    Parameters
    ----------
    network: net
        An mxnet network object.

    layer: node
        Node to convert.

    module: module
        A module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """

    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    target_shape = node['shape']

    if any(item <= 0 for item in target_shape):
        raise NotImplementedError('Special dimensional values less than or equal to 0 are not supported yet.'
                                  'Feel free to file an issue here: https://github.com/dmlc/mxnet/issues.')

    if 'reverse' in node and node['reverse'] == 'True':
        raise NotImplementedError('"reverse" parameter is not supported by yet.'
                                  'Feel free to file an issue here: https://github.com/dmlc/mxnet/issues.')

    mode = 0 # CHANNEL_FIRST
    builder.add_reshape(name, input_name, output_name, target_shape, mode)


def convert_transpose(net, node, module, builder):
    """Convert a transpose layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    param = node['attr']
    from ast import literal_eval
    axes = literal_eval(param['axes'])
    builder.add_permute(name, axes, input_name, output_name)


def convert_flatten(net, node, module, builder):
    """Convert a flatten layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    mode = 0 # CHANNEL_FIRST
    builder.add_flatten(name, mode, input_name, output_name)


def convert_softmax(net, node, module, builder):
    """Convert a softmax layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    builder.add_softmax(name=name,
                        input_name=input_name,
                        output_name=output_name)


def convert_activation(net, node, module, builder):
    """Convert an activation layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    mx_non_linearity = node['attr']['act_type']
    #TODO add SCALED_TANH, SOFTPLUS, SOFTSIGN, SIGMOID_HARD, LEAKYRELU, PRELU, ELU, PARAMETRICSOFTPLUS, THRESHOLDEDRELU, LINEAR
    if mx_non_linearity == 'relu':
        non_linearity = 'RELU'
    elif mx_non_linearity == 'tanh':
        non_linearity = 'TANH'
    elif mx_non_linearity == 'sigmoid':
        non_linearity = 'SIGMOID'
    else:
        raise TypeError('Unknown activation type %s' % mx_non_linearity)
    builder.add_activation(name = name,
                           non_linearity = non_linearity,
                           input_name = input_name,
                           output_name = output_name)


def convert_elementwise_add(net, node, module, builder):
    """Convert an elementwise add layer from mxnet to coreml.

        Parameters
        ----------
        network: net
        A mxnet network object.

        layer: node
        Node to convert.

        module: module
        An module for MXNet

        builder: NeuralNetworkBuilder
        A neural network builder object.
        """

    input_names, output_name = _get_input_output_name(net, node, [0, 1])
    name = node['name']

    builder.add_elementwise(name, input_names, output_name, 'ADD')


def convert_dense(net, node, module, builder):
    """Convert a dense layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    param = node['attr']
    has_bias = True
    name = node['name']

    inputs = node['inputs']
    outputs = node['outputs']
    args, aux = module.get_params()
    W = args[_get_node_name(net, inputs[1][0])].asnumpy()
    if has_bias:
        Wb = args[_get_node_name(net, inputs[2][0])].asnumpy()
    else:
        Wb = None
    nC, nB = W.shape

    builder.add_inner_product(
        name=name,
        W=W,
        b=Wb,
        input_channels=nB,
        output_channels=nC,
        has_bias=has_bias,
        input_name=input_name,
        output_name=output_name
    )


def convert_convolution(net, node, module, builder):
    """Convert a convolution layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    param = node['attr']
    inputs = node['inputs']
    outputs = node['outputs']
    args, aux = module.get_params()
    from ast import literal_eval

    if 'no_bias' in param.keys():
        has_bias = not literal_eval(param['no_bias'])
    else:
        has_bias = True

    if literal_eval(param['pad']) != (0, 0):
        pad = literal_eval(param['pad'])
        builder.add_padding(
            name=name+"_pad",
            left=pad[1],
            right=pad[1],
            top=pad[0],
            bottom=pad[0],
            value=0,
            input_name=input_name,
            output_name=name+"_pad_output")
        input_name = name+"_pad_output"

    border_mode = "valid"

    n_filters = int(param['num_filter'])

    W = args[_get_node_name(net, inputs[1][0])].asnumpy()
    if has_bias:
        Wb = args[_get_node_name(net, inputs[2][0])].asnumpy()
    else:
        Wb = None

    channels = W.shape[1]
    stride_height, stride_width = literal_eval(param['stride'])
    kernel_height, kernel_width = literal_eval(param['kernel'])
    # TODO add padding here
    W = W.transpose((2, 3, 1, 0))
    builder.add_convolution(
        name=name,
        kernel_channels=channels,
        output_channels=n_filters,
        height=kernel_height,
        width=kernel_width,
        stride_height=stride_height,
        stride_width=stride_width,
        border_mode=border_mode,
        groups=1,
        W=W,
        b=Wb,
        has_bias=has_bias,
        is_deconv=False,
        output_shape=None,
        input_name=input_name,
        output_name=output_name)


def convert_pooling(net, node, module, builder):
    """Convert a pooling layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    inputs = node['inputs']
    param = node['attr']
    outputs = node['outputs']
    args, aux = module.get_params()

    layer_type_mx = param['pool_type']
    if layer_type_mx == 'max':
        layer_type = 'MAX'
    elif layer_type_mx == 'avg':
        layer_type = 'AVERAGE'
    else:
        raise TypeError("Pooling type %s not supported" % layer_type_mx)

    from ast import literal_eval
    stride_height, stride_width = literal_eval(param['stride'])
    kernel_width, kernel_height = literal_eval(param['kernel'])

    padding_type = 'VALID'
    if 'global_pool' in param.keys():
        is_global = literal_eval(param['global_pool'])
    else:
        is_global = False
    builder.add_pooling(
        name = name,
        height = kernel_height,
        width = kernel_width,
        stride_height = stride_height,
        stride_width = stride_width,
        layer_type = layer_type,
        padding_type = padding_type,
        exclude_pad_area = False,
        is_global = is_global,
        input_name = input_name,
        output_name = output_name)

    # Add padding if there is any
    poolingLayer = builder.nn_spec.layers[-1].pooling
    pad = literal_eval(param['pad'])
    for i in range(len(pad)):
        poolingLayer.valid.paddingAmounts.borderAmounts[i].startEdgeSize = pad[i]
        poolingLayer.valid.paddingAmounts.borderAmounts[i].endEdgeSize = pad[i]


def convert_batchnorm(net, node, module, builder):
    """Convert a transpose layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    inputs = node['inputs']

    from ast import literal_eval
    eps = 1e-3 # Default value of eps for MXNet.
    if 'attr' in node:
        if 'eps' in node['attr']:
            eps = literal_eval(node['attr']['eps'])
        if 'use_global_stats' in node['attr']:
            use_global_stats = literal_eval(node['attr']['use_global_stats'])
            if use_global_stats is False:
                raise Exception("CoreML doesn't support local batch-norm. Feel free to retrain your MXNet model "
                "with use_global_stats set to True and convert again. You could also use -f force flag to ignore "
                "this error; note that this may cause some differences in prediction b/w MXNet and CoreML.")
                # TODO provide the flag.

    args, aux = module.get_params()
    gamma = args[_get_node_name(net, inputs[1][0])].asnumpy()
    beta = args[_get_node_name(net, inputs[2][0])].asnumpy()
    mean = aux[_get_node_name(net, inputs[3][0])].asnumpy()
    variance = aux[_get_node_name(net, inputs[4][0])].asnumpy()
    nb_channels = gamma.shape[0]
    builder.add_batchnorm(
        name=name,
        channels=nb_channels,
        gamma=gamma,
        beta=beta,
        mean=mean,
        variance=variance,
        input_name=input_name,
        output_name=output_name,
        epsilon=eps)


def convert_concat(net, node, module, builder):
    """Convert concat layer from mxnet to coreml.

    Parameters
    ----------
    network: net
    A mxnet network object.

    layer: node
    Node to convert.

    module: module
    An module for MXNet

    builder: NeuralNetworkBuilder
    A neural network builder object.
    """
    # Get input and output names
    input_names, output_name = _get_input_output_name(net, node, 'all')
    name = node['name']
    mode = 'CONCAT'
    builder.add_elementwise(name = name, input_names = input_names,
            output_name = output_name, mode = mode)


def convert_deconvolution(net, node, module, builder):
    """Convert a deconvolution layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    module: module
        An module for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    param = node['attr']
    inputs = node['inputs']
    outputs = node['outputs']
    args, aux = module.get_params()

    from ast import literal_eval

    if 'no_bias' in param.keys():
        has_bias = not literal_eval(param['no_bias'])
    else:
        has_bias = False

    if literal_eval(param['pad']) != (0, 0):
        pad = literal_eval(param['pad'])
        builder.add_padding(
            name = name+"_pad", 
            left = pad[1],
            right = pad[1],
            top = pad[0],
            bottom = pad[0],
            value = 0,
            input_name = input_name,
            output_name = name+"_pad_output")
        input_name = name+"_pad_output"

    border_mode = "valid"

    n_filters = int(param['num_filter'])

    output_shape = None
    if 'target_shape' in param:
        target_shape = literal_eval(param['target_shape'])
        output_shape = (int(target_shape[0]), int(target_shape[1]))

    W = args[_get_node_name(net, inputs[1][0])].asnumpy()
    if has_bias:
        Wb = args[_get_node_name(net, inputs[2][0])].asnumpy()
    else:
        Wb = None

    channels = W.shape[0]
    stride_height, stride_width = literal_eval(param['stride'])
    kernel_height, kernel_width = literal_eval(param['kernel'])

    W = W.transpose((2, 3, 0, 1))
    builder.add_convolution(name = name,
             kernel_channels = channels,
             output_channels = n_filters,
             height = kernel_height,
             width = kernel_width,
             stride_height = stride_height,
             stride_width = stride_width,
             border_mode = border_mode,
             groups = 1,
             W = W,
             b = Wb,
             has_bias = has_bias,
             is_deconv = True,
             output_shape = output_shape,
             input_name = input_name,
             output_name = output_name)
