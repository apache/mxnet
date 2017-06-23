import numpy as _np

def _get_input_output_name(net, node, index = 0):
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

def convert_transpose(net, node, model, builder):
    """Convert a transpose layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    param = node['attr']
    from ast import literal_eval
    axes = literal_eval(param['axes'])
    builder.add_permute(name, input_name, output_name, axes)

def convert_flatten(net, node, model, builder):
    """Convert a flatten layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    builder.add_flatten(0, name, input_name, output_name)

def convert_softmax(net, node, model, builder):
    """Convert a softmax layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    builder.add_softmax(name = name,
                        input_name = input_name,
                        output_name = output_name)

def convert_activation(net, node, model, builder):
    """Convert an activation layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    mx_non_linearity = node['attr']['act_type']
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

def convert_elementwise_add(net, node, model, builder):
    """Convert an elementwise add layer from mxnet to coreml.

        Parameters
        ----------
        network: net
        A mxnet network object.

        layer: node
        Node to convert.

        model: model
        An model for MXNet

        builder: NeuralNetworkBuilder
        A neural network builder object.
        """

    input_names, output_name = _get_input_output_name(net, node,[0,1])
    name = node['name']

    builder.add_elementwise(name, input_names, output_name, 'ADD')

def convert_dense(net, node, model, builder):
    """Convert a dense layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    param = node['attr']
    has_bias = True
    name = node['name']

    inputs = node['inputs']
    outputs = node['outputs']
    args = model.arg_params
    W = args[_get_node_name(net, inputs[1][0])].asnumpy()
    if has_bias:
        Wb = args[_get_node_name(net, inputs[2][0])].asnumpy()
    else:
        Wb = None
    nC, nB = W.shape

    builder.add_inner_product(name = name,
            W = W,
            Wb = Wb,
            nB = nB,
            nC = nC,
            has_bias = has_bias,
            input_name = input_name,
            output_name = output_name)

def convert_convolution(net, node, model, builder):
    """Convert a convolution layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    param = node['attr']
    inputs = node['inputs']
    outputs = node['outputs']
    args = model.arg_params

    from ast import literal_eval

    if 'no_bias' in param.keys():
        has_bias = not literal_eval(param['no_bias'])
    else:
        has_bias = True

    border_mode = "same" if literal_eval(param['pad']) != (0, 0) else 'valid'
    border_mode = "valid"
    n_filters = int(param['num_filter'])
    output_shape = None  # (needed for de-conv)

    W = args[_get_node_name(net, inputs[1][0])].asnumpy()
    if has_bias:
        Wb = args[_get_node_name(net, inputs[2][0])].asnumpy()
    else:
        Wb = None

    n_filters, channels = W.shape[0:2]
    stride_height, stride_width = literal_eval(param['stride'])
    kernel_height, kernel_width = literal_eval(param['kernel'])

    W = W.transpose((2, 3, 1, 0))
    builder.add_convolution(name = name,
             kernelChannels = channels,
             outputChannels = n_filters,
             height = kernel_height,
             width = kernel_width,
             stride_height = stride_height,
             stride_width = stride_width,
             borderMode = border_mode,
             groups = 1,
             W = W,
             b = Wb,
             has_bias = has_bias,
             is_deconv = False,
             output_shape = output_shape,
             input_name = input_name,
             output_name = output_name)

    # Add padding if there is any
    convLayer = builder.nn_spec.layers[-1].convolution
    pad = literal_eval(param['pad'])
    for i in range(len(pad)):
        convLayer.valid.paddingAmounts.borderAmounts[i].startEdgeSize = pad[i]
        convLayer.valid.paddingAmounts.borderAmounts[i].endEdgeSize = pad[i]

def convert_pooling(net, node, model, builder):
    """Convert a pooling layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    inputs = node['inputs']
    param = node['attr']
    outputs = node['outputs']
    args = model.arg_params

    layer_type_mx = param['pool_type']
    if layer_type_mx == 'max':
        layer_type= 'MAX'
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
    builder.add_pooling(name = name,
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

def convert_batchnorm(net, node, model, builder):
    """Convert a transpose layer from mxnet to coreml.

    Parameters
    ----------
    network: net
        A mxnet network object.

    layer: node
        Node to convert.

    model: model
        An model for MXNet

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = _get_input_output_name(net, node)
    name = node['name']
    param = node['attr']
    inputs = node['inputs']
    outputs = node['outputs']
    args = model.arg_params
    aux = model.aux_params

    gamma = args[_get_node_name(net, inputs[1][0])].asnumpy()
    beta = args[_get_node_name(net, inputs[2][0])].asnumpy()
    mean = aux[_get_node_name(net, inputs[3][0])].asnumpy()
    variance = aux[_get_node_name(net, inputs[4][0])].asnumpy()

    nb_channels = gamma.shape[0]

    builder.add_batchnorm(
        name = name,
        channels = nb_channels,
        gamma = gamma,
        beta = beta,
        mean = mean,
        variance = variance,
        input_name = input_name,
        output_name = output_name)

def convert_concat(net, node, model, builder):
    """Convert concat layer from mxnet to coreml.

    Parameters
    ----------
    network: net
    A mxnet network object.

    layer: node
    Node to convert.

    model: model
    An model for MXNet

    builder: NeuralNetworkBuilder
    A neural network builder object.
    """
    # Get input and output names
    input_names, output_name = _get_input_output_name(net, node, 'all')
    name = node['name']
    mode = 'CONCAT'
    builder.add_elementwise(name = name, input_names = input_names,
            output_name = output_name, mode = mode)
