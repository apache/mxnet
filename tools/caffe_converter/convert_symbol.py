from __future__ import print_function
import argparse
import re
import caffe_parser

def _get_input(proto):
    """Get input size
    """
    layer = caffe_parser.get_layers(proto)
    if len(proto.input_dim) > 0:
        input_dim = proto.input_dim
    elif len(proto.input_shape) > 0:
        input_dim = proto.input_shape[0].dim
    elif layer[0].type == "Input":
        input_dim = layer[0].input_param.shape[0].dim
        layer.pop(0)
    else:
        raise ValueError('Cannot find input size')

    assert layer[0].type != "Input", 'only support single input'
    # We assume the first bottom blob of first layer is the output from data layer
    input_name = layer[0].bottom[0]
    return input_name, input_dim, layer

def _convert_conv_param(param):
    """Convert convolution layer parameter from Caffe to MXNet
    """
    pad = 0
    if isinstance(param.pad, int):
        pad = param.pad
    else:
        pad = 0 if len(param.pad) == 0 else param.pad[0]
    stride = 1
    if isinstance(param.stride, int):
        stride = param.stride
    else:
        stride = 1 if len(param.stride) == 0 else param.stride[0]
    kernel_size = ''
    if isinstance(param.kernel_size, int):
        kernel_size = param.kernel_size
    else:
        kernel_size = param.kernel_size[0]
    dilate = 1
    if isinstance(param.dilation, int):
        dilate = param.dilation
    else:
        dilate = 1 if len(param.dilation) == 0 else param.dilation[0]
    # convert to string except for dilation
    param_string = "num_filter=%d, pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d), no_bias=%s" % \
                   (param.num_output, pad, pad, kernel_size, kernel_size, stride, stride, not param.bias_term)
    # deal with dilation. Won't be in deconvolution
    if dilate > 1:
        param_string += ", dilate=(%d, %d)" % (dilate, dilate)
    return param_string

def _convert_pooling_param(param):
    """Convert the pooling layer parameter
    """
    param_string = "pooling_convention='full', "
    if param.global_pooling:
        param_string += "global_pool=True, kernel=(1,1)"
    else:
        param_string += "pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d)" % (
            param.pad, param.pad, param.kernel_size, param.kernel_size,
            param.stride, param.stride)
    if param.pool == 0:
        param_string += ", pool_type='max'"
    elif param.pool == 1:
        param_string += ", pool_type='avg'"
    else:
        raise ValueError("Unknown Pooling Method!")
    return param_string

def _parse_proto(prototxt_fname):
    """Parse Caffe prototxt into symbol string
    """
    proto = caffe_parser.read_prototxt(prototxt_fname)

    # process data layer
    input_name, input_dim, layer = _get_input(proto)
    # only support single input, so always use `data` as the input data
    mapping = {input_name: 'data'}
    need_flatten = {input_name: False}
    symbol_string = "import mxnet as mx\n" \
                    + "data = mx.symbol.Variable(name='data')\n";

    connection = dict()
    symbols = dict()
    top = dict()
    flatten_count = 0
    output_name = ""
    prev_name = None

    # convert reset layers one by one
    for i in range(len(layer)):
        type_string = ''
        param_string = ''
        skip_layer = False
        name = re.sub('[-/]', '_', layer[i].name)
        if layer[i].type == 'Convolution' or layer[i].type == 4:
            type_string = 'mx.symbol.Convolution'
            param_string = _convert_conv_param(layer[i].convolution_param)
            need_flatten[name] = True
        if layer[i].type == 'Deconvolution' or layer[i].type == 39:
            type_string = 'mx.symbol.Deconvolution'
            param_string = _convert_conv_param(layer[i].convolution_param)
            need_flatten[name] = True
        if layer[i].type == 'Pooling' or layer[i].type == 17:
            type_string = 'mx.symbol.Pooling'
            param_string = _convert_pooling_param(layer[i].pooling_param)
            need_flatten[name] = True
        if layer[i].type == 'ReLU' or layer[i].type == 18:
            type_string = 'mx.symbol.Activation'
            param_string = "act_type='relu'"
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'TanH' or layer[i].type == 23:
            type_string = 'mx.symbol.Activation'
            param_string = "act_type='tanh'"
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'Sigmoid' or layer[i].type == 19:
            type_string = 'mx.symbol.Activation'
            param_string = "act_type='sigmoid'"
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'LRN' or layer[i].type == 15:
            type_string = 'mx.symbol.LRN'
            param = layer[i].lrn_param
            param_string = "alpha=%f, beta=%f, knorm=%f, nsize=%d" % (
                param.alpha, param.beta, param.k, param.local_size)
            need_flatten[name] = True
        if layer[i].type == 'InnerProduct' or layer[i].type == 14:
            type_string = 'mx.symbol.FullyConnected'
            param = layer[i].inner_product_param
            param_string = "num_hidden=%d, no_bias=%s" % (
                param.num_output, not param.bias_term)
            need_flatten[name] = False
        if layer[i].type == 'Dropout' or layer[i].type == 6:
            type_string = 'mx.symbol.Dropout'
            param = layer[i].dropout_param
            param_string = "p=%f" % param.dropout_ratio
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'Softmax' or layer[i].type == 20:
            type_string = 'mx.symbol.SoftmaxOutput'
        if layer[i].type == 'Flatten' or layer[i].type == 8:
            type_string = 'mx.symbol.Flatten'
            need_flatten[name] = False
        if layer[i].type == 'Split' or layer[i].type == 22:
            type_string = 'split'  # will process later
        if layer[i].type == 'Concat' or layer[i].type == 3:
            type_string = 'mx.symbol.Concat'
            need_flatten[name] = True
        if layer[i].type == 'Crop':
            type_string = 'mx.symbol.Crop'
            need_flatten[name] = True
            param_string = 'center_crop=True'
        if layer[i].type == 'BatchNorm':
            type_string = 'mx.symbol.BatchNorm'
            param = layer[i].batch_norm_param
            param_string = 'use_global_stats=%s, fix_gamma=False' % param.use_global_stats
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'Scale':
            assert layer[i-1].type == 'BatchNorm'
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
            skip_layer = True
            prev_name = re.sub('[-/]', '_', layer[i-1].name)
        if layer[i].type == 'PReLU':
            type_string = 'mx.symbol.LeakyReLU'
            param = layer[i].prelu_param
            param_string = "act_type='prelu', slope=%f" % param.filler.value
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'Eltwise':
            type_string = 'mx.symbol.broadcast_add'
            param_string = ""
            need_flatten[name] = False
        if layer[i].type == 'Reshape':
            type_string = 'mx.symbol.Reshape'
            need_flatten[name] = False
            param = layer[i].reshape_param
            param_string = "shape=(%s)" % (','.join(param.shape.dim),)
        if layer[i].type == 'AbsVal': 
        	type_string = 'mx.symbol.abs' 
        	need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]] 

        if skip_layer:
            assert len(layer[i].bottom) == 1
            symbol_string += "%s = %s\n" % (name, prev_name)
        elif type_string == '':
            raise ValueError('Unknown layer %s!' % layer[i].type)
        elif type_string != 'split':
            bottom = layer[i].bottom
            if param_string != "":
                param_string = ", " + param_string
            if len(bottom) == 1:
                if need_flatten[mapping[bottom[0]]] and type_string == 'mx.symbol.FullyConnected':
                    flatten_name = "flatten_%d" % flatten_count
                    symbol_string += "%s=mx.symbol.Flatten(name='%s', data=%s)\n" % (
                        flatten_name, flatten_name, mapping[bottom[0]])
                    flatten_count += 1
                    need_flatten[flatten_name] = False
                    bottom[0] = flatten_name
                    mapping[bottom[0]] = bottom[0]
                symbol_string += "%s = %s(name='%s', data=%s %s)\n" % (
                    name, type_string, name, mapping[bottom[0]], param_string)
            else:
                symbol_string += "%s = %s(name='%s', *[%s] %s)\n" % (
                    name, type_string, name, ','.join([mapping[x] for x in bottom]), param_string)
        for j in range(len(layer[i].top)):
            mapping[layer[i].top[j]] = name
        output_name = name
    return symbol_string, output_name, input_dim

def convert_symbol(prototxt_fname):
    """Convert caffe model definition into Symbol

    Parameters
    ----------
    prototxt_fname : str
        Filename of the prototxt file

    Returns
    -------
    Symbol
        Converted Symbol
    tuple
        Input shape
    """
    sym, output_name, input_dim = _parse_proto(prototxt_fname)
    exec(sym)
    _locals = locals()
    exec("ret = " + output_name, globals(), _locals)
    ret = _locals['ret']
    return ret, input_dim

def main():
    parser = argparse.ArgumentParser(
        description='Convert caffe prototxt into Symbol')
    parser.add_argument('prototxt', help='The prototxt filename')
    parser.add_argument('output', help='filename for the output json file')
    args = parser.parse_args()

    sym, _ = convert_symbol(args.prototxt)
    sym.save(args.output)

if __name__ == '__main__':
    main()
