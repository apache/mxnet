from __future__ import print_function
from google.protobuf import text_format
import argparse
import re
import sys
import math

caffe_flag = True
try:
    import caffe
    from caffe.proto import caffe_pb2
except ImportError:
    caffe_flag = False
    import caffe_parse.caffe_pb2


def read_proto_solver_file(file_path):
    solver_config = ''
    if caffe_flag:
        solver_config = caffe.proto.caffe_pb2.NetParameter()
    else:
        solver_config = caffe_parse.caffe_pb2.NetParameter()
    return read_proto_file(file_path, solver_config)


def read_proto_file(file_path, parser_object):
    file = open(file_path, "r")
    if not file:
        raise Exception("ERROR (" + file_path + ")!")
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object


def conv_param_to_string(param):
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

def find_layer(layers, name):
    for layer in layers:
        if layer.name == name:
            return layer
    return None

def proto2script(proto_file):
    proto = read_proto_solver_file(proto_file)
    connection = dict()
    symbols = dict()
    top = dict()
    flatten_count = 0
    symbol_string = ""
    layer = ''
    if len(proto.layer):
        layer = proto.layer
    elif len(proto.layers):
        layer = proto.layers
    else:
        raise Exception('Invalid proto file.')
        # Get input size to network
    input_dim = [1, 3, 224, 224]  # default
    if len(proto.input_dim) > 0:
        input_dim = proto.input_dim
    elif len(proto.input_shape) > 0:
        input_dim = proto.input_shape[0].dim
    elif layer[0].type == "Input":
        input_dim = layer[0].input_param.shape._values[0].dim
        layer.pop(0)
    else:
        raise Exception('Invalid proto file.')
        # We assume the first bottom blob of first layer is the output from data layer
    input_name = layer[0].bottom[0]
    output_name = ""
    mapping = {input_name: 'data'}
    need_flatten = {input_name: False}
    for i in range(len(layer)):
        type_string = ''
        param_string = ''
        name = re.sub('[-/]', '_', layer[i].name)
        from_name = 'data='
        bottom_order = []
        if layer[i].type == 'Convolution' or layer[i].type == 4:
            type_string = 'mx.symbol.Convolution'
            param_string = conv_param_to_string(layer[i].convolution_param)
            need_flatten[name] = True
        if layer[i].type == 'Deconvolution' or layer[i].type == 39:
            type_string = 'mx.symbol.Deconvolution'
            param_string = conv_param_to_string(layer[i].convolution_param)
            need_flatten[name] = True
        if layer[i].type == 'Pooling' or layer[i].type == 17:
            type_string = 'mx.symbol.Pooling'
            param = layer[i].pooling_param
            param_string = ''
            param_string += "pooling_convention='full', "
            if param.global_pooling:
                # there must be a param `kernel` in a pooling layer
                param_string += "global_pool=True, kernel=(1,1)"
            else:
                param_string += "pad=(%d,%d), kernel=(%d,%d), stride=(%d,%d)" % \
                                (param.pad, param.pad, param.kernel_size, param.kernel_size, param.stride, param.stride)
            if param.pool == 0:
                param_string += ", pool_type='max'"
            elif param.pool == 1:
                param_string += ", pool_type='avg'"
            else:
                raise Exception("Unknown Pooling Method!")
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
            param_string = "alpha=%f, beta=%f, knorm=%f, nsize=%d" % \
                           (param.alpha, param.beta, param.k, param.local_size)
            need_flatten[name] = True
        if layer[i].type == 'InnerProduct' or layer[i].type == 14:
            type_string = 'mx.symbol.FullyConnected'
            param = layer[i].inner_product_param
            param_string = "num_hidden=%d, no_bias=%s" % (param.num_output, not param.bias_term)
            need_flatten[name] = False
        if layer[i].type == 'Dropout' or layer[i].type == 6:
            type_string = 'mx.symbol.Dropout'
            param = layer[i].dropout_param
            param_string = "p=%f" % param.dropout_ratio
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'Softmax' or layer[i].type == 20:
            if layer[i].softmax_param.axis == 2:
                symbol_string += "%s = mx.symbol.transpose(%s, axes=(0,2,1))\n" %\
                    (mapping[layer[i].bottom[0]], mapping[layer[i].bottom[0]])
                type_string = 'mx.symbol.SoftmaxActivation'
                param_string = "mode='channel'"
                need_flatten[name] = False
            else:
                type_string = 'mx.symbol.SoftmaxOutput'
        if layer[i].type == 'Flatten' or layer[i].type == 8:
            if 'softmax' in layer[i].bottom[0]:
                type_string = 'identical'
            else:
                type_string = 'mx.symbol.Flatten'
            need_flatten[name] = False
        if layer[i].type == 'Split' or layer[i].type == 22:
            type_string = 'split'
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
            param_string = 'use_global_stats=%s' % param.use_global_stats
        if layer[i].type == 'PReLU':
            type_string = 'mx.symbol.LeakyReLU'
            param = layer[i].prelu_param
            param_string = "act_type='prelu', slope=%f" % param.filler.value
            need_flatten[name] = need_flatten[mapping[layer[i].bottom[0]]]
        if layer[i].type == 'Normalize':
            bottom = re.sub('[-/]', '_', layer[i].bottom[0])
            conv_layer = find_layer(layer, bottom)
            assert conv_layer is not None
            param = layer[i].norm_param
            assert not param.across_spatial and not param.channel_shared
            assert param.scale_filler.type == 'constant'
            if conv_layer.type == 'Convolution':
                scale_name = "%s_scale" % name
                symbol_string += "%s=mx.sym.Variable(name='%s', shape=(1, %d, 1, 1), init=mx.init.Constant(%f))\n" % \
                    (scale_name, scale_name, conv_layer.convolution_param.num_output,
                    param.scale_filler.value)
                symbol_string += "%s=mx.symbol.L2Normalization(name='%s', data=%s, mode='channel')\n" %\
                    (name, name, mapping[layer[i].bottom[0]])
                symbol_string += "%s=mx.symbol.broadcast_mul(lhs=%s, rhs=%s)\n" %\
                    (name, scale_name, name)
                type_string = 'split'
                need_flatten[name] = True
            else:
                raise ValueError('Unknown/Invalid normalize layer!')
        if layer[i].type == 'Permute':
            type_string = 'mx.symbol.transpose'
            param_string = "axes=(%s)" % (','.join([str(x) for x in layer[i].permute_param.order]))
            need_flatten[name] = True
            from_name = ''
        if layer[i].type == 'PriorBox':
            param = layer[i].prior_box_param
            if layer[i].bottom[0] == 'data':
                bottom_order = [1]
            else:
                bottom_order = [0]
            try:
                min_size = param.min_size[0] / input_dim[2]
                max_size = math.sqrt(param.min_size[0] * param.max_size[0]) / input_dim[2]
                sizes = '(%f, %f)' %(min_size, max_size)
            except AttributeError:
                min_size = param.min_size[0] / input_dim[2]
                sizes = '(%f)' %(min_size)
            ars = list(param.aspect_ratio)
            ratios = [1.]
            for ar in ars:
                ratios.append(ar)
                if param.flip:
                    ratios.append(1. / ar)
            ratios_string = '(' + ','.join(str(x) for x in ratios) + ')'
            clip = param.clip
            if (param.step_h > 0 or param.step_w > 0):
                step_h = param.step_h
                step_w = param.step_w
            elif param.step > 0:
                step_h = param.step
                step_w = param.step
            else:
                step_h = -1
                step_w = -1
            finput_dim = float(input_dim[2])
            step = '(%f, %f)' % (step_h / finput_dim, step_w / finput_dim)
            assert param.offset == 0.5, "currently only support offset = 0.5"
            symbol_string += '%s = mx.contrib.symbol.MultiBoxPrior(%s, sizes=%s, ratios=%s, clip=%s, steps=%s, name="%s")\n' % \
                (name, mapping[layer[i].bottom[0]], sizes, ratios_string, clip, step, name)
            symbol_string += '%s = mx.symbol.Flatten(data=%s)\n' % (name, name)
            type_string = 'split'
            need_flatten[name] = False
        if layer[i].type == 'Reshape':
            type_string = 'mx.symbol.Reshape'
            param = layer[i].reshape_param
            param_string = 'shape=(' + ','.join([str(x) for x in list(param.shape.dim)]) + ')'
            need_flatten[name] = True
        if layer[i].type == 'DetectionOutput':
            bottom_order = [1, 0, 2]
            param = layer[i].detection_output_param
            assert param.share_location == True
            assert param.background_label_id == 0
            nms_param = param.nms_param
            type_string = 'mx.contrib.symbol.MultiBoxDetection'
            param_string = "nms_threshold=%f, nms_topk=%d" % \
                (nms_param.nms_threshold, nms_param.top_k)
        if type_string == '':
            raise Exception('Unknown Layer %s!' % layer[i].type)
        if type_string == 'identical':
            bottom = layer[i].bottom
            symbol_string += "%s = %s\n" % (name, mapping[bottom[0]])
        elif type_string != 'split':
            bottom = layer[i].bottom
            if param_string != "":
                param_string = ", " + param_string
            if len(bottom) == 1:
                if need_flatten[mapping[bottom[0]]] and type_string == 'mx.symbol.FullyConnected':
                    flatten_name = "flatten_%d" % flatten_count
                    symbol_string += "%s=mx.symbol.Flatten(name='%s', data=%s)\n" % \
                                     (flatten_name, flatten_name, mapping[bottom[0]])
                    flatten_count += 1
                    need_flatten[flatten_name] = False
                    bottom[0] = flatten_name
                    mapping[bottom[0]] = bottom[0]
                symbol_string += "%s = %s(%s%s %s, name='%s')\n" % \
                                 (name, type_string, from_name, mapping[bottom[0]], param_string, name)
            else:
                if not bottom_order:
                    bottom_order = range(len(bottom))
                symbol_string += "%s = %s(name='%s', *[%s] %s)\n" % \
                                 (name, type_string, name, ','.join([mapping[bottom[x]] for x in bottom_order]), param_string)
                if layer[i].type == 'Concat' and layer[i].concat_param.axis == 2:
                    symbol_string += "%s = mx.symbol.Reshape(data=%s, shape=(0, -1, 4), name='%s')\n" %\
                        (name, name, name)
        for j in range(len(layer[i].top)):
            mapping[layer[i].top[j]] = name
        output_name = name
    return symbol_string, output_name, input_dim


def proto2symbol(proto_file):
    sym, output_name, input_dim = proto2script(proto_file)
    sym = "import mxnet as mx\n" \
          + "data = mx.symbol.Variable(name='data')\n" \
          + sym
    exec(sym)
    _locals = locals()
    exec("ret = " + output_name, globals(), _locals)
    ret = _locals['ret']
    return ret, input_dim


def main():
    symbol_string, output_name, input_dim = proto2script(sys.argv[1])
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as fout:
            fout.write(symbol_string)
    else:
        print(symbol_string)


if __name__ == '__main__':
    main()
