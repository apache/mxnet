# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
    """
    Convert convolution layer parameter from Caffe to MXNet
    """
    param_string = "num_filter=%d" % param.num_output

    pad_w = 0
    pad_h = 0
    if isinstance(param.pad, int):
        pad = param.pad
        param_string += ", pad=(%d, %d)" % (pad, pad)
    else:
        if len(param.pad) > 0:
            pad = param.pad[0]
            param_string += ", pad=(%d, %d)" % (pad, pad)
        else:
            if isinstance(param.pad_w, int):
                pad_w = param.pad_w
            if isinstance(param.pad_h, int):
                pad_h = param.pad_h
            param_string += ", pad=(%d, %d)" % (pad_h, pad_w)

    if isinstance(param.kernel_size, int):
        kernel_size = param.kernel_size
        param_string += ", kernel=(%d,%d)" % (kernel_size, kernel_size)
    else:
        if len(param.kernel_size) > 0:
            kernel_size = param.kernel_size[0]
            param_string += ", kernel=(%d,%d)" % (kernel_size, kernel_size)
        else:
            assert isinstance(param.kernel_w, int)
            kernel_w = param.kernel_w
            assert isinstance(param.kernel_h, int)
            kernel_h = param.kernel_h
            param_string += ", kernel=(%d,%d)" % (kernel_h, kernel_w)

    stride = 1
    if isinstance(param.stride, int):
        stride = param.stride
    else:
        stride = 1 if len(param.stride) == 0 else param.stride[0]

    param_string += ", stride=(%d,%d)" % (stride, stride)

    dilate = 1
    if hasattr(param, 'dilation'):
        if isinstance(param.dilation, int):
            dilate = param.dilation
        else:
            dilate = 1 if len(param.dilation) == 0 else param.dilation[0]

    param_string += ", no_bias=%s" % (not param.bias_term)

    # deal with dilation. Won't be in deconvolution
    if dilate > 1:
        param_string += ", dilate=(%d, %d)" % (dilate, dilate)

    if isinstance(param.group, int):
        if param.group != 1:
            param_string += ", num_group=%d" % param.group

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

def _find_layer(layers, name):
    for layer in layers:
        if layer.name == name:
            return layer
    return None

def _parse_proto(prototxt_fname):
    """Parse Caffe prototxt into symbol string
    """
    proto = caffe_parser.read_prototxt(prototxt_fname)

    # process data layer
    input_name, input_dim, layers = _get_input(proto)
    # only support single input, so always use `data` as the input data
    mapping = {input_name: 'data'}
    need_flatten = {input_name: False}
    symbol_string = "import mxnet as mx\ndata = mx.symbol.Variable(name='data')\n"

    flatten_count = 0
    output_name = ""
    prev_name = None

    # convert reset layers one by one
    for i, layer in enumerate(layers):
        type_string = ''
        param_string = ''
        skip_layer = False
        bottom_order = []
        name = re.sub('[-/]', '_', layer.name)
        if layer.type == 'Convolution' or layer.type == 4:
            type_string = 'mx.symbol.Convolution'
            param_string = _convert_conv_param(layer.convolution_param)
            need_flatten[name] = True
        if layer.type == 'Deconvolution' or layer.type == 39:
            type_string = 'mx.symbol.Deconvolution'
            param_string = _convert_conv_param(layer.convolution_param)
            need_flatten[name] = True
        if layer.type == 'Pooling' or layer.type == 17:
            type_string = 'mx.symbol.Pooling'
            param_string = _convert_pooling_param(layer.pooling_param)
            need_flatten[name] = True
        if layer.type == 'ReLU' or layer.type == 18:
            type_string = 'mx.symbol.Activation'
            param_string = "act_type='relu'"
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
        if layer.type == 'TanH' or layer.type == 23:
            type_string = 'mx.symbol.Activation'
            param_string = "act_type='tanh'"
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
        if layer.type == 'Sigmoid' or layer.type == 19:
            type_string = 'mx.symbol.Activation'
            param_string = "act_type='sigmoid'"
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
        if layer.type == 'LRN' or layer.type == 15:
            type_string = 'mx.symbol.LRN'
            param = layer.lrn_param
            param_string = "alpha=%f, beta=%f, knorm=%f, nsize=%d" % (
                param.alpha, param.beta, param.k, param.local_size)
            need_flatten[name] = True
        if layer.type == 'InnerProduct' or layer.type == 14:
            type_string = 'mx.symbol.FullyConnected'
            param = layer.inner_product_param
            param_string = "num_hidden=%d, no_bias=%s" % (
                param.num_output, not param.bias_term)
            need_flatten[name] = False
        if layer.type == 'Dropout' or layer.type == 6:
            type_string = 'mx.symbol.Dropout'
            param = layer.dropout_param
            param_string = "p=%f" % param.dropout_ratio
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
        if layer.type == 'Softmax' or layer.type == 20:
            if layer.softmax_param.axis == 2:
                symbol_string += "%s = mx.symbol.transpose(%s, axes=(0,2,1))\n" %\
                    (mapping[layer.bottom[0]], mapping[layer.bottom[0]])
                type_string = 'mx.symbol.SoftmaxActivation'
                param_string = "mode='channel'"
                need_flatten[name] = False
            else:
                type_string = 'mx.symbol.SoftmaxOutput'
        if layer.type == 'Flatten' or layer.type == 8:
            if 'softmax' in layer.bottom[0]:
                prev_name = re.sub('[-/]', '_', layers[i-1].name)
                skip_layer = True
            else:
                type_string = 'mx.symbol.Flatten'
            need_flatten[name] = False
        if layer.type == 'Split' or layer.type == 22:
            type_string = 'split'  # will process later
        if layer.type == 'Concat' or layer.type == 3:
            type_string = 'mx.symbol.Concat'
            need_flatten[name] = True
        if layer.type == 'Crop':
            type_string = 'mx.symbol.Crop'
            need_flatten[name] = True
            param_string = 'center_crop=True'
        if layer.type == 'BatchNorm':
            type_string = 'mx.symbol.BatchNorm'
            param = layer.batch_norm_param
            # CuDNN requires eps to be greater than 1e-05
            # We compensate for this change in convert_model
            epsilon = param.eps
            if (epsilon <= 1e-05):
                epsilon = 1e-04
            # if next layer is scale, don't fix gamma
            fix_gamma = layers[i+1].type != 'Scale'
            param_string = 'use_global_stats=%s, fix_gamma=%s, eps=%f' % (
                param.use_global_stats, fix_gamma, epsilon)
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
        if layer.type == 'Scale':
            assert layers[i-1].type == 'BatchNorm'
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
            skip_layer = True
            prev_name = re.sub('[-/]', '_', layers[i-1].name)
        if layer.type == 'PReLU':
            type_string = 'mx.symbol.LeakyReLU'
            param = layer.prelu_param
            param_string = "act_type='prelu', slope=%f" % param.filler.value
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
        if layer.type == 'Eltwise':
            type_string = 'mx.symbol.broadcast_add'
            param_string = ""
            need_flatten[name] = False
        if layer.type == 'Reshape':
            type_string = 'mx.symbol.Reshape'
            param = layer.reshape_param
            param_string = 'shape=(' + ','.join([str(x) for x in list(param.shape.dim)]) + ')'
            need_flatten[name] = True
        if layer.type == 'AbsVal':
            type_string = 'mx.symbol.abs'
            need_flatten[name] = need_flatten[mapping[layer.bottom[0]]]
        if layer.type == 'Normalize':
            bottom = re.sub('[-/]', '_', layer.bottom[0])
            conv_layer = _find_layer(layers, bottom)
            assert conv_layer is not None
            param = layer.norm_param
            assert not param.across_spatial and not param.channel_shared
            assert param.scale_filler.type == 'constant'
            if conv_layer.type == 'Convolution':
                scale_name = "%s_scale" % name
                symbol_string += "%s=mx.sym.Variable(name='%s', shape=(1, %d, 1, 1), init=mx.init.Constant(%f))\n" % \
                    (scale_name, scale_name, conv_layer.convolution_param.num_output,
                    param.scale_filler.value)
                symbol_string += "%s=mx.symbol.L2Normalization(name='%s', data=%s, mode='channel')\n" %\
                    (name, name, mapping[layer.bottom[0]])
                symbol_string += "%s=mx.symbol.broadcast_mul(lhs=%s, rhs=%s)\n" %\
                    (name, scale_name, name)
                type_string = 'split'
                need_flatten[name] = True
            else:
                raise ValueError('Unknown/Invalid normalize layer!')
        if layer.type == 'Permute':
            type_string = 'mx.symbol.transpose'
            param_string = "axes=(%s)" % (','.join([str(x) for x in layer.permute_param.order]))
            need_flatten[name] = True
            from_name = ''
        if layer.type == 'PriorBox':
            param = layer.prior_box_param
            if layer.bottom[0] == 'data':
                bottom_order = [1]
            else:
                bottom_order = [0]
            try:
                import math
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
            finput_dimh = float(input_dim[2])
            finput_dimw = float(input_dim[3])
            step = '(%f, %f)' % (step_h / finput_dimh, step_w / finput_dimw)
            assert param.offset == 0.5, "currently only support offset = 0.5"
            symbol_string += '%s = mx.contrib.symbol.MultiBoxPrior(%s, sizes=%s, ratios=%s, clip=%s, steps=%s, name="%s")\n' % \
                (name, mapping[layer.bottom[0]], sizes, ratios_string, clip, step, name)
            symbol_string += '%s = mx.symbol.Flatten(data=%s)\n' % (name, name)
            type_string = 'split'
            need_flatten[name] = False
        if layer.type == 'DetectionOutput':
            bottom_order = [1, 0, 2]
            param = layer.detection_output_param
            assert param.share_location == True
            assert param.background_label_id == 0
            nms_param = param.nms_param
            type_string = 'mx.contrib.symbol.MultiBoxDetection'
            param_string = "nms_threshold=%f, nms_topk=%d, clip=False" % \
                (nms_param.nms_threshold, nms_param.top_k)
        if skip_layer:
            assert len(layer.bottom) == 1
            symbol_string += "%s = %s\n" % (name, prev_name)
        elif type_string == '':
            raise ValueError('Unknown layer %s!' % layer.type)
        elif type_string != 'split':
            bottom = layer.bottom
            if param_string != "":
                param_string = ", " + param_string
            if len(bottom) == 1:
                # print(need_flatten)
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
                if not bottom_order:
                    bottom_order = range(len(bottom))
                symbol_string += "%s = %s(name='%s', *[%s] %s)\n" % \
                                 (name, type_string, name, ','.join([mapping[bottom[x]] for x in bottom_order]), param_string)
                if layer.type == 'Concat' and layer.concat_param.axis == 2:
                    symbol_string += "%s = mx.symbol.Reshape(data=%s, shape=(0, -1, 4), name='%s')\n" %\
                        (name, name, name)
        for j in range(len(layer.top)):
            mapping[layer.top[j]] = name
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
    exec(sym)                   # pylint: disable=exec-used
    _locals = locals()
    exec("ret = " + output_name, globals(), _locals)  # pylint: disable=exec-used
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
