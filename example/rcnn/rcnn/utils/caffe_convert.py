# This script will not work unless all paths are set right

import os
import sys
import mxnet as mx
import numpy as np
fast_rcnn_path = None
sys.path.insert(0, os.path.join(fast_rcnn_path, 'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(fast_rcnn_path, 'lib'))
import caffe
from rcnn.symbol import get_symbol_vgg_test

def load_model(caffeproto, caffemodel, arg_shape_dic):
    def get_caffe_iter(layer_names, layers):
        for layer_idx, layer in enumerate(layers):
            layer_name = layer_names[layer_idx].replace('/', '_')
            layer_type = layer.type
            layer_blobs = layer.blobs
            yield (layer_name, layer_type, layer_blobs)

    net_caffe = caffe.Net(caffeproto, caffemodel, caffe.TEST)
    layer_names = net_caffe._layer_names
    layers = net_caffe.layers
    iter = ''
    iter = get_caffe_iter(layer_names, layers)
    first_conv = True

    arg_params = {}
    for layer_name, layer_type, layer_blobs in iter:
        if layer_type == 'Convolution' or layer_type == 'InnerProduct' or layer_type == 4 or layer_type == 14:
            assert(len(layer_blobs) == 2)
            wmat = np.array(layer_blobs[0].data).reshape(layer_blobs[0].num, layer_blobs[0].channels, layer_blobs[0].height, layer_blobs[0].width)
            bias = np.array(layer_blobs[1].data)
            if first_conv:
                print 'Swapping BGR of caffe into RGB in mxnet'
                wmat[:, [0, 2], :, :] = wmat[:, [2, 0], :, :]

            assert(wmat.flags['C_CONTIGUOUS'] is True)
            assert(bias.flags['C_CONTIGUOUS'] is True)
            print 'converting layer {0}, wmat shape = {1}, bias shape = {2}'.format(layer_name, wmat.shape, bias.shape)
            wmat = wmat.reshape((wmat.shape[0], -1))
            bias = bias.reshape((bias.shape[0], 1))
            weight_name = layer_name + "_weight"
            bias_name = layer_name + "_bias"
            
            if weight_name not in arg_shape_dic:
                print weight_name + ' not found in arg_shape_dic.'
                continue
            wmat = wmat.reshape(arg_shape_dic[weight_name])
            arg_params[weight_name] = mx.nd.zeros(wmat.shape)
            arg_params[weight_name][:] = wmat

            bias = bias.reshape(arg_shape_dic[bias_name])
            arg_params[bias_name] = mx.nd.zeros(bias.shape)
            arg_params[bias_name][:] = bias

            if first_conv and (layer_type == 'Convolution' or layer_type == 4):
                first_conv = False
    
    return arg_params

proto_path = os.path.join(fast_rcnn_path, 'models', 'VGG16', 'test.prototxt')
model_path = os.path.join(fast_rcnn_path, 'data', 'fast_rcnn_models', 'vgg16_fast_rcnn_iter_40000.caffemodel')

symbol = get_symbol_vgg_test()
arg_shapes, out_shapes, aux_shapes = symbol.infer_shape(**{'data': (1, 3, 224, 224), 'rois': (1, 5)})
arg_shape_dic = { name: shape for name, shape in zip(symbol.list_arguments(), arg_shapes) }

arg_params = load_model(proto_path, model_path, arg_shape_dic)

model = mx.model.FeedForward(ctx=mx.cpu(), symbol=symbol, arg_params=arg_params,
                             aux_params={}, num_epoch=1,
                             learning_rate=0.01, momentum=0.9, wd=0.0001)
model.save('model/ref')
