from __future__ import print_function
import mxnet as mx
import numpy as np
import argparse
import re
from convert_symbol import proto2symbol

caffe_flag = True
try:
    import caffe
except ImportError:
    import caffe_parse.parse_from_protobuf as parse

    caffe_flag = False


def get_caffe_iter(layer_names, layers):
    for layer_idx, layer in enumerate(layers):
        layer_name = re.sub('[-/]', '_', layer_names[layer_idx])
        layer_type = layer.type
        layer_blobs = layer.blobs
        yield (layer_name, layer_type, layer_blobs)


def get_iter(layers):
    for layer in layers:
        layer_name = re.sub('[-/]', '_', layer.name)
        layer_type = layer.type
        layer_blobs = layer.blobs
        yield (layer_name, layer_type, layer_blobs)


def main():
    parser = argparse.ArgumentParser(description='Caffe prototxt to mxnet model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument('caffe_prototxt', help='The prototxt file in Caffe format')
    parser.add_argument('caffe_model', help='The binary model parameter file in Caffe format')
    parser.add_argument('save_model_name', help='The name of the output model prefix')
    args = parser.parse_args()

    prob, input_dim = proto2symbol(args.caffe_prototxt)

    layers = ''
    layer_names = ''

    if caffe_flag:
        caffe.set_mode_cpu()
        net_caffe = caffe.Net(args.caffe_prototxt, args.caffe_model, caffe.TEST)
        layer_names = net_caffe._layer_names
        layers = net_caffe.layers
    else:
        layers = parse.parse_caffemodel(args.caffe_model)

    arg_shapes, output_shapes, aux_shapes = prob.infer_shape(data=tuple(input_dim))
    arg_names = prob.list_arguments()
    arg_shape_dic = dict(zip(arg_names, arg_shapes))
    arg_params = {}

    iter = ''
    if caffe_flag:
        iter = get_caffe_iter(layer_names, layers)
    else:
        iter = get_iter(layers)
    first_conv = True

    for layer_name, layer_type, layer_blobs in iter:
        if layer_type == 'Convolution' or layer_type == 'InnerProduct' or layer_type == 4 or layer_type == 14 \
                or layer_type == 'PReLU':
            if layer_type == 'PReLU':
                assert (len(layer_blobs) == 1)
                wmat = layer_blobs[0].data
                weight_name = layer_name + '_gamma'
                arg_params[weight_name] = mx.nd.zeros(wmat.shape)
                arg_params[weight_name][:] = wmat
                continue
            assert (len(layer_blobs) == 2)
            wmat_dim = []
            if getattr(layer_blobs[0].shape, 'dim', None) is not None:
                if len(layer_blobs[0].shape.dim) > 0:
                    wmat_dim = layer_blobs[0].shape.dim
                else:
                    wmat_dim = [layer_blobs[0].num, layer_blobs[0].channels, layer_blobs[0].height,
                                layer_blobs[0].width]
            else:
                wmat_dim = list(layer_blobs[0].shape)
            wmat = np.array(layer_blobs[0].data).reshape(wmat_dim)
            bias = np.array(layer_blobs[1].data)
            channels = wmat_dim[1]
            if channels == 3 or channels == 4:  # RGB or RGBA
                if first_conv:
                    print('Swapping BGR of caffe into RGB in mxnet')
                    wmat[:, [0, 2], :, :] = wmat[:, [2, 0], :, :]

            assert (wmat.flags['C_CONTIGUOUS'] is True)
            assert (bias.flags['C_CONTIGUOUS'] is True)
            print('converting layer {0}, wmat shape = {1}, bias shape = {2}'.format(layer_name, wmat.shape, bias.shape))
            wmat = wmat.reshape((wmat.shape[0], -1))
            bias = bias.reshape((bias.shape[0], 1))
            weight_name = layer_name + "_weight"
            bias_name = layer_name + "_bias"

            if weight_name not in arg_shape_dic:
                print(weight_name + ' not found in arg_shape_dic.')
                continue
            wmat = wmat.reshape(arg_shape_dic[weight_name])
            arg_params[weight_name] = mx.nd.zeros(wmat.shape)
            arg_params[weight_name][:] = wmat

            bias = bias.reshape(arg_shape_dic[bias_name])
            arg_params[bias_name] = mx.nd.zeros(bias.shape)
            arg_params[bias_name][:] = bias

            if first_conv and (layer_type == 'Convolution' or layer_type == 4):
                first_conv = False

    model = mx.mod.Module(symbol=prob, label_names=['prob_label', ])
    model.bind(data_shapes=[('data', tuple(input_dim))])
    model.init_params(arg_params=arg_params, aux_params={})

    model.save_checkpoint(args.save_model_name, 1)


if __name__ == '__main__':
    main()
