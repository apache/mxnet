import mxnet as mx
import caffe
import argparse
from convert_symbol import proto2symbol

def main():
    parser = argparse.ArgumentParser(description='Caffe prototxt to mxnet model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument('caffe_prototxt', help='The prototxt file in Caffe format')
    parser.add_argument('caffe_model', help='The binary model parameter file in Caffe format')
    parser.add_argument('save_model_name', help='The name of the output model prefix')
    args = parser.parse_args()

    prob = proto2symbol(args.caffe_prototxt)
    caffe.set_mode_cpu()
    net_caffe = caffe.Net(args.caffe_prototxt, args.caffe_model, caffe.TEST)
    arg_shapes, output_shapes, aux_shapes = prob.infer_shape(data=(1,3,224,224))
    arg_names = prob.list_arguments()
    arg_shape_dic = dict(zip(arg_names, arg_shapes))
    arg_params = {}

    first_conv = True
    layer_names = net_caffe._layer_names
    for layer_idx, layer in enumerate(net_caffe.layers):
            layer_name = layer_names[layer_idx].replace('/', '_')
            if layer.type == 'Convolution' or layer.type == 'InnerProduct':
                assert(len(layer.blobs) == 2)
                wmat = layer.blobs[0].data
                bias = layer.blobs[1].data
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

                wmat = wmat.reshape(arg_shape_dic[weight_name])
                arg_params[weight_name] = mx.nd.zeros(wmat.shape)
                arg_params[weight_name][:] = wmat

                bias = bias.reshape(arg_shape_dic[bias_name])
                arg_params[bias_name] = mx.nd.zeros(bias.shape)
                arg_params[bias_name][:] = bias

                if first_conv and layer.type == 'Convolution':
                    first_conv = False

    model = mx.model.FeedForward(ctx=mx.cpu(), symbol=prob,
            arg_params=arg_params, aux_params={}, num_round=1,
            learning_rate=0.05, momentum=0.9, wd=0.0001)

    model.save(args.save_model_name)

if __name__ == '__main__':
    main()