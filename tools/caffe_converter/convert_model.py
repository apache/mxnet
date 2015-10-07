import mxnet as mx
import caffe
import argparse

data = mx.symbol.Variable(name='data')
conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
pool3 = mx.symbol.Pooling(name='pool3', data=relu3_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
pool4 = mx.symbol.Pooling(name='pool4', data=relu4_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')
conv5_2 = mx.symbol.Convolution(name='conv5_2', data=relu5_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu5_2 = mx.symbol.Activation(name='relu5_2', data=conv5_2 , act_type='relu')
conv5_3 = mx.symbol.Convolution(name='conv5_3', data=relu5_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
relu5_3 = mx.symbol.Activation(name='relu5_3', data=conv5_3 , act_type='relu')
pool5 = mx.symbol.Pooling(name='pool5', data=relu5_3 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max')
flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool5)
fc6 = mx.symbol.FullyConnected(name='fc6', data=flatten_0 , num_hidden=4096, no_bias=False)
relu6 = mx.symbol.Activation(name='relu6', data=fc6 , act_type='relu')
drop6 = mx.symbol.Dropout(name='drop6', data=relu6 , p=0.500000)
fc7 = mx.symbol.FullyConnected(name='fc7', data=drop6 , num_hidden=4096, no_bias=False)
relu7 = mx.symbol.Activation(name='relu7', data=fc7 , act_type='relu')
drop7 = mx.symbol.Dropout(name='drop7', data=relu7 , p=0.500000)
fc8 = mx.symbol.FullyConnected(name='fc8', data=drop7 , num_hidden=1000, no_bias=False)
prob = mx.symbol.Softmax(name='prob', data=fc8 )

parser = argparse.ArgumentParser(description='Caffe prototxt to mxnet model parameter converter.\
                Note that only basic functions are implemented. You are welcomed to contribute to this file.')
parser.add_argument('caffe_prototxt', help='The prototxt file in Caffe format')
parser.add_argument('caffe_model', help='The binary model parameter file in Caffe format')
parser.add_argument('save_model_name', help='The name of the output model prefix')
args = parser.parse_args()

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
                print 'Swapping BGR of caffe into RGB in cxxnet'
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
