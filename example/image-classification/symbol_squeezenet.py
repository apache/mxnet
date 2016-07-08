"""
paper: http://arxiv.org/abs/1602.07360
github: https://github.com/DeepScale/SqueezeNet

@article{SqueezeNet,
    Author = {Forrest N. Iandola and Matthew W. Moskewicz and Khalid Ashraf and Song Han and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <1MB model size},
    Journal = {arXiv:1602.07360},
    Year = {2016}
}

squeezenet
"""

import find_mxnet
import mxnet as mx

# Basic Conv + BN + ReLU factory
def FireModelFactory(data, size):
    if size == 1:
        n_s1x1 = 16
        n_e1x1 = 64
        n_e3x3 = 64
    elif size == 2:
        n_s1x1 = 32
        n_e1x1 = 128
        n_e3x3 = 128
    elif size == 3:
        n_s1x1 = 48
        n_e1x1 = 192
        n_e3x3 = 192
    elif size == 4:
        n_s1x1 = 64
        n_e1x1 = 256
        n_e3x3 = 256

    squeeze1x1 = mx.symbol.Convolution(
            data   = data, 
            kernel = (1,1), 
            pad    = (0,0),
            num_filter = n_s1x1 )

    relu_squeeze1x1 = mx.symbol.Activation( data=squeeze1x1, act_type="relu" )

    expand1x1 = mx.symbol.Convolution(
            data   = relu_squeeze1x1,
            kernel = (1,1),
            pad    = (0,0),
            num_filter = n_e1x1 )

    relu_expand1x1 = mx.symbol.Activation(data=expand1x1, act_type="relu" )

    expand3x3 = mx.symbol.Convolution(
            data   = relu_squeeze1x1,
            kernel = (3,3),
            pad    = (1,1),
            num_filter = n_e3x3 )

    relu_expand3x3 = mx.symbol.Activation(data=expand3x3, act_type="relu" )

    concat = mx.symbol.Concat( *[relu_expand1x1, relu_expand3x3] )

    return concat 

def get_symbol(num_classes = 1000):
    data = mx.symbol.Variable(name="data")

    conv1 = mx.symbol.Convolution(data=data, kernel=(7,7), stride=(2,2), num_filter=96)
    relu_conv1 = mx.symbol.Activation(data=conv1, act_type="relu")
    maxpool1 = mx.symbol.Pooling(data=relu_conv1, kernel=(3,3), stride=(2,2), pool_type="max")

    fire2 = FireModelFactory(data=maxpool1, size=1)
    fire3 = FireModelFactory(data=fire2, size=1)
    fire4 = FireModelFactory(data=fire3, size=2)

    maxpool4 = mx.symbol.Pooling(data=fire4, kernel=(3,3), stride=(2,2), pool_type="max")

    fire5 = FireModelFactory(data=maxpool4, size=2)
    fire6 = FireModelFactory(data=fire5, size=3)
    fire7 = FireModelFactory(data=fire6, size=3)
    fire8 = FireModelFactory(data=fire7, size=4)

    maxpool8 = mx.symbol.Pooling(data=fire8, kernel=(3,3), stride=(2,2), pool_type="max")

    fire9 = FireModelFactory(data=maxpool8, size=4)
    dropout_fire9 = mx.symbol.Dropout(data=fire9, p=0.5)

    conv10 = mx.symbol.Convolution(data=dropout_fire9, kernel=(1,1), pad=(1,1), num_filter=1000)
    relu_conv10 = mx.symbol.Activation(data=conv10, act_type="relu")
    avgpool10 = mx.symbol.Pooling(data=relu_conv10, global_pool="true", kernel=(13,13), stride=(1,1), pool_type="avg")

    flatten = mx.symbol.Flatten(data=avgpool10, name='flatten')

    softmax = mx.symbol.SoftmaxOutput(data=flatten, name="softmax")
    return softmax
