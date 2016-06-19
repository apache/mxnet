"""
simplified inception-bn.py for images has size around 28 x 28
"""

import find_mxnet
import mxnet as mx

# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu", mirror_attr={}):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type, attr=mirror_attr)
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3, mirror_attr):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1), mirror_attr=mirror_attr)
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max', attr=mirror_attr)
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3, mirror_attr):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1, mirror_attr=mirror_attr)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3, mirror_attr=mirror_attr)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat

def get_symbol(num_classes = 10, force_mirroring=False):
    if force_mirroring:
        attr = {'force_mirroring': 'true'}
    else:
        attr = {}

    data = mx.symbol.Variable(name="data")
    conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu", mirror_attr=attr)
    in3a = SimpleFactory(conv1, 32, 32, mirror_attr=attr)
    in3b = SimpleFactory(in3a, 32, 48, mirror_attr=attr)
    in3c = DownsampleFactory(in3b, 80, mirror_attr=attr)
    in4a = SimpleFactory(in3c, 112, 48, mirror_attr=attr)
    in4b = SimpleFactory(in4a, 96, 64, mirror_attr=attr)
    in4c = SimpleFactory(in4b, 80, 80, mirror_attr=attr)
    in4d = SimpleFactory(in4c, 48, 96, mirror_attr=attr)
    in4e = DownsampleFactory(in4d, 96, mirror_attr=attr)
    in5a = SimpleFactory(in4e, 176, 160, mirror_attr=attr)
    in5b = SimpleFactory(in5a, 176, 160, mirror_attr=attr)
    pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool", attr=attr)
    flatten = mx.symbol.Flatten(data=pool, name="flatten1", attr=attr)
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name="fc1")
    softmax = mx.symbol.SoftmaxOutput(data=fc, name="softmax")
    return softmax
