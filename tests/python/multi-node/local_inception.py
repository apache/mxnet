#!/usr/bin/env python
# pylint: skip-file
import mxnet as mx
from common import cifar10, accuracy
import logging

# symbol

# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type)
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1))
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max')
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat

data = mx.symbol.Variable(name="data")
conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu")
in3a = SimpleFactory(conv1, 32, 32)
in3b = SimpleFactory(in3a, 32, 48)
in3c = DownsampleFactory(in3b, 80)
in4a = SimpleFactory(in3c, 112, 48)
in4b = SimpleFactory(in4a, 96, 64)
in4c = SimpleFactory(in4b, 80, 80)
in4d = SimpleFactory(in4c, 48, 96)
in4e = DownsampleFactory(in4d, 96)
in5a = SimpleFactory(in4e, 176, 160)
in5b = SimpleFactory(in5a, 176, 160)
pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool")
flatten = mx.symbol.Flatten(data=pool, name="flatten1")
fc = mx.symbol.FullyConnected(data=flatten, num_hidden=10, name="fc1")
softmax = mx.symbol.Softmax(data=fc, name="loss")

def test_inception(devs, kv_type):
    # guarantee the same weight init for each run
    mx.random.seed(0)
    logging.basicConfig(level=logging.DEBUG)

    (train, val) = cifar10(batch_size = 128, input_shape=(3,28,28))

    model = mx.model.FeedForward.create(
        ctx           = devs,
        symbol        = softmax,
        X             = train,
        kvstore       = kv_type,
        eval_data = val,
        num_round     = 1,
        learning_rate = 0.1,
        momentum      = 0.9,
        wd            = 0.00001,
        initializer   = mx.init.Uniform(0.07))

    return accuracy(model, val)

if __name__ == "__main__":
    # base = test_inception(mx.gpu(), 'none')

    gpus = [mx.gpu(i) for i in range(2)]
    acc1 =  test_inception(gpus, 'local_update_cpu')
    # acc2 =  test_inception(gpus, 'local_allreduce_cpu')
    # acc3 =  test_inception(gpus, 'local_allreduce_device')

    # assert base > 0.95
    # assert abs(base - acc1) < 1e-3
    # assert abs(base - acc2) < 1e-3
    # assert abs(base - acc3) < 1e-3
