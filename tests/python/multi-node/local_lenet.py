#!/usr/bin/env python
# pylint: skip-file
import mxnet as mx
from common import mnist, accuracy, cifar10
import logging

## define lenet
# input
data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                          kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                          kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet = mx.symbol.Softmax(data=fc2)

def test_lenet(devs, kv_type):
    # guarantee the same weight init for each run
    mx.random.seed(0)
    logging.basicConfig(level=logging.DEBUG)

    # (train, val) = cifar10(batch_size = 128, input_shape=(3,28,28))
    (train, val) = mnist(batch_size = 100, input_shape=(1,28,28))

    model = mx.model.FeedForward.create(
        ctx           = devs,
        kvstore       = kv_type,
        symbol        = lenet,
        X             = train,
        num_round     = 3,
        learning_rate = 0.1,
        momentum      = 0.9,
        wd            = 0.00001)

    return accuracy(model, val)

if __name__ == "__main__":
    gpus = [mx.gpu(i) for i in range(2)]

    base = test_lenet(mx.gpu(), 'none')
    acc1 = test_lenet(mx.gpu(), 'none')
    acc2 = test_lenet(gpus, 'local_update_cpu')
    acc3 = test_lenet(gpus, 'local_allreduce_cpu')
    acc4 = test_lenet(gpus, 'local_allreduce_device')

    assert base > 0.95
    # assert base > 0.5
    assert abs(base - acc1) < 1e-3
    assert abs(base - acc2) < 1e-3
    assert abs(base - acc3) < 1e-3
    assert abs(base - acc4) < 1e-3
