#!/usr/bin/env python
# pylint: skip-file
import mxnet as mx
import os, sys
import logging
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../common/'))
from common import mnist, accuracy

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.Softmax(fc3, name = 'sm')

def test_mlp(devs, kv_type):
    # guarantee the same weight init for each run
    mx.random.seed(0)
    logging.basicConfig(level=logging.DEBUG)

    (train, val) = mnist(batch_size = 100,
                         input_shape = (784,))
    # train
    model  = mx.model.FeedForward.create(
        symbol        = softmax,
        ctx           = devs,
        X             = train,
        num_round     = 2,
        learning_rate = 0.1,
        wd            = 0.0004,
        momentum      = 0.9,
        kvstore       = kv_type)

    return accuracy(model, val)

if __name__ == "__main__":
    base = test_mlp(mx.cpu(), 'none')
    print base
    assert base > 0.95

    cpus = [mx.cpu(i) for i in range(2)]
    acc =  test_mlp(cpus, 'local_update_cpu')
    print acc
    assert abs(base - acc) < 1e-4

    # acc =  test_mlp(cpus, 'local_allreduce_cpu')
    # assert abs(base - acc) < 1e-4
