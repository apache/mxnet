#!/usr/bin/env python
# pylint: skip-file
import common
import mxnet as mx
import logging

def test_lenet(devs, kv_type):
    # guarantee the same weight init for each run
    mx.random.seed(0)
    logging.basicConfig(level=logging.DEBUG)

    # (train, val) = common.cifar10(batch_size = 128, input_shape=(3,28,28))
    (train, val) = common.mnist(batch_size = 100, input_shape=(1,28,28))

    model = mx.model.FeedForward.create(
        ctx           = devs,
        kvstore       = kv_type,
        symbol        = common.lenet(),
        X             = train,
        num_epoch     = 3,
        learning_rate = 0.1,
        momentum      = 0.9,
        wd            = 0.00001)

    return common.accuracy(model, val)

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
