#!/usr/bin/env python
# pylint: skip-file
import common
import mxnet as mx
import logging

def test_inception(devs, kv_type):
    # guarantee the same weight init for each run
    mx.random.seed(0)
    logging.basicConfig(level=logging.DEBUG)

    (train, val) = common.cifar10(batch_size = 128, input_shape=(3,28,28))

    model = mx.model.FeedForward.create(
        ctx           = devs,
        symbol        = common.inception(),
        X             = train,
        eval_data     = val,
        kvstore       = kv_type,
        num_epoch     = 10,
        learning_rate = 0.1,
        momentum      = 0.9,
        wd            = 0.00001,
        initializer   = mx.init.Uniform(0.07))

    return common.accuracy(model, val)

if __name__ == "__main__":
    base = test_inception(mx.gpu(), 'none')

    gpus = [mx.gpu(i) for i in range(2)]
    acc1 =  test_inception(gpus, 'local_update_cpu')
    acc2 =  test_inception(gpus, 'local_allreduce_cpu')
    acc3 =  test_inception(gpus, 'local_allreduce_device')

    assert base > 0.95
    assert abs(base - acc1) < 1e-3
    assert abs(base - acc2) < 1e-3
    assert abs(base - acc3) < 1e-3
