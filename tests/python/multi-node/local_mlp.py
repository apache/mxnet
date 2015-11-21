#!/usr/bin/env python
import common
import mxnet as mx
import logging

def test_mlp(devs, kv_type):
    # guarantee the same weight init for each run
    mx.random.seed(0)
    logging.basicConfig(level=logging.DEBUG)

    (train, val) = common.mnist(batch_size = 100,
                                input_shape = (784,))

    # train
    model  = mx.model.FeedForward.create(
        symbol        = common.mlp(),
        ctx           = devs,
        X             = train,
        num_epoch     = 4,
        learning_rate = 0.1,
        wd            = 0.0004,
        momentum      = 0.9,
        kvstore       = kv_type)

    return common.accuracy(model, val)

if __name__ == "__main__":
    base = test_mlp(mx.cpu(), 'none')
    assert base > 0.95

    cpus = [mx.cpu(i) for i in range(2)]
    acc1 =  test_mlp(cpus, 'local_update_cpu')
    acc2 =  test_mlp(cpus, 'local_allreduce_cpu')
    acc3 =  test_mlp(cpus, 'local_allreduce_device')

    assert abs(base - acc1) < 1e-3
    assert abs(base - acc2) < 1e-3
    assert abs(base - acc3) < 1e-3
