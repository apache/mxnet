#!/usr/bin/env python
# pylint: skip-file
import common
import mxnet as mx
import logging

mx.random.seed(0)
logging.basicConfig(level=logging.DEBUG)
kv = mx.kvstore.create('dist_async')
(train, val) = common.cifar10(num_parts = kv.num_workers,
                              part_index = kv.rank,
                              batch_size = 128,
                              input_shape=(3,28,28))
devs = [mx.gpu(i) for i in range(2)]
model = mx.model.FeedForward.create(
    ctx           = devs,
    kvstore       = kv,
    symbol        = common.inception(),
    X             = train,
    eval_data     = val,
    num_epoch     = 20,
    learning_rate = 0.05,
    momentum      = 0.9,
    wd            = 0.00001,
    initializer   = mx.init.Uniform(0.07))

common.accuracy(model, val)
