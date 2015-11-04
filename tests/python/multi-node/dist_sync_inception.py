#!/usr/bin/env python
# pylint: skip-file
import common
import mxnet as mx
import logging

mx.random.seed(0)
logging.basicConfig(level=logging.DEBUG)

kv = mx.kvstore.create('dist_sync')

# feed each machine the whole data
(train, val) = common.cifar10(batch_size = 128, input_shape=(3,28,28))

model = mx.model.FeedForward.create(
    ctx           = mx.gpu(kv.rank),
    kvstore       = kv,
    symbol        = common.inception(),
    X             = train,
    num_epoch     = 4,
    learning_rate = 0.1,
    momentum      = 0.9,
    wd            = 0.00001,
    initializer   = mx.init.Uniform(0.07))

common.accuracy(model, val)
