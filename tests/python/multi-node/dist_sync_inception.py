#!/usr/bin/env python
# pylint: skip-file
import common
import mxnet as mx
import logging

mx.random.seed(0)
logging.basicConfig(level=logging.DEBUG)

kv = mx.kvstore.create('dist_sync')

(train, val) = common.cifar10(
    batch_size = 128,
    input_shape=(3,28,28))

model = mx.model.FeedForward.create(
    ctx           = mx.gpu(kv.rank),
    kvstore       = kv,
    symbol        = common.inception(),
    X             = train,
    eval_data     = val,
    num_epoch     = 4,
    epoch_size    = 60000 / 128,
    learning_rate = 0.1,
    momentum      = 0.9,
    wd            = 0.00001,
    initializer   = mx.init.Uniform(0.07))
