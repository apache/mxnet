#!/usr/bin/env python
import common
import mxnet as mx
import logging

mx.random.seed(0)
logging.basicConfig(level=logging.DEBUG)

kv = mx.kvstore.create('dist_async')

(train, val) = common.mnist(num_parts = kv.num_workers,
                            part_index = kv.rank,
                            batch_size = 100,
                            input_shape = (784,))

# train
model  = mx.model.FeedForward.create(
    symbol        = common.mlp(),
    ctx           = mx.cpu(),
    X             = train,
    num_epoch     = 4,
    learning_rate = 0.05,
    wd            = 0.0004,
    momentum      = 0.9,
    kvstore       = kv)

common.accuracy(model, val)
