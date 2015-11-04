#!/usr/bin/env python
import mxnet as mx
import logging
import common

mx.random.seed(0)
logging.basicConfig(level=logging.DEBUG)

kv = mx.kvstore.create('dist_async')

# feed each machine the whole data
(train, val) = common.mnist(num_parts = kv.num_workers,
                            part_index = kv.rank,
                            batch_size = 100,
                            input_shape = (1,28,28))

model  = mx.model.FeedForward.create(
    ctx           = mx.gpu(kv.rank),
    kvstore       = kv,
    symbol        = common.lenet(),
    X             = train,
    num_epoch     = 10,
    learning_rate = 0.05,
    momentum      = 0.9,
    wd            = 0.00001)

common.accuracy(model, val)
