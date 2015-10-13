#!/usr/bin/env python
import mxnet as mx
import logging
import common

mx.random.seed(0)
logging.basicConfig(level=logging.DEBUG)

kv = mx.kvstore.create('dist_sync')

# feed each machine the whole data
(train, val) = common.mnist(batch_size = 100,
                            input_shape = (1,28,28))

# train, worker i uses gpu i
model  = mx.model.FeedForward.create(
    ctx           = mx.gpu(kv.rank),
    kvstore       = kv,
    symbol        = common.lenet(),
    X             = train,
    num_round     = 3,
    learning_rate = 0.1,
    momentum      = 0.9,
    wd            = 0.00001)

common.accuracy(model, val)
