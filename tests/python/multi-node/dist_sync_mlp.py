#!/usr/bin/env python

import mxnet as mx
import logging
import common

mx.random.seed(0)
logging.basicConfig(level=logging.DEBUG)

kv = mx.kvstore.create('dist_sync')

# feed each machine the whole data
(train, val) = common.mnist(batch_size = 100,
                            input_shape = (784,))

# train
model  = mx.model.FeedForward.create(
    symbol        = common.mlp(),
    ctx           = mx.cpu(),
    X             = train,
    num_round     = 4,
    learning_rate = 0.1,
    wd            = 0.0004,
    momentum      = 0.9,
    kvstore       = kv)

common.accuracy(model, val)
