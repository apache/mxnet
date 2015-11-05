#!/usr/bin/env python
import imagenet
import mxnet as mx
import logging

logging.basicConfig(level=logging.DEBUG)

# for single gpu
# kv = mx.kvstore.create('local')
# batch_size = 48
# devs = mx.gpu(0)

# dist_async - async sgd
# dist_sync - BSP sgd
kv = mx.kvstore.create('dist_async')
# assume each worker has two gpus
devs = [mx.gpu(i) for i in range(2)]
batch_size = 96

(train, val) = imagenet.ilsvrc12(num_parts = kv.num_workers,
                                part_index = kv.rank,
                                batch_size = batch_size,
                                input_shape = (3, 224, 224))

model = mx.model.FeedForward(
    ctx           = devs,
    symbol        = imagenet.inception(1000),
    num_epoch     = 20,
    learning_rate = 0.05,
    momentum      = 0.9,
    wd            = 0.00001)

model.fit(X        = train,
          eval_data     = val,
          kvstore       = kv,
          batch_end_callback = mx.callback.Speedometer(batch_size, 10))
