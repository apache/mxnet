#!/usr/bin/env python
import mxnet as mx
import logging
import imagenet

logging.basicConfig(level=logging.DEBUG)

kv = mx.kvstore.create('dist_sync')

batch_size = 96
(train, val) = imagenet.ilsvrc12(num_parts = kv.num_workers,
                                part_index = kv.rank,
                                batch_size = batch_size,
                                input_shape = (3, 224, 224))

# assume each worker has two gpus
devs = [mx.gpu(i) for i in range(2)]

model = mx.model.FeedForward(
    ctx           = devs,
    symbol        = imagenet.inception(1000),
    num_round     = 20,
    learning_rate = 0.05,
    momentum      = 0.9,
    wd            = 0.00001)

model.fit(X        = train,
          eval_data     = val,
          kvstore       = kv,
          epoch_end_callback = mx.callback.Speedometer(batch_size, 5))
