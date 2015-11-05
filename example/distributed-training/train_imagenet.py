#!/usr/bin/env python
import imagenet
import mxnet as mx
import logging

# data directory
data_dir = "../../../ilsvrc12/"
# local, dist_async or dist_sync
kv_type = 'dist_sync'
# batch size
batch_size = 96
# number of gpus used in a worker
num_gpus = 2
# learning rate
learning_rate = 0.1

kv = mx.kvstore.create(kv_type)

(train, val) = imagenet.ilsvrc12(
    data_dir = data_dir,
    num_parts = kv.num_workers,
    part_index = kv.rank,
    batch_size = batch_size)

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    ctx           = [mx.gpu(i) for i in range(num_gpus)],
    symbol        = imagenet.inception(1000),
    num_epoch     = 20,
    epoch_size    = 1281167 / batch_size / kv.num_workers,
    learning_rate = learning_rate,
    momentum      = 0.9,
    wd            = 0.00001)

model.fit(X        = train,
          eval_data     = val,
          kvstore       = kv,
          batch_end_callback = mx.callback.Speedometer(batch_size, 10))
