#!/usr/bin/env python
import imagenet
import mxnet as mx
import logging

## in local machine:
# data_dir = "../../../ilsvrc12/"
## in amazon s3:
data_dir = "s3://dmlc/ilsvrc12/"
## in hdfs:
# data_dir = hdfs:///dmlc/ilsvrc12/

## non-distributed version (single machine)
# kv_type = 'local'
## distributed version, can be dist_async or dist_sync
kv_type = 'dist_async'

## batch size for one gpu
batch_size_per_gpu = 40
## number of gpus used in a worker
num_gpus = 1
## learning rate
learning_rate = 0.05

batch_size = batch_size_per_gpu * num_gpus
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
