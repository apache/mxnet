#!/usr/bin/env python
import mnist
import mxnet as mx
import logging

## in amazon s3:
data_dir = "s3://dmlc/mnist/"
## in local machine:
# data_dir = "data"
## in hdfs:
# data_dir = hdfs:///dmlc/mnist/

## can be local, dist_async or dist_sync
kv_type = 'dist_sync'
## batch size
batch_size = 100
## number of gpus used in a worker
num_gpus = 1
## learning rate
learning_rate = 0.1


kv = mx.kvstore.create(kv_type)

(train, val) = mnist.data(data_dir = data_dir,
                            num_parts = kv.num_workers,
                            part_index = kv.rank,
                            batch_size = batch_size)

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    ctx           = [mx.gpu(i) for i in range(num_gpus)],
    symbol        = mnist.lenet(),
    num_epoch     = 40,
    epoch_size    = 60000 / batch_size / kv.num_workers,
    learning_rate = learning_rate,
    momentum      = 0.9,
    wd            = 0.00001,
    initializer   = mx.init.Uniform(0.07))

model.fit(
    X             = train,
    eval_data     = val,
    kvstore       = kv)
