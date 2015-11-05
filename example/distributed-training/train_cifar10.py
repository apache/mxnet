#!/usr/bin/env python
import cifar10
import mxnet as mx
import logging
import math

# dist_async or dist_sync
# kv_type = 'dist_async'
kv_type = 'local'
# data dir
data_dir = "data/cifar"
# batch size
batch_size = 256
# number of gpus used in a worker
num_gpus = 1
# learning rate
learning_rate = 0.1

if data_dir == "data/cifar":
    import sys
    sys.path.insert(0, "../../tests/python/common")
    import get_data
    get_data.GetCifar10()

kv = mx.kvstore.create(kv_type)

(train, val) = cifar10.data(data_dir = data_dir,
                            num_parts = kv.num_workers,
                            part_index = kv.rank,
                            batch_size = batch_size)

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    # ctx           = [mx.gpu(i) for i in range(num_gpus)],
    ctx = mx.gpu(1),
    symbol        = cifar10.inception(),
    num_epoch     = 20,
    epoch_size    = math.ceil(60000/batch_size/kv.num_workers),
    learning_rate = learning_rate,
    momentum      = 0.9,
    wd            = 0.00001,
    initializer   = mx.init.Uniform(0.07))

model.fit(
    X             = train,
    eval_data     = val,
    kvstore       = kv,
    batch_end_callback = mx.callback.Speedometer(batch_size, 10))
