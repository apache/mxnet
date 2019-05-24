# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
import sys
sys.path.insert(0, '../../python')
import mxnet as mx
import numpy as np
import os, pickle, gzip
import logging
from mxnet.test_utils import get_cifar10

batch_size = 128

# small mlp network
def get_net():
    data = mx.symbol.Variable('data')
    float_data = mx.symbol.Cast(data=data, dtype="float32")
    fc1 = mx.symbol.FullyConnected(float_data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(fc3, name="softmax")
    return softmax

# check data
get_cifar10()

def get_iterator_uint8(kv):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordUInt8Iter(
        path_imgrec = "data/cifar/train.rec",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    train = mx.io.PrefetchingIter(train)

    val = mx.io.ImageRecordUInt8Iter(
        path_imgrec = "data/cifar/test.rec",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

def get_iterator_uint8_with_param(kv, ctx):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank,
        dtype       ='uint8',
        ctx         = ctx)
    train = mx.io.PrefetchingIter(train)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank,
        dtype       ='uint8',
        ctx         = ctx)

    return (train, val)

def get_iterator_int8(kv):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordInt8Iter(
        path_imgrec = "data/cifar/train.rec",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    train = mx.io.PrefetchingIter(train)

    val = mx.io.ImageRecordInt8Iter(
        path_imgrec = "data/cifar/test.rec",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

def get_iterator_int8_with_param(kv, ctx):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank,
        dtype       ='int8',
        ctx         = ctx)
    train = mx.io.PrefetchingIter(train)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank,
        dtype       = 'int8',
        ctx         = ctx)

    return (train, val)

def get_iterator_float32(kv):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        mean_img    = "data/cifar/mean.bin",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    train = mx.io.PrefetchingIter(train)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        mean_img    = "data/cifar/mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

num_epoch = 1

def run_cifar10(train, val, use_module):
    train.reset()
    val.reset()
    devs = [mx.cpu(0)]
    net = get_net()
    mod = mx.mod.Module(net, context=devs)
    optim_args = {'learning_rate': 0.001, 'wd': 0.00001, 'momentum': 0.9}
    eval_metrics = ['accuracy']
    if use_module:
        executor = mx.mod.Module(net, context=devs)
        executor.fit(
            train,
            eval_data=val,
            optimizer_params=optim_args,
            eval_metric=eval_metrics,
            num_epoch=num_epoch,
            arg_params=None,
            aux_params=None,
            begin_epoch=0,
            batch_end_callback=mx.callback.Speedometer(batch_size, 50),
            epoch_end_callback=None)
    else:
        executor = mx.model.FeedForward.create(
            net,
            train,
            ctx=devs,
            eval_data=val,
            eval_metric=eval_metrics,
            num_epoch=num_epoch,
            arg_params=None,
            aux_params=None,
            begin_epoch=0,
            batch_end_callback=mx.callback.Speedometer(batch_size, 50),
            epoch_end_callback=None,
            **optim_args)

    ret = executor.score(val, eval_metrics)
    if use_module:
        ret = list(ret)
        logging.info('final accuracy = %f', ret[0][1])
        assert (ret[0][1] > 0.08)
    else:
        logging.info('final accuracy = %f', ret[0])
        assert (ret[0] > 0.08)

class CustomDataIter(mx.io.DataIter):
    def __init__(self, data):
        super(CustomDataIter, self).__init__()
        self.data = data
        self.batch_size = data.provide_data[0][1][0]

        # use legacy tuple
        self.provide_data = [(n, s) for n, s in data.provide_data]
        self.provide_label = [(n, s) for n, s in data.provide_label]

    def reset(self):
        self.data.reset()

    def next(self):
        return self.data.next()

    def iter_next(self):
        return self.data.iter_next()

    def getdata(self):
        return self.data.getdata()

    def getlabel(self):
        return self.data.getlable()

    def getindex(self):
        return self.data.getindex()

    def getpad(self):
        return self.data.getpad()

def test_cifar10():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    kv = mx.kvstore.create("local")
    # test float32 input
    (train, val) = get_iterator_float32(kv)
    run_cifar10(train, val, use_module=False)
    run_cifar10(train, val, use_module=True)

    # test legecay tuple in provide_data and provide_label
    run_cifar10(CustomDataIter(train), CustomDataIter(val), use_module=False)
    run_cifar10(CustomDataIter(train), CustomDataIter(val), use_module=True)

    # test uint8 input
    (train, val) = get_iterator_uint8(kv)
    run_cifar10(train, val, use_module=False)
    run_cifar10(train, val, use_module=True)

    for ctx in ("gpu", "cpu"):
        (train, val) = get_iterator_uint8_with_param(kv, ctx)
        run_cifar10(train, val, use_module=False)
        run_cifar10(train, val, use_module=True)

    # test int8 input
    (train, val) = get_iterator_int8(kv)
    run_cifar10(train, val, use_module=False)
    run_cifar10(train, val, use_module=True)

    for ctx in ("gpu", "cpu"):
        (train, val) = get_iterator_int8_with_param(kv, ctx)
        run_cifar10(train, val, use_module=False)
        run_cifar10(train, val, use_module=True)

if __name__ == "__main__":
    test_cifar10()
