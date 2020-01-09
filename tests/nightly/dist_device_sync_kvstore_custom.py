#!/usr/bin/env python

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

import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import numpy.random as rnd
import time
import argparse

# parser
parser = argparse.ArgumentParser(description='kvstore test')
parser.add_argument('--name', type=str, default='dist_device_sync')
args = parser.parse_args()

def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

# setup
keys = ['3', '5', '7']
init_test_keys = [str(i) for i in range(200,300)]
init_test_keys_big = [str(i) for i in range(300,400)]
init_test_keys_device = [str(i) for i in range(400,500)]
init_test_keys_device_big = [str(i) for i in range(500,600)]

shape = (2, 3)
big_shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND

kv = mx.kv.create(args.name)
my_rank = kv.rank
my_num_workers = kv.num_workers

def test_pushpull():
    num_gpus = 2
    def check_default_keys(nrepeat=3):
        # init kv dns keys
        kv.broadcast('3', mx.nd.ones(shape, ctx=mx.gpu()), mx.nd.ones(shape, ctx=mx.gpu()))
        kv.broadcast('99', mx.nd.ones(big_shape, ctx=mx.gpu()), mx.nd.ones(big_shape, ctx=mx.gpu()))
        for i in range(nrepeat):
            scale = my_rank + 1
            num = (my_num_workers + 1) * my_num_workers * num_gpus / 2

            arrs = [mx.nd.ones(shape, ctx=mx.gpu(j)) * scale for j in range(num_gpus)]
            # inplace
            kv.pushpull('3', arrs)
            for arr in arrs:
                check_diff_to_scalar(arr, num)

            big_arrs = [mx.nd.ones(big_shape, ctx=mx.gpu(j)) * scale for j in range(num_gpus)]
            # inplace
            kv.pushpull('99', big_arrs)
            for big_arr in big_arrs:
                check_diff_to_scalar(big_arr, num)

    check_default_keys(nrepeat=3)
    print('worker ' + str(my_rank) + ' is done')

def test_broadcast():
    def check_broadcast(kv, cur_keys, cur_shape, device=False):
        ctx = mx.gpu(0) if device else mx.cpu(0)
        val = [mx.nd.zeros(cur_shape, ctx) for i in cur_keys]
        for i in range(len(cur_keys)):
            expected = i
            kv.broadcast(cur_keys[i], [mx.nd.ones(cur_shape, ctx) * i], out=val[i])
            check_diff_to_scalar(val[i], expected, my_rank)
    check_broadcast(kv, init_test_keys, shape)
    check_broadcast(kv, init_test_keys_big, big_shape)
    check_broadcast(kv, init_test_keys_device, shape, device=True)
    check_broadcast(kv, init_test_keys_device_big, big_shape, device=True)
    print('worker ' + str(my_rank) + ' is initialized')

def test_type():
    assert kv.type == args.name

if __name__ == "__main__":
    test_type()
    test_broadcast()
    test_push_pull()
