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

kv = mx.kv.create('horovod')
my_rank = kv.rank
my_num_workers = kv.num_workers


def test_pushpull():
    ctx = mx.gpu(kv.local_rank) if mx.context.num_gpus() > 0 else mx.cpu(kv.local_rank)
    scale = kv.rank + 1
    tensor = mx.nd.ones(shape, ctx) * scale
    kv.pushpull('3', tensor)

    expected = (kv.num_workers + 1) * kv.num_workers / 2
    check_diff_to_scalar(tensor, expected)
    print('worker ' + str(kv.local_rank) + ' passed test_pushpull')


def test_broadcast():
    ctx = mx.gpu(kv.local_rank) if mx.context.num_gpus() > 0 else mx.cpu(kv.local_rank)
    val = mx.nd.zeros(shape, ctx)
    kv.broadcast('0', mx.nd.ones(shape), out=val)
    expected = 1
    check_diff_to_scalar(val, expected, kv.rank)
    print('worker ' + str(kv.local_rank) + ' passed test_broadcast')


def test_type():
    assert kv.type == 'horovod'


if __name__ == "__main__":
    test_type()
    test_broadcast()
    test_pushpull()
