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

def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

# setup
keys = ['3', '5', '7']
shape = (2, 3)
big_shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND

kv = mx.kv.create('dist_sync_allreduce')

def init_kv():
    my_rank = kv.rank
    nworker = kv.num_workers
    return kv, my_rank, nworker

def test_sync_pushpull():
    kv, my_rank, nworker = init_kv()
    def check_pushpull(kv, my_rank, nworker):
        nrepeat = 3
        for i in range(nrepeat):
            val = mx.nd.zeros(shape)
            val2 = mx.nd.zeros(big_shape)
            in_ = mx.nd.ones(shape)
            in2_ = mx.nd.ones(big_shape)
            kv.pushpull('3', in_, val)
            kv.pushpull('99', in2_, val2)
            num = nworker;
            check_diff_to_scalar(val, num)
            check_diff_to_scalar(val2, num)

    check_pushpull(kv, my_rank, nworker)
    print('worker ' + str(my_rank) + ' pushpull is done')

def test_sync_broadcast():
    kv, my_rank, nworker = init_kv()
    def check_broadcast(kv, my_rank, nworker):
        nrepeat = 3
        for i in range(nrepeat):
            if my_rank == 0:
                val = mx.nd.ones(shape)
                val2 = mx.nd.ones(big_shape)
            else:
                val = mx.nd.zeros(shape)
                val2 = mx.nd.zeros(big_shape)
            kv.broadcast('3', val, 0)
            kv.broadcast('99', val2, 0)
            num = 1
            check_diff_to_scalar(val, num)
            check_diff_to_scalar(val2, num)
    check_broadcast(kv, my_rank, nworker)
    print('worker ' + str(my_rank) + ' broadcast is done')
if __name__ == "__main__":
    test_sync_pushpull()
    test_sync_broadcast()
