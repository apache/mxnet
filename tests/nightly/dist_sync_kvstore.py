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

# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import time

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), A.asnumpy()

# setup
keys = [3, 5, 7]
rate = 2
shape = (2, 2)
big_shape = (1200, 1200)        # big than BIGARRAY_BOUND


kv = mx.kv.create('dist_sync')

# init kv
kv.init(keys, [mx.nd.ones(shape)] * len(keys))
kv.init(99, mx.nd.ones(big_shape))
# init updater on servers
kv.set_optimizer(mx.optimizer.create('test', rate))

my_rank = kv.rank
nworker = kv.num_workers

def test_sync_push_pull():
    nrepeat = 3
    for i in range(nrepeat):
        kv.push(3, mx.nd.ones(shape)*(my_rank+1))
        kv.push(99, mx.nd.ones(big_shape)*(my_rank+1))

    num = (nworker + 1 ) * nworker * rate / 2 * nrepeat + 1
    val = mx.nd.zeros(shape)
    kv.pull(3, out = val)
    check_diff_to_scalar(val, num)
    # print val.asnumpy()

    val2 = mx.nd.zeros(big_shape)
    kv.pull(99, out = val2)
    check_diff_to_scalar(val2, num)

if __name__ == "__main__":
    test_sync_push_pull()
