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

import mxnet as mx
import numpy as np
import unittest
import os

shapes = [(10), (100), (1000), (10000), (100000), (2,2), (2,3,4,5,6,7,8)]
keys = [1,2,3,4,5,6,7]
num_gpus = len(mx.test_utils.list_gpus())


if num_gpus > 8 :
    print("The machine has {} gpus. We will run the test on 8 gpus.".format(num_gpus))
    print("There is a limit for all PCI-E hardware on creating number of P2P peers. The limit is 8.")
    num_gpus = 8;

gpus = range(1, 1+num_gpus)

class EnvManager:
    def __init__(self, key, val):
        self._key = key
        self._next_val = val
        self._prev_val = None

    def __enter__(self):
        try:
            self._prev_val = os.environ[self._key]
        except KeyError:
            self._prev_val = ''
        os.environ[self._key] = self._next_val

    def __exit__(self, ptype, value, trace):
        os.environ[self._key] = self._prev_val

def test_device_pushpull():
    def check_dense_pushpull(kv_type):
        for shape, key in zip(shapes, keys):
            for n_gpus in gpus:
                kv_device = mx.kv.create(kv_type)
                a = mx.nd.ones(shape, mx.gpu(0))
                cur_key = str(key*max(gpus)+n_gpus)
                kv_device.init(cur_key, a)
                arr_list = [mx.nd.ones(shape, mx.gpu(x)) for x in range(n_gpus)]
                res = [mx.nd.zeros(shape, mx.gpu(x)) for x in range(n_gpus)]
                kv_device.push(cur_key, arr_list)
                kv_device.pull(cur_key, res)
                for x in range(n_gpus):
                    assert(np.sum(np.abs((res[x]-n_gpus).asnumpy()))==0)

    envs1 = '1'
    key1 = 'MXNET_KVSTORE_TREE_ARRAY_BOUND'
    envs2 = ['','1']
    key2  = 'MXNET_KVSTORE_USETREE'
    for i in range(2):
        for val2 in envs2:
            with EnvManager(key2, val2):
                check_dense_pushpull('local')
                check_dense_pushpull('device')

        os.environ[key1] = envs1
    os.environ[key1] = ''

    print ("Passed")

if __name__ == '__main__':
    test_device_pushpull()
