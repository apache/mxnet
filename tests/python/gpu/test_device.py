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
import pytest
import os
import logging
from mxnet.test_utils import environment

shapes = [(10), (100), (1000), (10000), (100000), (2,2), (2,3,4,5,6,7,8)]
keys = [1,2,3,4,5,6,7]
num_gpus = mx.context.num_gpus()


if num_gpus > 8 :
    logging.warn("The machine has {} gpus. We will run the test on 8 gpus.".format(num_gpus))
    logging.warn("There is a limit for all PCI-E hardware on creating number of P2P peers. The limit is 8.")
    num_gpus = 8;

gpus = range(1, 1+num_gpus)

@pytest.mark.skipif(mx.context.num_gpus() < 1, reason="test_device_pushpull needs at least 1 GPU")
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

    kvstore_tree_array_bound_values = [None, '1']
    kvstore_usetree_values = [None, '1']
    for y in kvstore_tree_array_bound_values:
        for x in kvstore_usetree_values:
            with environment({'MXNET_KVSTORE_USETREE': x,
                              'MXNET_KVSTORE_TREE_ARRAY_BOUND': y}):
                check_dense_pushpull('local')
                check_dense_pushpull('device')


if __name__ == '__main__':
    test_device_pushpull()
