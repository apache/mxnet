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
init_test_keys = [str(i) for i in range(200,300)]
init_test_keys_big = [str(i) for i in range(300,400)]
init_test_keys_device = [str(i) for i in range(400,500)]
init_test_keys_device_big = [str(i) for i in range(500,600)]

rate = 2
shape = (2, 3)
big_shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND

kv = mx.kv.create('dist_device_sync')

def init_kv():
    # init kv dns keys
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    kv.init('9', mx.nd.ones(shape))
    kv.init('10', mx.nd.ones(shape))
    kv.init('99', mx.nd.ones(big_shape))
    kv.init('100', mx.nd.ones(big_shape))
    # worker info
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    return kv, my_rank, nworker

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()
    num_gpus = 2
    def check_default_keys(kv, my_rank, nworker, nrepeat=3):
        # checks pull after push in loop, because behavior during
        # consecutive pushes doesn't offer any guarantees
        for i in range(nrepeat):
            scale = my_rank + 1
            num = (nworker + 1) * nworker * rate * num_gpus / 2 * (i + 1) + 1

            arr = [mx.nd.ones(shape, ctx=mx.gpu(j)) * scale for j in range(num_gpus)]
            val = mx.nd.zeros(shape)
            kv.push('9', arr)
            kv.pull('9', out=val)
            check_diff_to_scalar(val, num)
            vals = [mx.nd.zeros(shape) for j in range(num_gpus)]
            kv.pushpull('10', arr, out=vals)
            for v in vals:
                check_diff_to_scalar(v, num)

            big_arr = [mx.nd.ones(big_shape, ctx=mx.gpu(j)) * scale for j in range(num_gpus)]
            big_val = mx.nd.zeros(big_shape)
            kv.push('99', big_arr)
            kv.pull('99', out=big_val)
            check_diff_to_scalar(big_val, num)
            big_vals = [mx.nd.zeros(big_shape) for j in range(num_gpus)]
            kv.pushpull('100', big_arr, out=big_vals)
            for bv in big_vals:
                check_diff_to_scalar(bv, num)

    check_default_keys(kv, my_rank, nworker, nrepeat=3)
    print('worker ' + str(my_rank) + ' is done')

def test_sync_init():
    def check_init(kv, cur_keys, cur_shape, device=False):
        ctx = mx.gpu(0) if device else mx.cpu()
        val = [mx.nd.zeros(cur_shape, ctx) for i in cur_keys]
        for i in range(len(cur_keys)):
            expected = i
            kv.init(cur_keys[i], [mx.nd.ones(cur_shape, ctx) * i])
            kv.pull(cur_keys[i], out=val[i])
            check_diff_to_scalar(val[i], expected)
    check_init(kv, init_test_keys, shape)
    check_init(kv, init_test_keys_big, big_shape)
    check_init(kv, init_test_keys_device, shape, device=True)
    check_init(kv, init_test_keys_device_big, big_shape, device=True)
    my_rank = kv.rank
    print('worker ' + str(my_rank) + ' is initialized')

def test_gluon_trainer_type():
    def check_trainer_kv_update(update_on_kv):
        x = mx.gluon.Parameter('x', shape=(10,1), lr_mult=1.0)
        x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
        try:
            trainer = mx.gluon.Trainer([x], 'sgd', {'learning_rate': 0.1},
                                       kvstore=kv, update_on_kvstore=update_on_kv)
            trainer._init_kvstore()
            assert trainer._kv_initialized
            if update_on_kv is not None:
                assert trainer._update_on_kvstore is update_on_kv
        except ValueError:
            assert update_on_kv is False

    check_trainer_kv_update(False)
    check_trainer_kv_update(True)
    check_trainer_kv_update(None)
    my_rank = kv.rank
    print('worker ' + str(my_rank) + ' passed test_gluon_trainer_type')

if __name__ == "__main__":
    test_sync_init()
    # TODO(szha): disabled due to repeated failures. tracked in #18098
    # test_sync_push_pull()
    test_gluon_trainer_type()
