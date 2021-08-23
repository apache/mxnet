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
import mxnet as mx
import json

key = '99'
shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND
kv = mx.kv.create('dist_sync')

def init_kv():
    # init kv dns keys
    kv.init(key, mx.nd.ones(shape))
    kv.set_optimizer(mx.optimizer.create('sgd'))
    return kv, kv.rank, kv.num_workers

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()
    def check_default_keys(kv, my_rank):
        nrepeat = 10
        # checks pull after push in loop, because behavior during
        # consecutive pushes doesn't offer any guarantees
        for i in range(nrepeat):
            kv.push(key, mx.nd.ones(shape, dtype='float32') * (my_rank+1))
            val = mx.nd.zeros(shape, dtype='float32')
            kv.pull(key, out=val)
            mx.nd.waitall()
    check_default_keys(kv, my_rank)

if __name__ == "__main__":
    server_filename_suffix = 'test_profile_server.json'
    worker_filename_suffix = 'test_profile_worker.json'
    mx.profiler.set_config(filename=server_filename_suffix, profile_all=True, profile_process='server')
    mx.profiler.set_config(filename='rank' + str(kv.rank) + '_' + worker_filename_suffix, profile_all=True, profile_process='worker')
    mx.profiler.set_state(state='run', profile_process='server')
    mx.profiler.set_state(state='run', profile_process='worker')
    test_sync_push_pull()
    mx.profiler.set_state(state='stop', profile_process='server')
    mx.profiler.set_state(state='stop', profile_process='worker')

    import glob, os

    # will only work when launcher mode is local, as used for integration test
    if kv.rank == 0:
        for rank in range(kv.num_workers):
            for suffix in [worker_filename_suffix, server_filename_suffix]:
                # throws value error if file is not proper json
                filename = 'rank' + str(rank) + '_' + suffix
                print(glob.glob('*'), os.getcwd())
                with open(filename, 'r') as f:
                    j = json.load(f)



