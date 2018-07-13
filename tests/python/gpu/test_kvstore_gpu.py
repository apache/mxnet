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
import os
import mxnet as mx
import numpy as np
import unittest
import logging
from mxnet.test_utils import assert_almost_equal, default_context
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

shape = (4, 4)
keys = [5, 7, 11]
str_keys = ['b', 'c', 'd']
logging.basicConfig(level=logging.INFO)

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

def init_kv_with_str(stype='default', kv_type='local'):
    """init kv """
    kv = mx.kv.create(kv_type)
    # single
    kv.init('a', mx.nd.zeros(shape, stype=stype))
    # list
    kv.init(str_keys, [mx.nd.zeros(shape=shape, stype=stype)] * len(keys))
    return kv

def test_dense_push_pull():
    shapes = [(1026), (1,2,3,4,5,6,7,8)]
    keys = [1,2,3,4,5,6,7]

    def check_dense_push_pull(kv_type):
        def check_dense_pull(kv_type, ctxs):
            n = 0
            n_devs = len(ctxs)
            for context in ctxs:
                kv = mx.kv.create(kv_type)
                a = mx.nd.ones(shape, ctxs[0])
                cur_key = str(key*n_devs+n)
                kv.init(cur_key, a)
                arr_list = [mx.nd.ones(shape, ctx=context) for context in ctxs]
                res = [mx.nd.zeros(shape, ctx=context) for context in ctxs]
                kv.push(cur_key, arr_list)
                kv.pull(cur_key, res)
                n += 1
                for x in range(n_devs):
                    #if np.sum(np.abs((res[x]-n_devs).asnumpy()))!=0:
                    print(x, (res[x]-n_devs).asnumpy())
                    assert(np.sum(np.abs((res[x]-n_devs).asnumpy()))==0)

        for key in keys:
            check_dense_pull(kv_type, [mx.gpu(0)])
            check_dense_pull(kv_type, [mx.cpu(0)])
            check_dense_pull(kv_type, [mx.gpu(i) for i in range(4)])
            check_dense_pull(kv_type, [mx.cpu(i) for i in range(4)])

    key1 = 'MXNET_KVSTORE_GPUARRAY_BOUND'
    envs2 = ['','1']
    key2  = 'MXNET_KVSTORE_USETREE'
    for i in range(2):
        for val2 in envs2:
            with EnvManager(key2, val2):
                check_dense_push_pull('local')
                check_dense_push_pull('device')

        os.environ[key1] = '0'
    os.environ[key1] = ''

# Test seed 89411477 (module seed 1829754103) resulted in a py3-gpu CI runner core dump.
# Not reproducible, so this test is back on random seeds.
@with_seed()
def test_rsp_push_pull():
    def check_rsp_push_pull(kv_type, is_push_cpu=True):
        kv = init_kv_with_str('row_sparse', kv_type)
        kv.init('e', mx.nd.ones(shape).tostype('row_sparse'))
        push_ctxs = [mx.cpu(i) if is_push_cpu else mx.gpu(i) for i in range(2)]
        kv.push('e', [mx.nd.ones(shape, ctx=context).tostype('row_sparse') for context in push_ctxs])

        def check_rsp_pull(kv, count, ctxs, is_same_rowid=False, use_slice=False):
            num_rows = shape[0]
            row_ids = []
            all_row_ids = np.arange(num_rows)
            vals = [mx.nd.sparse.zeros(shape=shape, ctx=ctxs[i], stype='row_sparse') for i in range(count)]
            if is_same_rowid:
                row_id = np.random.randint(num_rows, size=num_rows)
                row_ids = [mx.nd.array(row_id)] * count
            elif use_slice:
                total_row_ids = mx.nd.array(np.random.randint(num_rows, size=count*num_rows))
                row_ids = [total_row_ids[i*num_rows : (i+1)*num_rows] for i in range(count)]
            else:
                for i in range(count):
                    row_id = np.random.randint(num_rows, size=num_rows)
                    row_ids.append(mx.nd.array(row_id))
            row_ids_to_pull = row_ids[0] if (len(row_ids) == 1 or is_same_rowid) else row_ids
            vals_to_pull = vals[0] if len(vals) == 1 else vals

            kv.row_sparse_pull('e', out=vals_to_pull, row_ids=row_ids_to_pull)
            for val, row_id in zip(vals, row_ids):
                retained = val.asnumpy()
                excluded_row_ids = np.setdiff1d(all_row_ids, row_id.asnumpy())
                for row in range(num_rows):
                    expected_val = np.zeros_like(retained[row])
                    expected_val += 0 if row in excluded_row_ids else 2
                    assert_almost_equal(retained[row], expected_val)

        check_rsp_pull(kv, 1, [mx.gpu(0)])
        check_rsp_pull(kv, 1, [mx.cpu(0)])
        check_rsp_pull(kv, 4, [mx.gpu(i//2) for i in range(4)])
        check_rsp_pull(kv, 4, [mx.gpu(i//2) for i in range(4)], is_same_rowid=True)
        check_rsp_pull(kv, 4, [mx.cpu(i) for i in range(4)])
        check_rsp_pull(kv, 4, [mx.cpu(i) for i in range(4)], is_same_rowid=True)
        check_rsp_pull(kv, 4, [mx.gpu(i//2) for i in range(4)], use_slice=True)
        check_rsp_pull(kv, 4, [mx.cpu(i) for i in range(4)], use_slice=True)

    envs = ["","1"]
    key  = "MXNET_KVSTORE_USETREE"
    for val in envs:
        with EnvManager(key, val):
            print('done')
            check_rsp_push_pull('local')
            check_rsp_push_pull('device')
            check_rsp_push_pull('device', is_push_cpu=False)

@with_seed()
def test_row_sparse_pull_single_device():
    envs = ["","1"]
    key  = "MXNET_KVSTORE_USETREE"
    for val in envs:
        with EnvManager(key, val):
            kvstore = mx.kv.create('device')
            copy = mx.nd.random_normal(shape=(4,4), ctx=mx.gpu(0))
            grad = copy.tostype("row_sparse")

            k = 0
            kvstore.init(k, grad)
            idx = grad.indices
            kvstore.push(k, grad)
            kvstore.row_sparse_pull(k, out=grad, row_ids=idx)

            assert_almost_equal(grad.asnumpy(), copy.asnumpy())

@with_seed()
def test_rsp_push_pull_large_rowid():
    envs = ["","1"]
    key  = "MXNET_KVSTORE_USETREE"
    for val in envs:
        with EnvManager(key, val):
            num_rows = 793470
            val = mx.nd.ones((num_rows, 1)).tostype('row_sparse').copyto(mx.gpu())
            kv = mx.kv.create('device')
            kv.init('a', val)
            out = mx.nd.zeros((num_rows,1), stype='row_sparse').copyto(mx.gpu())
            kv.push('a', val)
            kv.row_sparse_pull('a', out=out, row_ids=mx.nd.arange(0, num_rows, dtype='int64'))
            assert(out.indices.shape[0] == num_rows)

if __name__ == '__main__':
    import nose
    nose.runmodule()
