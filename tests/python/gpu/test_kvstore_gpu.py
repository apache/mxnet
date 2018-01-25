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
import mxnet as mx
import numpy as np
import unittest
from mxnet.test_utils import assert_almost_equal, default_context

shape = (4, 4)
keys = [5, 7, 11]
str_keys = ['b', 'c', 'd']


def init_kv_with_str(stype='default', kv_type='local'):
    """init kv """
    kv = mx.kv.create(kv_type)
    # single
    kv.init('a', mx.nd.zeros(shape, stype=stype))
    # list
    kv.init(str_keys, [mx.nd.zeros(shape=shape, stype=stype)] * len(keys))
    return kv


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
                row_ids = [mx.nd.array(row_id, dtype='int64')] * count
            elif use_slice:
                total_row_ids = mx.nd.array(np.random.randint(num_rows, size=count*num_rows), dtype='int64')
                row_ids = [total_row_ids[i*num_rows : (i+1)*num_rows] for i in range(count)]
            else:
                for i in range(count):
                    row_id = np.random.randint(num_rows, size=num_rows)
                    row_ids.append(mx.nd.array(row_id, dtype='int64'))
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

    # test fails intermittently. temporarily disabled till it gets fixed. tracked at https://github.com/apache/incubator-mxnet/issues/9384
    # check_rsp_push_pull('local')
    check_rsp_push_pull('device')
    check_rsp_push_pull('device', is_push_cpu=False)


if __name__ == '__main__':
    test_rsp_push_pull()
