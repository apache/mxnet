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
import numpy.random as rnd
import time

def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

# setup
keys = ['3', '5', '7']
rsp_keys = ['9', '11', '13']

rate = 2
shape = (2, 3)
big_shape = (1200, 1200)        # bigger than BIGARRAY_BOUND


def init_kv():
    kv = mx.kv.create('dist_sync')
    # init kv dns keys
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    kv.init('99', mx.nd.ones(big_shape))
    # init kv row_sparse keys
    kv.init(rsp_keys, [mx.nd.ones(shape).tostype('row_sparse')] * len(rsp_keys))
    kv.init('100', mx.nd.ones(big_shape).tostype('row_sparse'))
    # worker info
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    return kv, my_rank, nworker

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()
    def check_default_keys(kv, my_rank, nworker):
        nrepeat = 3
        for i in range(nrepeat):
            kv.push('3', mx.nd.ones(shape)*(my_rank+1))
            kv.push('99', mx.nd.ones(big_shape)*(my_rank+1))

        num = (nworker + 1) * nworker * rate / 2 * nrepeat + 1
        val = mx.nd.zeros(shape)
        kv.pull('3', out=val)
        check_diff_to_scalar(val, num)

        val2 = mx.nd.zeros(big_shape)
        kv.pull('99', out=val2)
        check_diff_to_scalar(val2, num)

    def check_row_sparse_keys(kv, my_rank, nworker):
        nrepeat = 3
        # prepare gradient
        v = mx.nd.zeros(shape)
        my_row = my_rank % shape[0]
        v[my_row] = my_rank + 1
        # push
        for i in range(nrepeat):
            kv.push('9', v.tostype('row_sparse'))
        # select a random subset of rows this worker is interested in
        num_rows = shape[0]
        row_ids_np = np.random.randint(num_rows, size=num_rows)
        row_ids = mx.nd.array(row_ids_np, dtype='int64')
        # perform pull
        val = mx.nd.zeros(shape, stype='row_sparse')
        kv.row_sparse_pull('9', out=val, row_ids=row_ids)
        # prepare updated values
        updated_val = mx.nd.ones(shape)
        for rank in range(nworker):
            row = rank % shape[0]
            updated_val[row] += (rank + 1) * rate * nrepeat
        # verify subset of updated values
        expected = mx.nd.zeros(shape)
        for row in row_ids_np:
            expected[row] = updated_val[row]
        check_diff_to_scalar(val, expected)

    def check_row_sparse_keys_with_zeros(kv, my_rank, nworker):
        nrepeat = 3
        # prepare gradient
        v = mx.nd.zeros(shape)
        big_v = mx.nd.zeros(big_shape)
        # push
        for i in range(nrepeat):
            kv.push('11', v.tostype('row_sparse'))
            kv.push('100', big_v.tostype('row_sparse'))

        # pull a subset of rows this worker is interested in
        all_row_ids = np.arange(shape[0])
        val = mx.nd.ones(shape).tostype('row_sparse')
        big_val = mx.nd.ones(big_shape).tostype('row_sparse')
        kv.row_sparse_pull('11', out=val, row_ids=mx.nd.array(all_row_ids, dtype='int64'))
        big_num_rows = shape[0]
        big_all_row_ids = np.arange(big_shape[0])
        kv.row_sparse_pull('100', out=big_val, row_ids=mx.nd.array(big_all_row_ids, dtype='int64'))
        # verify results
        check_diff_to_scalar(val, mx.nd.ones(shape))
        check_diff_to_scalar(big_val, mx.nd.ones(big_shape))

    def check_big_row_sparse_keys(kv, my_rank, nworker):
        mx.random.seed(123)
        rnd.seed(123)
        density = 0.3
        nrepeat = 3
        # prepare gradient
        v = mx.nd.zeros(big_shape)
        idx_sample = rnd.rand(big_shape[0])
        indices = np.argwhere(idx_sample < density).flatten()
        # each worker chooses a subset of the indices to update
        update_rows = []
        for rank in range(nworker):
            rows = []
            i = 0
            step = (rank + 1) * 2
            while i < len(indices):
                rows.append(indices[i])
                i += step
            update_rows.append(np.array(rows))
        # rows to update for this worker
        for row in update_rows[my_rank]:
            v[row] = my_rank + 1
        # push
        for i in range(nrepeat):
            kv.push('100', v.tostype('row_sparse'))

        # select a random subset of rows this worker is interested in
        mx.random.seed(my_rank)
        rnd.seed(my_rank)
        num_rows = big_shape[0]
        row_ids_np = np.random.randint(num_rows, size=num_rows)
        row_ids = mx.nd.array(row_ids_np, dtype='int64')
        # perform pull
        val = mx.nd.zeros(big_shape, stype='row_sparse')
        kv.row_sparse_pull('100', out=val, row_ids=row_ids)
        # prepare expected result
        updated_val = mx.nd.ones(big_shape)
        # apply updates from each worker
        for rank in range(nworker):
            for row in update_rows[rank]:
                updated_val[row] += (rank + 1) * rate * nrepeat

        expected = mx.nd.zeros(big_shape)
        for row in row_ids_np:
            expected[row] = updated_val[row]
        check_diff_to_scalar(val, expected, rank=my_rank)

    check_default_keys(kv, my_rank, nworker)
    check_row_sparse_keys(kv, my_rank, nworker)
    check_row_sparse_keys_with_zeros(kv, my_rank, nworker)
    check_big_row_sparse_keys(kv, my_rank, nworker)
    print('worker ' + str(my_rank) + ' is done')


def test_compressed():
    def init_kv(type):
        kv = mx.kv.create(type)
        pos_threshold = 0.5
        neg_threshold = -0.5
        kv.set_compress({'compress': '2bit', 'pos_threshold': pos_threshold, 'neg_threshold': neg_threshold})
        # init kv dns keys
        kv.init('99', mx.nd.zeros(big_shape))
        kv.init('3', mx.nd.zeros(shape))
        my_rank = kv.rank
        nworker = kv.num_workers
        return kv, pos_threshold, neg_threshold, my_rank, nworker

    def verify_residual(kv, pos_threshold):
        for d in [('99', big_shape), ('3', shape)]:
            kv.push(d[0], mx.nd.ones(d[1])*0.4)
            val=mx.nd.zeros(d[1])
            kv.pull(d[0],val)
            check_diff_to_scalar(val, 0)
            kv.push(d[0], mx.nd.ones(d[1])*(pos_threshold - 0.4))
            val2 = mx.nd.zeros(d[1])
            kv.pull(d[0],val2)
            check_diff_to_scalar(val2, pos_threshold)
            kv.push(d[0], mx.nd.ones(d[1])*0.2)
            val3= mx.nd.zeros(d[1])
            kv.pull(d[0], val3)
            check_diff_to_scalar(val3, 0)
            kv.push(d[0], mx.nd.ones(d[1])*(pos_threshold-0.2))
            val4 = mx.nd.zeros(d[1])
            kv.pull(d[0],val4)
            check_diff_to_scalar(val4, pos_threshold)

    def check_ones(kv, pos):
        kv.push('99',mx.nd.ones(big_shape)*pos*2)
        val = mx.nd.zeros(big_shape)
        kv.pull('99', val)
        check_diff_to_scalar(val, pos)

    def check_zero(kv):
      kv.push('99', mx.nd.zeros(big_shape))
      val = mx.nd.zeros(big_shape)
      kv.pull('99', val)
      check_diff_to_scalar(val, 0)

    for type in ['dist_sync']:
        kv, pos, neg, rank, nworker = init_kv(type)
        check_zero(kv)
        verify_residual(kv, pos)
        check_ones(kv, pos)
    
if __name__ == "__main__":
    test_sync_push_pull()
    test_compressed()
