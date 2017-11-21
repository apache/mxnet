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
from mxnet.test_utils import assert_almost_equal
from test_kvstore import compute_expected_2bit_quantization

def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

# setup
keys = ['3', '5', '7']
rsp_keys = ['9', '11', '13']
init_test_keys = [str(i) for i in range(200,300)]
init_test_keys_big = [str(i) for i in range(300,400)]
init_test_keys_device = [str(i) for i in range(400,500)]
init_test_keys_device_big = [str(i) for i in range(500,600)]

rate = 2
shape = (2, 3)
irregular_shape = (1211,1211)
big_shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND

kv = mx.kv.create('dist_sync')

def init_kv():
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

def init_kv_compressed(kv):
    threshold = 0.5
    kv.set_gradient_compression({'type': '2bit', 'threshold':threshold})
    # init kv compression keys
    kv.init('11221', mx.nd.zeros(big_shape))
    kv.init('112221', mx.nd.zeros(irregular_shape))
    kv.init('1121', mx.nd.zeros(shape))
    # to test inactive mode
    kv.init('1122', mx.nd.ones(shape))
    return kv, threshold

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()
    def check_default_keys(kv, my_rank, nworker):
        nrepeat = 3
        # checks pull after push in loop, because behavior during
        # consecutive pushes doesn't offer any guarantees
        for i in range(nrepeat):
            kv.push('3', mx.nd.ones(shape)*(my_rank+1))
            kv.push('99', mx.nd.ones(big_shape)*(my_rank+1))
            num = (nworker + 1) * nworker * rate / 2 * (i + 1) + 1
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
                updated_val[row] += (rank + 1) * rate * (i+1)
            # verify subset of updated values
            expected = mx.nd.zeros(shape)
            for row in row_ids_np:
                expected[row] = updated_val[row]
            check_diff_to_scalar(val, expected)

    def check_row_sparse_keys_with_zeros(kv, my_rank, nworker):
        nrepeat = 3
        # prepare gradient
        v = mx.nd.sparse.zeros('row_sparse', shape)
        big_v = mx.nd.sparse.zeros('row_sparse', big_shape)
        # push
        for i in range(nrepeat):
            kv.push('11', v)
            kv.push('100', big_v)
            # pull a subset of rows this worker is interested in
            all_row_ids = np.arange(shape[0])
            val = mx.nd.sparse.zeros('row_sparse', shape)
            big_val = mx.nd.sparse.zeros('row_sparse', big_shape)
            kv.row_sparse_pull('11', out=val, row_ids=mx.nd.array(all_row_ids))
            big_all_row_ids = np.arange(big_shape[0])
            kv.row_sparse_pull('100', out=big_val, row_ids=mx.nd.array(big_all_row_ids))
            # verify results
            check_diff_to_scalar(val, 1)
            check_diff_to_scalar(big_val, 1)
            # pull empty weights
            kv.row_sparse_pull('11', out=val, row_ids=mx.nd.array([]))
            kv.row_sparse_pull('100', out=big_val, row_ids=mx.nd.array([]))
            check_diff_to_scalar(val, 0)
            check_diff_to_scalar(big_val, 0)

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
            row_ids = mx.nd.array(row_ids_np)
            # perform pull
            val = mx.nd.zeros(big_shape, stype='row_sparse')
            kv.row_sparse_pull('100', out=val, row_ids=row_ids)
            # prepare expected result
            updated_val = mx.nd.ones(big_shape)
            # apply updates from each worker
            for rank in range(nworker):
                for row in update_rows[rank]:
                    updated_val[row] += (rank + 1) * rate * (i+1)

            expected = mx.nd.zeros(big_shape)
            for row in row_ids_np:
                expected[row] = updated_val[row]
            check_diff_to_scalar(val, expected, rank=my_rank)

    def check_compr_residual(kv, threshold, nworker):
        for k,s in [('1121', shape),('112221',irregular_shape),('11221', big_shape)]:
            # doesn't meet threshold
            kv.push(k, mx.nd.ones(s)*0.4)
            val=mx.nd.zeros(s)
            kv.pull(k,val)
            check_diff_to_scalar(val, 0)

            # just meets threshold with residual
            kv.push(k, mx.nd.ones(s)*(threshold - 0.4))
            val2 = mx.nd.zeros(s)
            kv.pull(k,val2)
            curval = threshold * rate * nworker
            check_diff_to_scalar(val2, curval)

            # doesn't meet threshold
            kv.push(k, mx.nd.ones(s)*0.2)
            val3= mx.nd.zeros(s)
            kv.pull(k, val3)
            check_diff_to_scalar(val3, curval)

            # exceeds again
            kv.push(k, mx.nd.ones(s)*(threshold-0.2))
            val4 = mx.nd.zeros(s)
            kv.pull(k,val4)
            curval += threshold*rate*nworker
            check_diff_to_scalar(val4, curval)
            # residual is 0 now

    def check_compr_ones(kv, threshold, nworker):
        for k,s in [('1121', shape),('112221',irregular_shape),('11221', big_shape)]:
            val = mx.nd.zeros(s)
            kv.pull(k, val)
            curval = val[0][0].asnumpy()[0]
            kv.push(k,mx.nd.ones(s)*threshold)
            val2 = mx.nd.zeros(s)
            kv.pull(k, val2)
            newval = curval + rate*nworker*threshold
            check_diff_to_scalar(val2, newval)
            # residual = 0  again

    def check_compr_pull_before_push(kv):
        for k,s in [('1121', shape),('112221',irregular_shape),
                    ('11221', big_shape), ('1122',shape)]:
            if k=='1122':
                # tests that GC is not used for init of a key
                val = mx.nd.zeros(s)
                kv.pull(k, val)
                check_diff_to_scalar(val, 1)
            else:
                val = mx.nd.ones(s)
                kv.pull(k, val)
                check_diff_to_scalar(val, 0)

    def check_compr_zero(kv):
        for k,s in [('1121', shape),('112221',irregular_shape),('11221', big_shape)]:
            kv.push(k, mx.nd.zeros(s))
            # to check that all are set to 0s
            val = mx.nd.ones(s)
            kv.pull(k, val)
            check_diff_to_scalar(val, 0)

    def check_compr_random(kv, threshold, nworker):
        # set a seed so all workers generate same data. knowing this helps
        # calculate expected value after pull
        mx.random.seed(123)
        rnd.seed(123)
        nrepeat = 5
        compr_random_keys_shapes = [('2121', shape),('212221',irregular_shape),('21221', big_shape)]
        # use new keys so residual is 0 for calculation of expected
        for k,s in compr_random_keys_shapes:
            kv.init(k, mx.nd.zeros(s))
        for k,s in compr_random_keys_shapes:
            curr_residual = np.zeros(s)
            for l in range(nrepeat):
                orig_val = mx.nd.zeros(s)
                kv.pull(k, orig_val)

                grad = mx.nd.array(rnd.rand(s[0], s[1]))
                # creates a copy because push changes grad because of assignment
                grad_cpy = mx.nd.array(grad)
                kv.push(k, grad)
                val = mx.nd.zeros(s)
                kv.pull(k, val)

                diff = val - orig_val

                # compute expected by using simulation of operator
                compr, curr_residual, decompr = compute_expected_2bit_quantization(grad_cpy, curr_residual, threshold)
                decompr *= nworker * rate
                assert_almost_equal(diff.asnumpy(), decompr)

    print ('worker '+str(my_rank)+' started with non compression tests')
    check_default_keys(kv, my_rank, nworker)
    check_row_sparse_keys(kv, my_rank, nworker)
    check_row_sparse_keys_with_zeros(kv, my_rank, nworker)
    check_big_row_sparse_keys(kv, my_rank, nworker)
    print('worker ' + str(my_rank) + ' is done with non compression tests')

    # don't run non compressed keys after this as kvstore now is set to compressed
    print ('worker '+str(my_rank)+' started with compression tests')
    kv, threshold = init_kv_compressed(kv)
    check_compr_pull_before_push(kv)
    check_compr_zero(kv)
    check_compr_residual(kv, threshold, nworker)
    check_compr_ones(kv, threshold, nworker)
    check_compr_random(kv, threshold, nworker)
    print('worker ' + str(my_rank) + ' is done with compression tests')

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

if __name__ == "__main__":
    test_sync_init()
    test_sync_push_pull()
