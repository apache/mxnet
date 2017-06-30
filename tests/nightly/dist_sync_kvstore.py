#!/usr/bin/env python
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
    kv.init(rsp_keys, [mx.nd.ones(shape)._to_rsp()] * len(rsp_keys))
    kv.init('100', mx.nd.ones(big_shape)._to_rsp())
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
        kv.pull('3', out = val)
        check_diff_to_scalar(val, num)

        val2 = mx.nd.zeros(big_shape)
        kv.pull('99', out = val2)
        check_diff_to_scalar(val2, num)

    def check_row_sparse_keys(kv, my_rank, nworker):
        nrepeat = 3
        # prepare gradient
        v = mx.nd.zeros(shape)
        my_row = my_rank % shape[0]
        v[my_row] = my_rank + 1
        # push
        for i in range(nrepeat):
            kv.push('9', v._to_rsp())
        # pull a subset of rows this worker is interested in
        val = v.copyto(mx.cpu())._to_rsp()
        kv.pull('9', out = val)
        # prepare expected result
        expected =  mx.nd.zeros(shape)
        # initial value
        expected[my_row] = 1
        # apply updates from workers
        for rank in range(nworker):
            row = rank % shape[0]
            if row != my_row:
                continue
            expected[my_row] += (rank + 1) * rate * nrepeat
        # verify results
        check_diff_to_scalar(val, expected)

    def check_row_sparse_keys_with_zeros(kv, my_rank, nworker):
        nrepeat = 3
        # prepare gradient
        v = mx.nd.zeros(shape)
        big_v = mx.nd.zeros(big_shape)
        # push
        for i in range(nrepeat):
            kv.push('11', v._to_rsp())
            kv.push('100', big_v._to_rsp())
        # pull a subset of rows this worker is interested in
        val = mx.nd.ones(shape)._to_rsp()
        big_val = mx.nd.ones(big_shape)._to_rsp()
        kv.pull('11', out = val)
        kv.pull('100', out = big_val)
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
        i = 0
        # each worker chooses a subset of the indices to update
        my_indices = []
        my_step = (my_rank + 1) * 2
        while i < len(indices):
            my_indices.append(indices[i])
            v[indices[i]] = my_rank + 1
            i += my_step
        my_indices = np.array(my_indices)
        # push
        for i in range(nrepeat):
            kv.push('100', v._to_rsp())
        # pull a subset of rows this worker is interested in
        val = v.copyto(mx.cpu())._to_rsp()
        kv.pull('100', out = val)
        # prepare expected result
        expected = mx.nd.zeros(big_shape)
        # initial value
        i = 0
        for i in my_indices:
            expected[i] = 1
        # apply updates from each worker
        for i in range(len(my_indices)):
            for rank in range(nworker):
                step = (rank + 1) * 2
                if (i * my_step) % step == 0:
                    expected[my_indices[i]] += (rank + 1) * rate * nrepeat
        check_diff_to_scalar(val, expected, rank=my_rank)

    check_default_keys(kv, my_rank, nworker)
    check_row_sparse_keys(kv, my_rank, nworker)
    check_row_sparse_keys_with_zeros(kv, my_rank, nworker)
    check_big_row_sparse_keys(kv, my_rank, nworker)
    print('worker ' + str(my_rank) + ' is done')

if __name__ == "__main__":
    test_sync_push_pull()
