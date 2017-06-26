#!/usr/bin/env python
# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import time

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (A.asnumpy(), x)

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

        num = (nworker + 1 ) * nworker * rate / 2 * nrepeat + 1
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
        for col in range(shape[1]):
            v[my_row][col] = my_rank + 1
        # push
        for i in range(nrepeat):
            kv.push('9', v._to_rsp())
        # pull a subset of rows this worker is interested in
        val = v.copyto(mx.cpu())._to_rsp()
        kv.pull('9', out = val)
        # prepare expected result
        expected =  mx.nd.zeros(shape)
        # initial value
        for col in range(shape[1]):
            expected[my_row][col] = 1
        # apply updates from workers
        for rank in range(nworker):
            row = rank % shape[0]
            if row != my_row:
                continue
            for col in range(shape[1]):
                expected[my_row][col] += (rank + 1) * rate * nrepeat
        #print("expect ", expected.asnumpy())
        check_diff_to_scalar(val, expected)

    check_default_keys(kv, my_rank, nworker)
    check_row_sparse_keys(kv, my_rank, nworker)
    print('done')

if __name__ == "__main__":
    test_sync_push_pull()
