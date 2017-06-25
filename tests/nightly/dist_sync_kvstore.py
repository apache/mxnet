#!/usr/bin/env python
# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import time

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), A.asnumpy()

# setup
keys = ['3', '5', '7']
rsp_keys = ['9', '11', '13']

rate = 2
shape = (2, 2)
big_shape = (1200, 1200)        # big than BIGARRAY_BOUND


def init_kv():
    kv = mx.kv.create('dist_sync')
    # init kv
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    kv.init('99', mx.nd.ones(big_shape))
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    return kv, my_rank, nworker

def init_kv_rsp():
    kv = mx.kv.create('dist_sync')
    # init kv
    kv.init(rsp_keys, [mx.nd.ones(shape).to_rsp()] * len(rsp_keys))
    # kv.init(99, mx.nd.ones(big_shape))
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    return kv, my_rank, nworker

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()
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
    print('done')

def test_sync_push_pull_row_sparse():
    kv, my_rank, nworker = init_kv_rsp()
    nrepeat = 2

    v = mx.nd.zeros(shape)
    my_row = my_rank % shape[0]
    for col in range(shape[1]):
        v[my_row][col] = my_rank + 1

    for i in range(nrepeat):
        kv.push('9', v.to_rsp())
        # kv.push(99, mx.nd.ones(big_shape)*(my_rank+1))

    # pull a subset of rows this worker is interested in
    val = v.copyto(mx.cpu()).to_rsp()
    kv.pull('9', out = val)

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
    # print('done')
    #val2 = mx.nd.zeros(big_shape)
    #kv.pull(99, out = val2)
    #check_diff_to_scalar(val2, num)

if __name__ == "__main__":
    test_sync_push_pull()
    test_sync_push_pull_row_sparse()
