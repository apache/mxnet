#!/usr/bin/env python
# pylint: skip-file
#
# run on local machine
# $ ln -s ../../../dmlc-core/tracker/dmlc_local.py .
# $ ./dmlc_local.py -n 4 -s 4 ./test_kvstore.py

def updater(key, recved, stored):
    print "key: %d" & key
    stored += recved * 2

import mxnet as mx
import numpy as np
import time

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), A.asnumpy()

# setup
keys = [3, 5, 7]
rate = 2
shape = (2, 2)
big_shape = (1200, 1200)        # big than BIGARRAY_BOUND

# init
kv = mx.kv.create('dist')
my_rank = kv.get_rank()
nworker = kv.get_group_size()

if my_rank == 0:
    # init key, value on servers
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    kv.init(99, mx.nd.ones(big_shape))
    # init updater on servers
    opt = mx.optimizer.create('test', rate)
    kv.set_optimizer(opt)

kv.barrier()
# print 'init worker %d' % my_rank

def test_sync_push_pull():
    nrepeat = 2
    for i in range(nrepeat):
        kv.push(3, mx.nd.ones(shape)*(my_rank+1))
        kv.push(99, mx.nd.ones(big_shape)*(my_rank+1))

    kv.wait([3, 99])
    num = (nworker + 1 ) * nworker * rate / 2 * nrepeat + 1
    val = mx.nd.zeros(shape)
    kv.pull(3, out = val)
    check_diff_to_scalar(val, num)

    val2 = mx.nd.zeros(big_shape)
    kv.pull(99, out = val2)
    check_diff_to_scalar(val2, num)
    # print val.asnumpy()

# kv.send_command_to_servers(1, 'helo')

# TODO async test, slice,
if __name__ == "__main__":
    test_sync_push_pull()
