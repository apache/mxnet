#!/usr/bin/env python
# pylint: skip-file
#
#
#

import mxnet as mx
import numpy as np
import time

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), A.asnumpy()

kv = mx.kv.create('dist')

my_rank = kv.get_rank()
nworker = kv.get_group_size()

shape = (2, 2)
keys = [3]

if my_rank == 0:
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))

kv.barrier()

val = mx.nd.zeros(shape)

# synchronzied push & pull
nrepeat = 3
for i in range(nrepeat):
    kv.push(3, mx.nd.ones(shape)*(my_rank+1))
kv.wait(3)
kv.barrier()

kv.pull(3, out = val)

num = (nworker + 1 ) * nworker / 2 * nrepeat + 1
check_diff_to_scalar(val, num)

# kv.pull(3, out = val)
# print val.asnumpy()

# kv.push(keys, [mx.nd.ones(shape)*(my_rank+2)] * len(keys))
# # kv.push(keys, [mx.nd.ones(shape)*(my_rank+2)] * len(keys))
# kv.pull(3, out = val)
# print val.asnumpy()

# kv.push(keys, [mx.nd.ones(shape)*(my_rank+3)] * len(keys))
# kv.pull(3, out = val)
# print val.asnumpy()

# kv.push(keys, [mx.nd.ones(shape)*(my_rank+4)] * len(keys))
# kv.pull(3, out = val)
# print val.asnumpy()
