#!/usr/bin/env python
# pylint: skip-file

import mxnet as mx
import numpy as np

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0)

kv = mx.kv.create('dist')
my_rank = kv.get_rank()
world = kv.get_group_size()

shape = (4, 4)
keys = [5, 7, 11]

if my_rank == 0:
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))

kv.barrier()

val = mx.nd.empty(shape)
kv.pull(3, out = val)
check_diff_to_scalar(val, 1)


kv.push(keys, [mx.nd.ones(shape)*my_rank] * len(keys))
kv.pull(3, out = val)
print val.asnumpy()

kv.push(keys, [mx.nd.ones(shape)*my_rank] * len(keys))
kv.pull(3, out = val)
print val.asnumpy()
