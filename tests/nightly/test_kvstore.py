#!/usr/bin/env python
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np

keys = [3, 5, 7]
# let the last shape exceed MXNET_KVSTORE_BIGARRAY_BOUND
shapes = [(4, 4), (100, 100), (2000, 2000)];

lr = .1
nworker = 4
nrepeat = 10

## generate data
data = [[[np.random.random(s)*2-1 for i in range(nworker)] for s in shapes] for j in range(nrepeat)]

## individual key interface
def test_kvstore(kv_type):
    print(kv_type)
    kv = mx.kv.create(kv_type)
    kv.set_optimizer(mx.optimizer.create('test', lr))
    for k, s in zip(keys, shapes):
        kv.init(k, mx.nd.zeros(s))

    res = [np.zeros(s) for s in shapes]
    for i in range(nrepeat):
        for j in range(len(keys)):
            kv.push(keys[j], [mx.nd.array(
                data[i][j][g], mx.gpu(g)) for g in range(nworker)])

        res = [a + b * lr for a, b in zip(res, [sum(d) for d in data[i]])]
        for j in range(len(keys)):
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=out)
            err = [np.sum(np.abs(o.asnumpy() - res[j])) for o in out]
            err = sum(err) / np.sum(np.abs(res[j]))
            assert(err < 1e-6), (err, shapes[j])

test_kvstore('local_update_cpu')
test_kvstore('local_allreduce_cpu')
test_kvstore('local_allreduce_device')

## group keys interface
def test_group_kvstore(kv_type):
    print(kv_type)
    kv = mx.kv.create(kv_type)
    kv.set_optimizer(mx.optimizer.create('test', lr))
    kv.init(keys, [mx.nd.zeros(s) for s in shapes])
    res = [np.zeros(s) for s in shapes]
    out = [[mx.nd.zeros(s, mx.gpu(g)) for g in range(nworker)] for s in shapes]
    for i in range(nrepeat):
        kv.push(keys, [[
            mx.nd.array(data[i][j][g], mx.gpu(g)) for g in range(nworker)]
                       for j in range(len(keys))])

        kv.pull(keys, out=out)
        res = [a + b * lr for a, b in zip(res, [sum(d) for d in data[i]])]
        for a, b in zip(res, out):
            err = [np.sum(np.abs(o.asnumpy() - a)) for o in b]
            err = sum(err) / np.sum(np.abs(a))
            assert(err < 1e-6), (err, a.shape)

test_group_kvstore('local_update_cpu')
test_group_kvstore('local_allreduce_cpu')
test_group_kvstore('local_allreduce_device')
