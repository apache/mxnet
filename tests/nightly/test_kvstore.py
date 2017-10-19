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

import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
from mxnet.test_utils import assert_almost_equal

def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

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
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=lr))
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

def test_compress_kvstore(kv_type, compress='2bit', neg=-0.5, pos=0.5):
    print(kv_type + ' with ' + compress + ' compression')
    rate = 2
    kv = mx.kv.create(kv_type)
    kv.set_compress({'compress':compress, 'neg_threshold':neg, 'pos_threshold':pos})
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    for k, s in zip(keys, shapes):
        kv.init(k, mx.nd.zeros(s))

    def pull_before_push(kv):
        for i in range(nrepeat):
            for j in range(len(keys)):
                out = [mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=out)
                exp = np.zeros_like(out[0].asnumpy())
                for o in out:
                    assert_almost_equal(o.asnumpy(), exp)

    def push_zeros(kv):
        for i in range(nrepeat):
            for j in range(len(keys)):
                kv.push(keys[j], [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)])
                out = [mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=out)
                exp = np.zeros_like(out[0].asnumpy())
                for o in out:
                    assert_almost_equal(o.asnumpy(), exp)

    def verify_residual(kv, neg_threshold, pos_threshold, rate):
        for j in range(len(keys)):
            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*0.4 for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            for o in out:
                check_diff_to_scalar(o, 0)

            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*(pos_threshold-0.3) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            curval = pos_threshold * rate * nworker
            for o in out:
                check_diff_to_scalar(o, curval)

            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*(0.2) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            for o in out:
                check_diff_to_scalar(o, curval)

            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*(pos_threshold-0.3) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            curval += pos_threshold*rate*nworker
            for o in out:
                check_diff_to_scalar(o, curval)
            # residual would be 0 now
        return curval

    def check_ones(kv, pos, rate, curval):
        newval = curval + rate*nworker*pos
        for j in range(len(keys)):
            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*pos for g in range(nworker)])
            out = [mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=out)
            for o in out:
                check_diff_to_scalar(o, newval)
        # residual would be 0 again

    def check_compr_random(kv, pos, neg):
        for j in range(len(keys)):
            orig_val = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=orig_val)
            grads = [mx.nd.random_uniform(-0.6,0.6, shape=shapes[j], ctx=mx.gpu(g)) for g in range(nworker)]
            kv.push(keys[j], grads)
            val = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=val)
            diff = [val[g] - orig_val[g] for g in range(nworker)]

            # compute expected by directly using operators
            comprs = []
	    decomprs = []
            # on cpu
            sum_dequantized_vals = np.zeros(shapes[j])
            for g in range(nworker):
                comprs.append(mx.contrib.nd.create_2bit(grads[g]))
                decomprs.append(mx.nd.zeros(grads[g].shape, ctx=mx.gpu(g)))
                mx.contrib.ndarray.quantize_2bit(grads[g], mx.nd.zeros(shapes[j], ctx=mx.gpu(g)), comprs[g], neg, pos)
                mx.contrib.ndarray.dequantize_2bit(comprs[g], decomprs[g])
                sum_dequantized_vals += ((decomprs[g]*rate).asnumpy())
            for g in range(nworker):
                assert_almost_equal(diff[g].asnumpy(), sum_dequantized_vals)

    pull_before_push(kv)
    push_zeros(kv)
    curval = verify_residual(kv, neg, pos, rate)
    check_ones(kv, pos, rate, curval)
    check_compr_random(kv, pos, neg)

test_kvstore('local_update_cpu')
test_kvstore('local_allreduce_cpu')
test_kvstore('local_allreduce_device')

# compression for local kvstore happens only when reduce is on device
test_compress_kvstore('local_allreduce_device')

## group keys interface
def test_group_kvstore(kv_type):
    print(kv_type)
    kv = mx.kv.create(kv_type)
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=lr))
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
