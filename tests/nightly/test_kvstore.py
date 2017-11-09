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
shapes = [(4, 4), (100, 100), (2000, 2000)]

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

def test_compress_kvstore(kv_type, compression='2bit', threshold=0.5):
    print(kv_type + ' with ' + compression + ' compression')
    rate = 2
    kv = mx.kv.create(kv_type)
    kv.set_gradient_compression({'compression':compression, 'threshold':threshold})
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    for k, s in zip(keys, shapes):
        kv.init(k, mx.nd.zeros(s))

    def compute_expected(arr, curr_residual, threshold):
        from struct import pack,unpack
        def bits2int(bits):
            bits = [int(x) for x in bits[::-1]]
            x = 0
            for i in range(len(bits)):
                x += bits[i]*2**i
            return x

        def as_float32(s):
            return unpack("f",pack("I", bits2int(s)))[0]

        # str_quant stores the quantized representation as a sequence of bits
        str_quant = ''
        new_residual = []
        decompr = []
        arr_npy = arr.asnumpy()
        curr_res_npy = curr_residual.asnumpy()
        for i, a in np.ndenumerate(arr_npy):
            a += curr_res_npy[i]
            if a >= threshold:
                str_quant += '11'
                new_residual.append(a - threshold)
                decompr.append(threshold)
            elif a <= (-1*threshold):
                str_quant += '10'
                new_residual.append(a + threshold)
                decompr.append(-1*threshold)
            else:
                str_quant += '00'
                new_residual.append(a)
                decompr.append(0)
        # append extra bits when size of array not a factor of 16
        if len(str_quant)%16 != 0:
            str_quant += '0'*(16 - len(str_quant)%16)

        compr = []
        # converts the string generated into integers 32chars at a time
        i = 0
        while i<len(str_quant):
            cur_float = str_quant[i+24:i+32] + str_quant[i+16:i+24] + str_quant[i+8:i+16] + str_quant[i:i+8]
            compr.append(as_float32(cur_float))
            i+=32
        return compr, new_residual, decompr

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

    def verify_residual(kv, threshold, rate):
        for j in range(len(keys)):
            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*0.4 for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            for o in out:
                check_diff_to_scalar(o, 0)

            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*(threshold-0.3) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            curval = threshold * rate * nworker
            for o in out:
                check_diff_to_scalar(o, curval)

            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*(0.2) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            for o in out:
                check_diff_to_scalar(o, curval)

            kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*(threshold-0.3) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j],out=out)
            curval += threshold*rate*nworker
            for o in out:
                check_diff_to_scalar(o, curval)
            # residual would be 0 now
        return curval

    def check_neg(kv, neg, rate, curval):
        for r in range(nrepeat):
            curval = curval + rate*nworker*neg
            for j in range(len(keys)):
                kv.push(keys[j], [mx.nd.ones(shapes[j], mx.gpu(g))*neg for g in range(nworker)])
                out = [mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=out)
                for o in out:
                    check_diff_to_scalar(o, curval)
            # residual would be 0 again

    def check_compr_random(kv, threshold):
        # use new keys so we can track residual
        random_keys = [13, 15, 17]
        for k, s in zip(random_keys, shapes):
            kv.init(k, mx.nd.zeros(s))
        for j in range(len(random_keys)):
            curr_residual = np.zeros(shapes[j])
            for l in range(nrepeat):
                orig_val = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=orig_val)

                grads = [mx.nd.random_uniform(-0.6,0.6, shape=shapes[j], ctx=mx.gpu(g)) for g in range(nworker)]
                kv.push(keys[j], grads)
                val = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=val)

                diffs = [val[g] - orig_val[g] for g in range(nworker)]
                # compute expected by using simulation of operator
                # on cpu
                sum_dequantized_vals = np.zeros(shapes[j])
                for g in range(nworker):
                    compr, curr_residual, decompr = compute_expected(grads[g], curr_residual, threshold)
                    sum_dequantized_vals += ((decompr*rate).asnumpy())

                for g in range(nworker):
                    assert_almost_equal(diffs[g].asnumpy(), sum_dequantized_vals)

    pull_before_push(kv)
    push_zeros(kv)
    curval = verify_residual(kv, threshold, rate)
    check_neg(kv, -1*threshold, rate, curval)
    check_compr_random(kv, threshold)

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
