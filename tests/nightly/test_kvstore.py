#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
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
import numpy.random as rnd
import copy

from mxnet.test_utils import assert_almost_equal


def check_diff_to_scalar(A, x, rank=None):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x)

def compute_1bit(arr, curr_residual, threshold):
    str_quant = ""
    new_residual = []
    decompr = []

    for idx, val in np.ndenumerate(arr):
        val += curr_residual[idx]
        if val > threshold:
            str_quant += "1"
            new_residual.append(val - 1)
            decompr.append(1)
        else:
            str_quant += "0"
            new_residual.append(val + 1)
            decompr.append(-1)

    # append extra bits when size of array not a factor of 32
    if len(str_quant) != 32:
        str_quant += "0" * (32 - len(str_quant) % 32)
    return str_quant, new_residual, decompr

def compute_2bit(arr, curr_residual, threshold):
    str_quant = ""
    new_residual = []
    decompr = []

    for idx, val in np.ndenumerate(arr):
        val += curr_residual[idx]
        if val >= threshold:
            str_quant += "11"
            new_residual.append(val - threshold)
            decompr.append(threshold)
        elif val <= -threshold:
            str_quant += "10"
            new_residual.append(val + threshold)
            decompr.append(-threshold)
        else:
            str_quant += "00"
            new_residual.append(val)
            decompr.append(0)

    # append extra bits when size of array not a factor of 16
    if len(str_quant) % 16 != 0:
        str_quant += "0" * (16 - len(str_quant) % 16)
    return str_quant, new_residual, decompr

def compute_expected_quantization(arr, curr_residual, threshold, quantize_func):

    from struct import pack,unpack
    def as_float32(s):
        return unpack("f",pack("I", int(s, 2)))[0]

    arr_npy = arr.asnumpy()
    # str_quant stores the quantized representation as a sequence of bits
    str_quant, new_residual, decompr = quantize_func(arr_npy, curr_residual, threshold)
    
    compr = []
    # converts the string generated into integers 32chars at a time
    for i in range(0, len(str_quant), 32):
        cur_float = str_quant[i+24:i+32] + str_quant[i+16:i+24] + str_quant[i+8:i+16] + str_quant[i:i+8]
        compr.append(as_float32(cur_float))

    return np.array(compr), np.array(new_residual).reshape(arr.shape), np.array(decompr).reshape(arr.shape)

## individual key interface
def test_kvstore(kv_type, stype):
    print(kv_type)
    kv = mx.kv.create(kv_type)
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=lr))
    for k, s in zip(keys, shapes):
        kv.init(k, mx.nd.zeros(s))

    res = [np.zeros(s) for s in shapes]
    for i in range(nrepeat):
        for j in range(len(keys)):
            kv.push(keys[j], [mx.nd.array(
                data[i][j][g], mx.gpu(g)).tostype(stype) for g in range(nworker)])

        res = [a + b * lr for a, b in zip(res, [sum(d) for d in data[i]])]
        for j in range(len(keys)):
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=out)
            err = [np.sum(np.abs(o.asnumpy() - res[j])) for o in out]
            err = sum(err) / np.sum(np.abs(res[j]))
            assert(err < 1e-6), (err, shapes[j])

def test_compress_kvstore(kv_type, compression='2bit', threshold=0.5):
    print(kv_type + ' with ' + compression + ' compression and threshold is ' + str(threshold))
    rate = 2
    quantize_func = None
    if compression == '1bit':
        quantize_func = compute_1bit
    elif compression == '2bit':
        quantize_func = compute_2bit
    else:
        raise RuntimeError("Unknown gradient compression type!")
    kv = mx.kv.create(kv_type)
    kv.set_gradient_compression({'type':compression, 'threshold':threshold})
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    for k, s in zip(keys, shapes):
        kv.init(k, mx.nd.zeros(s))
    # init one key with 1s so we can check if it was compressed during init
    kv.init(gc_init_test_key, mx.nd.ones(shapes[0]))
    # use different keys for random tests so that
    # we can track residual from start
    random_keys = [13, 15, 17]
    for k, s in zip(random_keys, shapes):
        kv.init(k, mx.nd.zeros(s))

    def pull_init_test(kv):
        # checks that compression is not applied to init of key
        out = [mx.nd.zeros(shapes[0], mx.gpu(g)) for g in range(nworker)]
        kv.pull(gc_init_test_key, out=out)
        exp = np.ones_like(out[0].asnumpy())
        for o in out:
            assert_almost_equal(o.asnumpy(), exp)

    def pull_before_push(kv):
        for i in range(nrepeat):
            for j in range(len(keys)):
                out = [mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=out)
                exp = np.zeros_like(out[0].asnumpy())
                for o in out:
                    assert_almost_equal(o.asnumpy(), exp)

    def push_ones(kv, sign=1):
        for i in range(nrepeat):
            for j in range(len(keys)):
                kv.push(keys[j], [sign * mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)])
                out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=out)
                if sign == 1:
                    exp = (i + 1) * rate * nworker * np.ones_like(out[0].asnumpy())
                else:
                    exp = (nrepeat - i - 1) * rate * nworker * np.ones_like(out[0].asnumpy())
                for o in out:
                    assert_almost_equal(o.asnumpy(), exp)

    def verify_residual_1bit(kv, threshold, rate):
        # current values must equal to zero
        for j in range(len(keys)):
            out = [mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=out)
            exp = np.zeros_like(out[0].asnumpy())
            for o in out:
                assert_almost_equal(o.asnumpy(), exp)
        
        curr_residual = 0
        curr_val = rate * nworker if 2 > threshold else -rate * nworker
        for j in range(len(keys)):
            kv.push(keys[j], [2 * mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=out)
            
            for o in out:
                check_diff_to_scalar(o, curr_val)

        curr_residual = 1 if 2 > threshold else 3
        curr_val += rate * nworker if 0 + curr_residual > threshold else -rate * nworker
        for j in range(len(keys)):
            kv.push(keys[j], [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=out)
            for o in out:
                check_diff_to_scalar(o, curr_val)

        curr_residual += -1 if curr_residual > threshold else +1
        curr_val += rate * nworker if -2 + curr_residual > threshold else -rate * nworker
        for j in range(len(keys)):
            kv.push(keys[j], [-2 * mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)])
            out = [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)]
            kv.pull(keys[j], out=out)
            for o in out:
                check_diff_to_scalar(o, curr_val)
    
    def push_zeros(kv):
        for i in range(nrepeat):
            for j in range(len(keys)):
                kv.push(keys[j], [mx.nd.zeros(shapes[j], mx.gpu(g)) for g in range(nworker)])
                out = [mx.nd.ones(shapes[j], mx.gpu(g)) for g in range(nworker)]
                kv.pull(keys[j], out=out)
                exp = np.zeros_like(out[0].asnumpy())
                for o in out:
                    assert_almost_equal(o.asnumpy(), exp)

    def verify_residual_2bit(kv, threshold, rate):
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
        for k, s in zip(random_keys, shapes):
            curr_residual = [np.zeros(s) for g in range(nworker)]
            orig_val = [mx.nd.zeros(s, mx.gpu(g)) for g in range(nworker)]
            kv.pull(k, out=orig_val)
            grads = [mx.nd.random_uniform(-0.6, 0.6, shape=s, ctx=mx.gpu(g)) for g in range(nworker)]
            grads_cpy = copy.deepcopy(grads)
            kv.push(k, grads)
            val = [mx.nd.zeros(s, mx.gpu(g)) for g in range(nworker)]
            kv.pull(k, out=val)
            diffs = [val[g] - orig_val[g] for g in range(nworker)]
            # compute expected by using simulation of operator
            # on cpu
            sum_dequantized_vals = np.zeros(s)
            for g in range(nworker):
                compr, curr_residual[g], decompr = compute_expected_quantization(
                                                    grads_cpy[g], curr_residual[g], threshold, quantize_func)
                sum_dequantized_vals += (decompr * rate)

            for g in range(nworker):
                assert_almost_equal(diffs[g].asnumpy(), sum_dequantized_vals)

    pull_init_test(kv)
    pull_before_push(kv)
    if compression == '1bit':
        push_ones(kv, sign=1)
        push_ones(kv, sign=-1)
        verify_residual_1bit(kv, threshold, rate)
    elif compression == '2bit':
        push_zeros(kv)
        curval = verify_residual_2bit(kv, threshold, rate)
        check_neg(kv, -1*threshold, rate, curval)
    check_compr_random(kv, threshold)

## group keys interface
def test_group_kvstore(kv_type, stype):
    print(kv_type)
    kv = mx.kv.create(kv_type)
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=lr))
    kv.init(keys, [mx.nd.zeros(s) for s in shapes])
    res = [np.zeros(s) for s in shapes]
    out = [[mx.nd.zeros(s, mx.gpu(g)) for g in range(nworker)] for s in shapes]
    for i in range(nrepeat):
        kv.push(keys, [[
            mx.nd.array(data[i][j][g], mx.gpu(g)).tostype(stype) for g in range(nworker)]
                       for j in range(len(keys))])

        kv.pull(keys, out=out)
        res = [a + b * lr for a, b in zip(res, [sum(d) for d in data[i]])]
        for a, b in zip(res, out):
            err = [np.sum(np.abs(o.asnumpy() - a)) for o in b]
            err = sum(err) / np.sum(np.abs(a))
            assert(err < 1e-6), (err, a.shape)

if __name__ == "__main__":
    keys = [3, 5, 7]
    # let the last shape exceed MXNET_KVSTORE_BIGARRAY_BOUND
    shapes = [(4, 4), (100, 100), (2000, 2000)]
    stypes = ['default', 'row_sparse']

    gc_init_test_key = 9

    lr = .1
    nworker = 4
    nrepeat = 10

    # generate data
    data = [[[np.random.random(s)*2-1 for i in range(nworker)] for s in shapes] for j in range(nrepeat)]

    for stype in stypes:
        test_kvstore('local_update_cpu', stype)
        test_kvstore('local_allreduce_cpu', stype)
        test_kvstore('local_allreduce_device', stype)

    ## compression for local kvstore happens only when reduce is on device
    test_compress_kvstore('local_allreduce_device', '1bit', -.5)
    test_compress_kvstore('local_allreduce_device', '1bit', 0)
    test_compress_kvstore('local_allreduce_device', '1bit', .5)
    test_compress_kvstore('local_allreduce_device', '2bit', .5)
    for stype in stypes:
        test_group_kvstore('local_update_cpu', stype)
        test_group_kvstore('local_allreduce_cpu', stype)
        test_group_kvstore('local_allreduce_device', stype)
