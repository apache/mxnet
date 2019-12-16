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

# pylint: skip-file
import mxnet as mx
import numpy as np
import unittest
from mxnet.test_utils import rand_ndarray, assert_almost_equal
from common import setup_module, with_seed, assertRaises, teardown
from mxnet.base import py_str, MXNetError

shape = (4, 4)
keys = [5, 7, 11]
str_keys = ['b', 'c', 'd']

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), (A, x)

def init_kv(name='device'):
    return mx.kv.create(name)

@with_seed()
def test_broadcast_single_kv_pair():
    """single key-value pair push & pull"""
    def check_single_kv_pair(kv, key):
        # single output
        ones = mx.nd.ones(shape)
        out = mx.nd.empty(shape)
        kv.broadcast(key, ones, out)
        check_diff_to_scalar(out, 1)
        # list output
        out_list = [mx.nd.empty(shape)] * 3
        key_list = key + key
        kv.broadcast(key_list, ones, out_list)
        for o in out_list:
            check_diff_to_scalar(o, 1)

    for name in ['device', 'teststore']:
        check_single_kv_pair(init_kv(name), 3)
        check_single_kv_pair(init_kv(name), 'a')

@with_seed()
def test_broadcast_list_kv_pair():
    """list key-value pair push & pull"""
    def check_list_kv_pair(kv, key):
        ones = [mx.nd.ones(shape)] * len(key)
        out = [mx.nd.empty(shape)] * len(key)
        kv.broadcast(key, ones, out)
        for o in out:
            check_diff_to_scalar(o, 1)
        out_list = [[mx.nd.empty(shape)] * 2 for _ in range(len(key))]
        key_list = [k + k for k in key]
        kv.broadcast(key_list, ones, out_list)
        for o in out_list:
            for oo in o:
                check_diff_to_scalar(oo, 1)

    check_list_kv_pair(init_kv(), keys)
    check_list_kv_pair(init_kv(), str_keys)

@with_seed()
def test_pushpull_single_kv_pair():
    """aggregate value on muliple devices"""
    def check_aggregator(kv, key, key_list=None):
        kv.broadcast(key, mx.nd.zeros(shape), out=mx.nd.empty(shape))
        # devices
        num_devs = 4
        devs = [mx.Context('cpu', i) for i in range(num_devs)]

        # single
        vals = [mx.nd.ones(shape, d) for d in devs]
        outs = [mx.nd.empty(shape, d) for d in devs]

        kv.pushpull(key, vals, out=outs)
        for out in outs:
            check_diff_to_scalar(out, num_devs)

        # inplace
        kv.pushpull(key, vals)
        for val in vals:
            check_diff_to_scalar(val, num_devs)

        # list
        if key_list is None:
            return
        num_keys = len(key_list)
        kv.broadcast(key_list, [mx.nd.zeros(shape)] * num_keys,
                     out=[mx.nd.empty(shape)] * num_keys)
        vals = [[mx.nd.ones(shape, d)*2.0 for d in devs]] * num_keys
        outs = [[mx.nd.empty(shape, d) for d in devs]] * num_keys
        kv.pushpull(key_list, vals, out=outs)
        for out in outs:
            for o in out:
                check_diff_to_scalar(o, num_devs * 2.0)

        # inplace
        kv.pushpull(key_list, vals)
        for val in vals:
            for v in val:
                check_diff_to_scalar(v, num_devs * 2.0)

    check_aggregator(init_kv('device'), 3, keys)
    check_aggregator(init_kv('device'), 'a', str_keys)
    check_aggregator(init_kv('teststore'), 3)
    check_aggregator(init_kv('teststore'), 'a')

@with_seed()
def test_pushpull_list_kv_pair():
    """aggregate value on muliple devices"""
    def check_aggregator(kv, key, key_list=None):
        kv.broadcast(key, mx.nd.zeros(shape), out=mx.nd.empty(shape))
        # devices
        num_devs = 4
        devs = [mx.Context('cpu', i) for i in range(num_devs)]

        # single
        vals = [mx.nd.ones(shape, d) for d in devs]
        outs = [mx.nd.empty(shape, d) for d in devs]

        kv.pushpull(key, vals, out=outs)
        for out in outs:
            check_diff_to_scalar(out, num_devs)

        # list
        if key_list is None:
            return
        num_keys = len(key_list)
        kv.broadcast(key_list, [mx.nd.zeros(shape)] * num_keys,
                     out=[mx.nd.empty(shape)] * num_keys)
        vals = [[mx.nd.ones(shape, d)*2.0 for d in devs]] * num_keys
        outs = [[mx.nd.empty(shape, d) for d in devs]] * num_keys
        kv.pushpull(key_list, vals, out=outs)
        for out in outs:
            for o in out:
                check_diff_to_scalar(o, num_devs * 2.0)

    check_aggregator(init_kv('device'), 3, keys)
    check_aggregator(init_kv('device'), 'a', str_keys)
    check_aggregator(init_kv('teststore'), 3)
    check_aggregator(init_kv('teststore'), 'a')


@with_seed()
def test_custom_store():
    kv = mx.kv.create('teststore')
    out = mx.nd.empty((1,))
    kv.broadcast(1, mx.nd.ones((1,)), out=out)
    check_diff_to_scalar(out, 1)
    assert type(kv).is_capable('optimizer') == False
    kv.broadcast(1, mx.nd.ones((1,)), out=out)
    check_diff_to_scalar(out, 1)
    arr_list = [mx.nd.empty((1,))] * 2
    kv.pushpull(1, [mx.nd.ones((1,))] * 2, out=arr_list)
    for arr in arr_list:
        check_diff_to_scalar(arr, 2)
    kv.pushpull(1, arr_list)
    for arr in arr_list:
        check_diff_to_scalar(arr, 4)

@with_seed()
def test_get_type_device():
    kvtype = 'teststore'
    kv = mx.kv.create(kvtype)
    assert kv.type == kvtype

@with_seed()
def test_set_optimizer():
    def check_unsupported_methods(kv):
        assert not kv.is_capable('optimizer')
        optimizer = mx.optimizer.create('sgd')
        assertRaises(NotImplementedError, kv.set_optimizer, optimizer)
        assertRaises(NotImplementedError, kv.save_optimizer_states, 'test')
        assertRaises(NotImplementedError, kv.load_optimizer_states, 'test')

    kv = mx.kv.create('teststore')
    check_unsupported_methods(kv)

if __name__ == '__main__':
    import nose
    nose.runmodule()
