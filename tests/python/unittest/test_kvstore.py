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

def init_kv(stype='default'):
    """init kv """
    kv = mx.kv.create()
    # single
    kv.init(3, mx.nd.zeros(shape=shape, stype=stype))
    # list
    kv.init(keys, [mx.nd.zeros(shape=shape, stype=stype)] * len(keys))
    return kv

def init_kv_with_str(stype='default'):
    """init kv """
    kv = mx.kv.create()
    # single
    kv.init('a', mx.nd.zeros(shape, stype=stype))
    # list
    kv.init(str_keys, [mx.nd.zeros(shape=shape, stype=stype)] * len(keys))
    return kv


def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0)


@with_seed()
def test_single_kv_pair():
    """single key-value pair push & pull"""
    def check_single_kv_pair(kv, key, stype):
        kv.push(key, mx.nd.ones(shape).tostype(stype))
        val = mx.nd.empty(shape)
        kv.pull(key, out=val)
        check_diff_to_scalar(val, 1)

    stypes = ['default', 'row_sparse']
    for stype in stypes:
        check_single_kv_pair(init_kv(), 3, stype)
        check_single_kv_pair(init_kv_with_str(), 'a', stype)

@with_seed()
def test_row_sparse_pull():
    kv = init_kv_with_str('row_sparse')
    kv.init('e', mx.nd.ones(shape).tostype('row_sparse'))

    def check_row_sparse_pull(kv, count):
        num_rows = shape[0]
        vals = []
        row_ids = []
        all_row_ids = np.arange(num_rows)
        for i in range(count):
            vals.append(mx.nd.zeros(shape).tostype('row_sparse'))
            row_id = np.random.randint(num_rows, size=num_rows)
            row_ids.append(mx.nd.array(row_id).reshape((2, num_rows//2)))
        row_ids_to_pull = row_ids[0] if len(row_ids) == 1 else row_ids
        vals_to_pull = vals[0] if len(vals) == 1 else vals

        kv.row_sparse_pull('e', out=vals_to_pull, row_ids=row_ids_to_pull)
        for val, row_id in zip(vals, row_ids):
            retained = val.asnumpy()
            excluded_row_ids = np.setdiff1d(all_row_ids, row_id.asnumpy())
            for row in range(num_rows):
                expected_val = np.zeros_like(retained[row])
                expected_val += 0 if row in excluded_row_ids else 1
                assert_almost_equal(retained[row], expected_val)

    check_row_sparse_pull(kv, 1)
    check_row_sparse_pull(kv, 4)

@with_seed()
def test_init():
    """test init"""
    def check_init(kv, key):
        kv.init(key, mx.nd.ones(shape)*4)
        a = mx.nd.zeros(shape)
        kv.pull(key, out=a)
        check_diff_to_scalar(a, 4)

    check_init(mx.kv.create(), 3)
    check_init(mx.kv.create(), 'a')

@with_seed()
def test_pull():
    """test pull"""
    def check_pull(kv):
        a = mx.nd.ones(shape)
        b = mx.nd.zeros(shape)
        kv.init('1', mx.nd.zeros(shape))
        kv.push('1', [a,a,a,a])
        kv.pull('1', b)
        check_diff_to_scalar(b, 4)
        kv.init('2', mx.nd.zeros(shape))
        kv.pull('2', b)
        check_diff_to_scalar(b, 0)

    check_pull(mx.kv.create('device'))
    check_pull(mx.kv.create())

@with_seed()
def test_list_kv_pair():
    """list key-value pair push & pull"""
    def check_list_kv_pair(kv, key, stype):
        kv.push(key, [mx.nd.ones(shape).tostype(stype)*4] * len(key))
        val = [mx.nd.empty(shape)] * len(key)
        kv.pull(key, out=val)
        for v in val:
            check_diff_to_scalar(v, 4)

    stypes = ['default', 'row_sparse']
    for stype in stypes:
        check_list_kv_pair(init_kv(), keys, stype)
        check_list_kv_pair(init_kv_with_str(), str_keys, stype)


@with_seed()
def test_aggregator():
    """aggregate value on muliple devices"""

    def check_aggregator(kv, key, key_list, stype):
        # devices
        num_devs = 4
        devs = [mx.Context('cpu', i) for i in range(num_devs)]

        # single
        vals = [mx.nd.ones(shape, d).tostype(stype) for d in devs]
        outs = [mx.nd.empty(shape, d) for d in devs]

        kv.push(key, vals)
        kv.pull(key, out=outs)

        for out in outs:
            check_diff_to_scalar(out, num_devs)

        # list
        vals = [[mx.nd.ones(shape, d).tostype(stype)*2.0 for d in devs]] * len(key_list)
        outs = [[mx.nd.empty(shape, d) for d in devs]] * len(key_list)
        kv.push(key_list, vals)
        kv.pull(key_list, out=outs)

        for out in outs:
            for o in out:
                check_diff_to_scalar(o, num_devs * 2.0)

    stypes = ['default', 'row_sparse']
    for stype in stypes:
        check_aggregator(init_kv(), 3, keys, stype)
        check_aggregator(init_kv_with_str(), 'a', str_keys, stype)


@with_seed()
def test_sparse_aggregator():
    """aggregate sparse ndarray on muliple devices"""
    def check_sparse_aggregator(sparse_pull):
        stype = 'row_sparse'
        kv = init_kv_with_str(stype)

        # devices
        num_devs = 4
        devs = [mx.Context('cpu', i) for i in range(num_devs)]

        # single
        vals = [rand_ndarray(shape, stype).copyto(devs[i]) for i in range(num_devs)]
        expected_sum = np.zeros(shape)
        for v in vals:
            expected_sum += v.asnumpy()

        # prepare row_ids
        kv.push('a', vals)
        if sparse_pull:
            all_rows = mx.nd.array(np.arange(shape[0]))
            kv.row_sparse_pull('a', out=vals, row_ids=[all_rows] * len(vals))
        else:
            kv.pull('a', out=vals, ignore_sparse=False)
        result_sum = np.zeros(shape)
        for v in vals:
            result_sum += v.asnumpy()
        assert_almost_equal(result_sum, expected_sum * num_devs)

        # list
        vals = [[rand_ndarray(shape, stype).copyto(devs[i]) for i in range(num_devs)]] * len(keys)
        expected_sum = np.zeros(shape)
        for v in vals[0]:
            expected_sum += v.asnumpy()

        kv.push(str_keys, vals)
        if sparse_pull:
            kv.row_sparse_pull(str_keys, out=vals, row_ids=[[all_rows] * num_devs] * len(vals))
        else:
            kv.pull(str_keys, out=vals, ignore_sparse=False)
        for vv in vals:
            result_sum = np.zeros(shape)
            for v in vv:
                result_sum += v.asnumpy()
            assert_almost_equal(result_sum, expected_sum * num_devs)

    check_sparse_aggregator(False)
    check_sparse_aggregator(True)

def updater(key, recv, local):
    """use updater: += with int keys"""
    assert(isinstance(key, int))
    local += recv

def str_updater(key, recv, local):
    """use updater: += with str keys"""
    if isinstance(key, bytes):
        key = py_str(key)
    assert(isinstance(key, str))
    local += recv

@with_seed()
def test_updater(dev='cpu'):
    """updater"""

    def check_updater(kv, key, key_list, stype):
        # devices
        num_devs = 4
        devs = [mx.Context(dev, i) for i in range(num_devs)]

        # single
        vals = [mx.nd.ones(shape, d).tostype(stype) for d in devs]
        outs = [mx.nd.empty(shape, d) for d in devs]

        kv.push(key, vals)
        kv.pull(key, out=outs)

        for out in outs:
            check_diff_to_scalar(out, num_devs)

        # list
        vals = [[mx.nd.ones(shape, d).tostype(stype) for d in devs]] * len(key_list)
        outs = [[mx.nd.empty(shape, d) for d in devs]] * len(key_list)

        num_push = 4
        for i in range(num_push):
            kv.push(key_list, vals)

        kv.pull(key_list, out=outs)

        for out in outs:
            for o in out:
                check_diff_to_scalar(o, num_devs * num_push)

    stypes = ['default', 'row_sparse']
    for stype in stypes:
        kv = init_kv()
        kv._set_updater(updater)
        check_updater(kv, 3, keys, stype)

        str_kv = init_kv_with_str()
        str_kv._set_updater(str_updater)
        check_updater(str_kv, 'a', str_keys, stype)

@with_seed()
def test_get_type():
    kvtype = 'local_allreduce_cpu'
    kv = mx.kv.create(kvtype)
    assert kv.type == kvtype

@with_seed()
def test_invalid_pull():
    def check_ignored_pull_single(kv, key):
        dns_val = (mx.nd.ones(shape) * 2)
        rsp_val = dns_val.tostype('row_sparse')
        kv.pull(key, out=rsp_val)
        check_diff_to_scalar(rsp_val, 2)

    def check_ignored_pull_list(kv, key):
        dns_val = [mx.nd.ones(shape) * 2] * len(key)
        rsp_val = [val.tostype('row_sparse') for val in dns_val]
        kv.pull(key, out=rsp_val)
        for v in rsp_val:
            check_diff_to_scalar(v, 2)

    def check_invalid_rsp_pull_single(kv, key):
        dns_val = mx.nd.ones(shape) * 2
        assertRaises(MXNetError, kv.row_sparse_pull,
                     key, out=dns_val, row_ids=mx.nd.array([1]))

    def check_invalid_rsp_pull_list(kv, key):
        dns_val = [mx.nd.ones(shape) * 2] * len(key)
        assertRaises(MXNetError, kv.row_sparse_pull, key, out=dns_val,
                     row_ids=[mx.nd.array([1])] * len(key))

    def check_invalid_key_types_single(kv, key):
        dns_val = mx.nd.ones(shape) * 2
        rsp_val = dns_val.tostype('row_sparse')
        assertRaises(MXNetError, kv.init, key, dns_val)
        assertRaises(MXNetError, kv.push, key, dns_val)
        assertRaises(MXNetError, kv.pull, key, dns_val)
        assertRaises(MXNetError, kv.row_sparse_pull, key, rsp_val,
                     row_ids=mx.nd.array([1]))

    def check_invalid_key_types_list(kv, key):
        dns_val = [mx.nd.ones(shape) * 2] * len(key)
        rsp_val = [val.tostype('row_sparse') for val in dns_val]
        assertRaises(MXNetError, kv.init, key, dns_val)
        assertRaises(MXNetError, kv.push, key, dns_val)
        assertRaises(MXNetError, kv.pull, key, dns_val)
        assertRaises(MXNetError, kv.row_sparse_pull, key, rsp_val,
                         row_ids=[mx.nd.array([1])] * len(key))

    int_kv = init_kv()
    str_kv = init_kv_with_str()

    kvs = [int_kv, str_kv]
    single_keys = [3, 'a']
    list_keys = [keys, str_keys]
    for i in range(2):
        # pull with rsp outputs should be ignored with no values updated
        check_ignored_pull_single(kvs[i], single_keys[i])
        check_ignored_pull_list(kvs[i], list_keys[i])
        # row_sparse_pull should be aborted when vals.stype != row_sparse
        check_invalid_rsp_pull_single(kvs[i], single_keys[i])
        check_invalid_rsp_pull_list(kvs[i], list_keys[i])
        # kvstore should be restricted to only accept either int or str keys
        check_invalid_key_types_single(kvs[i], single_keys[1 - i])
        check_invalid_key_types_list(kvs[i], list_keys[1 - i])

if __name__ == '__main__':
    import nose
    nose.runmodule()
