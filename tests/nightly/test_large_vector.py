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

import numpy as np
import mxnet as mx

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, create_vector
from mxnet import gluon, nd
from tests.python.unittest.common import with_seed

# dimension constants
LARGE_X = 5000000000
MEDIUM_X = 1000000000


def test_slice():
    a = nd.ones(LARGE_X)
    res = nd.slice(a, begin=(LARGE_X - MEDIUM_X), end=LARGE_X)
    assert a[0] == 1
    assert res.shape[0] == MEDIUM_X


def test_ndarray_zeros():
    a = nd.zeros(shape=LARGE_X)
    assert a[-1] == 0
    assert a.shape == (LARGE_X,)
    assert a.size == LARGE_X


def test_ndarray_ones():
    a = nd.ones(shape=LARGE_X)
    assert a[-1] == 1
    assert nd.sum(a).asnumpy() == LARGE_X


@with_seed()
def test_ndarray_random_uniform():
    a = nd.random.uniform(shape=LARGE_X)
    assert a[-1] != 0


@with_seed()
def test_ndarray_random_randint():
    a = nd.random.randint(100, 10000, shape=LARGE_X)
    assert a.shape == (LARGE_X,)
    # check if randint can generate value greater than 2**32 (large)
    low_large_value = 2**32
    high_large_value = 2**34
    a = nd.random.randint(low_large_value, high_large_value, dtype=np.int64)
    low = mx.nd.array([low_large_value], dtype='int64')
    high = mx.nd.array([high_large_value], dtype='int64')
    assert a > low  and a < high


def test_ndarray_empty():
    a = nd.empty(LARGE_X)
    assert a.shape == (LARGE_X,)


def test_elementwise():
    a = nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X)
    res = a + b
    assert res[-1].asnumpy() == 2
    res = a + 1
    assert res[-1].asnumpy() == 2
    res = nd.sqrt(a + 8)
    assert res[-1].asnumpy() == 3


def test_reduce():
    a = nd.ones(shape=(LARGE_X, 1))
    assert nd.sum(a).asnumpy() == a.shape[0] * a.shape[1]


def test_clip():
    a = create_vector(LARGE_X)
    res = nd.clip(a, a_min=100, a_max=1000)
    assert np.sum(res[-1].asnumpy() == 1000) == 1


def test_argmin():
    a = create_vector(LARGE_X, dtype=np.float32)
    assert a[0] == 0
    idx = mx.nd.argmin(a, axis=0)
    assert idx[0] == 0
    assert idx.shape[0] == 1


def test_take():
    a = nd.ones(shape=LARGE_X)
    idx = nd.arange(LARGE_X - 1000, LARGE_X)
    res = nd.take(a, idx)
    assert np.sum(res.asnumpy() == 1) == res.shape[0]


def test_slice_assign():
    a = nd.ones(shape=LARGE_X)
    a[LARGE_X-1:LARGE_X] = 1000
    assert np.sum(a[-1].asnumpy() == 1000) == 1


def test_expand_dims():
    a = nd.ones(shape=LARGE_X)
    res = nd.expand_dims(a, axis=0)
    assert res[0][0] == 1
    assert res.shape == (1, a.shape[0])


def test_squeeze():
    a = nd.ones(shape=LARGE_X)
    data = nd.expand_dims(a, axis=0)
    res = nd.squeeze(data)
    assert a[0] == res[0]
    assert res.shape == a.shape


def test_broadcast_div():
    a = nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X) * 2
    res = a / b
    assert np.sum(res.asnumpy() == 0.5) == a.shape[0]


def test_Dense(ctx=mx.cpu(0)):
    data = mx.nd.ones(shape=LARGE_X)
    linear = gluon.nn.Dense(2)
    linear.initialize(ctx=ctx)
    res = linear(data)
    res.wait_to_read()
    assert res.shape == (LARGE_X, 2)


def test_argsort():
    b = create_vector(size=LARGE_X)
    s = nd.argsort(b, axis=0, is_ascend=False, dtype=np.int64)
    assert (s[0].asnumpy() == (LARGE_X - 1)).all()


def test_sort():
    b = create_vector(size=LARGE_X)
    s = nd.sort(b, axis=0, is_ascend=False)
    assert np.sum(s[-1].asnumpy() == 0).all()
    s = nd.sort(b, is_ascend=True)
    assert np.sum(s[0].asnumpy() == 0).all()


def test_topk():
    b = create_vector(size=LARGE_X)
    ind = nd.topk(b, k=10, axis=0, dtype=np.int64)
    assert np.sum(ind.asnumpy() == (LARGE_X - 1)) == 1
    ind, val = mx.nd.topk(b, k=3, axis=0, dtype=np.int64, ret_typ="both", is_ascend=False)
    assert np.all(ind == val)
    val = nd.topk(b, k=1, axis=0, dtype=np.int64, ret_typ="value")
    assert val.sum() == (LARGE_X - 1)


def test_shape():
    b = create_vector(size=LARGE_X)
    #explicit wait_to_read()
    assert b[0] == 0
    assert b.shape[0] == LARGE_X


def test_size():
    b = create_vector(size=LARGE_X)
    #explicit wait_to_read()
    assert b[0] == 0
    assert b.size == LARGE_X


def test_copy():
    a = nd.ones(LARGE_X)
    b = a.copy()
    assert a[0] == b[0]
    assert b.shape == a.shape
    assert b.size == LARGE_X


def test_copy_to():
    a = create_vector(size=LARGE_X)
    # keeping dtype same as input uses parallel copy which is much faster
    b = nd.zeros(LARGE_X, dtype=np.int64)
    c = a.copyto(b)
    assert c is b
    assert b[-1] == LARGE_X-1
    assert b[0] == 0


def test_zeros_like():
    a = nd.ones(LARGE_X)
    b = nd.zeros_like(a)
    assert b[-1] == 0
    assert b.shape == a.shape


def test_ones_like():
    a = nd.zeros(LARGE_X)
    b = nd.ones_like(a)
    assert b[-1] == 1
    assert b.shape == a.shape


def test_concat():
    a = nd.ones(LARGE_X)
    b = nd.zeros(LARGE_X)
    c = nd.concat(a,b, dim=0)
    assert c[0][0] == 1
    assert c[-1][-1] == 0
    assert c.shape[0] == (2 * LARGE_X)


def test_sum():
    a = nd.ones(LARGE_X)
    b = nd.sum(a, axis=0)
    assert b[0] == LARGE_X


def test_prod():
    a = nd.ones(LARGE_X)
    b = nd.prod(a, axis=0)
    assert b[0] == 1


def test_min():
    a = create_vector(size=LARGE_X)
    b = nd.min(a, axis=0)
    assert b[0] == 0
    assert b[-1] == 0


def test_max():
    a = create_vector(size=LARGE_X)
    b = nd.max(a, axis=0)
    assert b[0] == (LARGE_X - 1)


def test_argmax():
    a = nd.ones(LARGE_X)
    b = nd.zeros(LARGE_X)
    c = nd.concat(a, b, dim=0)
    d = nd.argmax(c, axis=0)
    assert c.shape[0] == (2 * LARGE_X)
    assert d == 0


def np_softmax(x, axis=-1, temperature=1.0):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x/temperature)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


def test_iadd():
    a = nd.ones(LARGE_X)
    b = nd.ones(LARGE_X)
    c = b
    c += a
    assert c.shape == a.shape
    assert c[-1] == 2


def test_isub():
    a = nd.full(LARGE_X, 3)
    b = nd.ones(LARGE_X)
    c = a
    c -= b
    assert c.shape == a.shape
    assert c[-1] == 2


def test_imul():
    a = nd.full(LARGE_X, 3)
    b = nd.ones(LARGE_X)
    c = b
    c *= a
    assert c.shape == a.shape
    assert c[-1] == 3


def test_idiv():
    a = nd.full(LARGE_X, 4)
    b = nd.full(LARGE_X, 2)
    c = a
    c /= b
    assert c.shape == a.shape
    assert c[-1] == 2


def test_imod():
    a = nd.full(LARGE_X, 3)
    b = nd.full(LARGE_X, 2)
    c = a
    c %= b
    assert c.shape == a.shape
    assert c[0][-1] == 1


def test_eq():
    a = nd.full(LARGE_X, 3)
    b = nd.full(LARGE_X, 3)
    c = (a == b)
    assert np.sum(c[0].asnumpy() == 1).all()


def test_neq():
    a = nd.full(LARGE_X, 2)
    b = nd.full(LARGE_X, 3)
    c = (a != b)
    assert np.sum(c[0].asnumpy() == 1).all()


def test_lt():
    a = nd.full(LARGE_X, 2)
    b = nd.full(LARGE_X, 3)
    d = (a <= b)
    assert np.sum(d[0].asnumpy() == 1).all()


def test_lte():
    a = nd.full(LARGE_X, 2)
    b = nd.full(LARGE_X, 3)
    c = nd.full(LARGE_X, 2)
    d = (a <= b)
    assert np.sum(d[0].asnumpy() == 1).all()
    d = (a <= c)
    assert np.sum(d[0].asnumpy() == 1).all()


def test_gt():
    a = nd.full(LARGE_X, 3)
    b = nd.full(LARGE_X, 2)
    d = (a > b)
    assert np.sum(d[0].asnumpy() == 1).all()


def test_gte():
    a = nd.full(LARGE_X, 3)
    b = nd.full(LARGE_X, 2)
    c = nd.full(LARGE_X, 3)
    d = (a >= b)
    assert np.sum(d[0].asnumpy() == 1).all()
    d = (a >= c)
    assert np.sum(d[0].asnumpy() == 1).all()


def test_slice_like():
    a = create_vector(size=LARGE_X)
    b = nd.ones(LARGE_X//2)
    c = nd.slice_like(a, b)
    assert c.shape == b.shape
    assert c[0] == 0
    assert c[-1] == (LARGE_X//2-1)


def test_slice_axis():
    a = create_vector(size=LARGE_X)
    c = nd.slice_axis(a, axis=0, begin=0, end=LARGE_X//2)
    assert c.shape[0] == a.shape[0]//2
    assert c[-1][0] == (LARGE_X//2-1)


def test_full():
    a = nd.full(LARGE_X, 3)
    assert a.shape[0] == LARGE_X
    assert a[LARGE_X//2] == 3
    assert a[-1] == 3


def test_one_hot():
    a = nd.zeros(10)
    a[0] = 1
    a[-1] = 1
    b = nd.one_hot(a, LARGE_X)
    assert b[0][1] == 1
    assert b[-1][1] == 1


if __name__ == '__main__':
    import nose
    nose.runmodule()
