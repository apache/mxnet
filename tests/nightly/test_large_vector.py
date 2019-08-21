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

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d
from mxnet import gluon, nd
from tests.python.unittest.common import with_seed

# dimension constants
LARGE_X = 5000000000
MEDIUM_X = 1000000000
LARGE_Y = 100000
SMALL_Y = 1


def test_slice():
    a = nd.ones(LARGE_X)
    res = nd.slice(a, begin=(LARGE_X - MEDIUM_X), end=LARGE_X)
    assert res.shape[0] == MEDIUM_X


def test_gluon_embedding():
    m = gluon.nn.Embedding(1, LARGE_Y)
    m.initialize()
    a = nd.zeros((LARGE_Y, 1))
    b = m(a)
    assert b.shape == (LARGE_Y, 1, LARGE_Y)
    assert b.asnumpy().size == LARGE_X*2


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
    assert a.__gt__(low) and a.__lt__(high)


def test_ndarray_empty():
    a = nd.empty(LARGE_X)
    assert a.shape == (LARGE_X,)


def test_elementwise():
    a = nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X)
    res = a + b
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
    res = a + 1
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
    res = nd.sqrt(a + 3)
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]


def test_reduce():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    assert nd.sum(a).asnumpy() == a.shape[0] * a.shape[1]


def test_FullyConnected():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(SMALL_Y, SMALL_Y))
    res = nd.FullyConnected(a, b, num_hidden=b.shape[1], no_bias=True)
    assert np.sum(res[-1].asnumpy() == SMALL_Y) == b.shape[1]


def test_broadcast():
    a = nd.ones(shape=(LARGE_X, SMALL_Y*2))
    b = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
    res = nd.broadcast_to(b, shape=(b.shape[0], SMALL_Y*2))
    assert np.sum(res[-1].asnumpy() == LARGE_X) == res.shape[1]
    res = mx.nd.broadcast_like(b, a)
    assert np.sum(res[-1].asnumpy() == LARGE_X) == res.shape[1]


def test_clip():
    a = nd.arange(0, LARGE_X)
    res = nd.clip(a, a_min=100, a_max=1000)
    assert np.sum(res[-1].asnumpy() == 1000) == 101


def test_argmin():
    a = nd.arange(0, LARGE_X)
    idx = mx.nd.argmin(a, axis=0)
    assert idx.shape[0] == SMALL_Y


def test_tile():
    a = nd.arange(0, LARGE_X)
    b = nd.tile(a, reps=(1,2))
    assert b[0][LARGE_X] == b[0][0]
    assert b[0][LARGE_X-1] == b[0][-1]


def test_take():
    a = nd.ones(shape=LARGE_X)
    idx = nd.arange(LARGE_X - 1000, LARGE_X)
    res = nd.take(a, idx)
    assert np.sum(res.asnumpy() == 1) == res.shape[0]


def test_slice():
    a = nd.ones(shape=(2, LARGE_X))
    res = nd.slice(a, begin=(1, LARGE_X-1000000000), end=(2, LARGE_X))
    assert np.sum(res[-1].asnumpy() == 1) == res.shape[1]


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


def test_pick():
    a = mx.nd.ones(shape=(LARGE_X, 2))
    b = mx.nd.ones(shape=LARGE_X)
    res = mx.nd.pick(a, b)
    assert res.shape == b.shape


def test_depthtospace():
    def numpy_depth_to_space(x, blocksize):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
        return y

    shape_inp = (LARGE_X, 4, 1, 1)
    data = rand_ndarray(shape_inp, 'default')
    data_np = data.asnumpy()
    expected = numpy_depth_to_space(data_np, 2)
    output = mx.nd.depth_to_space(data, 2)
    assert_almost_equal(output.asnumpy(), expected, atol=1e-3, rtol=1e-3)


def test_spacetodepth():
    def numpy_space_to_depth(x, blocksize):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        tmp = np.reshape(x, [b, c, h // blocksize, blocksize, w // blocksize, blocksize])
        tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
        y = np.reshape(tmp, [b, c * (blocksize**2), h // blocksize, w // blocksize])
        return y

    shape_inp = (LARGE_X, 1, 2, 2)
    data = rand_ndarray(shape_inp, 'default')
    data_np = data.asnumpy()
    expected = numpy_space_to_depth(data_np, 2)
    output = mx.nd.space_to_depth(data, 2)
    assert_almost_equal(output.asnumpy(), expected, atol=1e-3, rtol=1e-3)

@with_seed()
def test_diag():
    a_np = np.random.random((LARGE_X, 2)).astype(np.float32)
    a = mx.nd.array(a_np)

    # k == 0
    r = mx.nd.diag(a)
    assert_almost_equal(r.asnumpy(), np.diag(a_np))

    # k == 1
    k = 1
    r = mx.nd.diag(a, k=k)
    assert_almost_equal(r.asnumpy(), np.diag(a_np, k=k))

    # k == -1
    k = -1
    r = mx.nd.diag(a, k=k)
    assert_almost_equal(r.asnumpy(), np.diag(a_np, k=k))


@with_seed()
def test_ravel_multi_index():
    x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, SMALL_Y, 4)
    x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, SMALL_Y, 3)
    x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, SMALL_Y, 2)
    indices_2d = [[x1, x2, x3], [y1, y2, y3]]
    idx = mx.nd.ravel_multi_index(mx.nd.array(indices_2d, dtype=np.int64), shape=(LARGE_X, 5))
    idx_numpy = np.ravel_multi_index(indices_2d, (LARGE_X, 5))
    assert np.sum(1 for i in range(idx.size) if idx[i] == idx_numpy[i]) == 3


@with_seed()
def test_unravel_index():
    x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, SMALL_Y, 4)
    x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, SMALL_Y, 3)
    x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, SMALL_Y, 2)
    original_2d_indices = [[x1, x2, x3], [y1, y2, y3]]
    idx_numpy = np.ravel_multi_index(original_2d_indices, (LARGE_X, 5))
    indices_2d = mx.nd.unravel_index(mx.nd.array(idx_numpy, dtype=np.int64), shape=(LARGE_X, 5))
    assert (indices_2d.asnumpy() == np.array(original_2d_indices)).all()


def create_large_vector(size, dtype=np.int64):
    a = nd.arange(0, size, dtype=dtype)
    # Implicitly calling nd.waitall()
    assert a[0] == 0
    return a


def test_transpose():
    b = nd.arange(0, LARGE_X, dtype=np.int64).reshape(1, LARGE_X)
    t = b.T
    assert t.shape == (LARGE_X, 1)
    assert t[-1, 0].asnumpy() == (LARGE_X - 1)


def test_swapaxes():
    b = nd.arange(0, LARGE_X, dtype=np.int64).reshape(LARGE_X, 1)
    t = nd.swapaxes(b, dim1=0, dim2=1)
    assert t.shape == (1, LARGE_X)
    assert t[0, -1].asnumpy() == (LARGE_X - 1)


def test_flip():
    b = nd.arange(0, LARGE_X, dtype=np.int64).reshape(1, LARGE_X)
    t = nd.flip(b, axis=0)
    assert t.shape == (LARGE_X, 1)
    assert t[-1, :].asnumpy() == 0


def test_softmax():
    input_data = mx.nd.ones(2, LARGE_X)
    true_output = np.full(LARGE_X, 0.5)
    output = nd.softmax(input_data, axis=0)
    assert_almost_equal(output.asnumpy(), true_output, rtol=1e-5, atol=1e-5)


def test_argsort():
    b = create_large_vector(size=LARGE_X)
    s = nd.argsort(b, axis=0, is_ascend=False, dtype=np.int64)
    mx.nd.waitall()
    assert (s[0].asnumpy() == (LARGE_X - 1)).all()


def test_sort():
    b = create_large_vector(size=LARGE_X)
    s = nd.sort(b, axis=0, is_ascend=False)
    assert np.sum(s[-1][SMALL_Y//2:SMALL_Y].asnumpy() == 0).all()
    s = nd.sort(b, is_ascend=True)
    assert np.sum(s[0].asnumpy() == 0).all()


def test_topk():
    b = create_large_vector(size=LARGE_X)
    k = nd.topk(b, k=10, axis=0, dtype=np.int64)
    assert np.sum(k.asnumpy() == (LARGE_X - 1)) == SMALL_Y
    ind, val = mx.nd.topk(b, k=3, axis=0, dtype=np.int64, ret_typ="both", is_ascend=False)
    assert np.all(ind == val)
    l = nd.topk(b, k=1, axis=0, dtype=np.int64, ret_typ="value")
    assert l.sum() == (LARGE_X - 1)


if __name__ == '__main__':
    import nose
    nose.runmodule()
