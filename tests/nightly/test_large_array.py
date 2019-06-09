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
MEDIUM_X = 10000
LARGE_X = 100000000
LARGE_Y = 50000000
SMALL_Y = 50
LARGE_SIZE = LARGE_X * SMALL_Y


def test_gluon_embedding():
    m = gluon.nn.Embedding(SMALL_Y, MEDIUM_X)
    m.initialize()
    a = nd.zeros((MEDIUM_X, SMALL_Y))
    b = m(a)
    assert b.shape == (MEDIUM_X, SMALL_Y, MEDIUM_X)
    assert b.asnumpy().size == LARGE_SIZE


def test_ndarray_zeros():
    a = nd.zeros(shape=(LARGE_X, SMALL_Y))
    assert a[-1][0] == 0
    assert a.shape == (LARGE_X, SMALL_Y)
    assert a.size == LARGE_SIZE


def test_ndarray_ones():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    assert a[-1][0] == 1
    assert nd.sum(a).asnumpy() == LARGE_SIZE


def test_ndarray_convert():
    a = nd.zeros(shape=(LARGE_X, SMALL_Y))
    b = a.astype(np.int32)
    b.wait_to_read()
    assert b.dtype == np.int32
    b = a.tostype('row_sparse')
    b.wait_to_read()
    assert isinstance(b, mx.nd.sparse.RowSparseNDArray)


@with_seed()
def test_ndarray_random_uniform():
    a = nd.random.uniform(shape=(LARGE_X, SMALL_Y))
    assert a[-1][0] != 0


@with_seed()
def test_ndarray_random_randint():
    a = nd.random.randint(100, 10000, shape=(LARGE_X, SMALL_Y))
    assert a.shape == (LARGE_X, SMALL_Y)
    # check if randint can generate value greater than 2**32 (large)
    low_large_value = 2**32
    high_large_value = 2**34
    a = nd.random.randint(low_large_value, high_large_value, dtype=np.int64)
    low = mx.nd.array([low_large_value], dtype='int64')
    high = mx.nd.array([high_large_value], dtype='int64')
    assert a.__gt__(low) and a.__lt__(high)


def test_ndarray_empty():
    a = nd.empty((LARGE_X, SMALL_Y))
    assert a.shape == (LARGE_X, SMALL_Y)


def test_elementwise():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(LARGE_X, SMALL_Y))
    res = a + b
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
    res = a + 1
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
    res = nd.sqrt(a + 3)
    assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]


def test_reduce():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    assert nd.sum(a).asnumpy() == a.shape[0] * a.shape[1]


def test_dot():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(SMALL_Y, SMALL_Y))
    res = nd.dot(a, b)
    assert np.sum(res[-1].asnumpy() == SMALL_Y) == b.shape[1]


def test_FullyConnected():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(SMALL_Y, SMALL_Y))
    res = nd.FullyConnected(a, b, num_hidden=b.shape[1], no_bias=True)
    assert np.sum(res[-1].asnumpy() == SMALL_Y) == b.shape[1]


def test_broadcast():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
    res = nd.broadcast_to(b, shape=(b.shape[0], SMALL_Y))
    assert np.sum(res[-1].asnumpy() == LARGE_X) == res.shape[1]
    res = mx.nd.broadcast_like(b, a)
    assert np.sum(res[-1].asnumpy() == LARGE_X) == a.shape[1]


def test_clip():
    a = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
    res = nd.clip(a, a_min=100, a_max=1000)
    assert np.sum(res[-1].asnumpy() == 1000) == a.shape[1]


def test_split():
    a = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
    outs = nd.split(a, num_outputs=SMALL_Y, axis=1)
    result = sum(1 for i, v in enumerate(outs) if i == v[0].asnumpy())
    assert result == a.shape[1]


def test_argmin():
    a = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
    idx = mx.nd.argmin(a, axis=0)
    assert idx.shape[0] == SMALL_Y


def test_tile():
    a = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
    b = nd.tile(a, reps=(1, SMALL_Y))
    assert np.sum(b[-1].asnumpy() == LARGE_X) == b.shape[1]


def test_take():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    idx = nd.arange(LARGE_X - 1000, LARGE_X)
    res = nd.take(a, idx)
    assert np.sum(res[-1].asnumpy() == 1) == res.shape[1]


def test_slice():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    res = nd.slice(a, begin=(LARGE_X-1000, 1), end=(LARGE_X, SMALL_Y))
    assert np.sum(res[-1].asnumpy() == 1) == res.shape[1]


def test_slice_assign():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    a[LARGE_X-1:LARGE_X] = 1000
    assert np.sum(a[-1].asnumpy() == 1000) == a.shape[1]


def test_expand_dims():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    res = nd.expand_dims(a, axis=1)
    assert res.shape == (a.shape[0], 1, a.shape[1])


def test_squeeze():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    data = nd.expand_dims(a, axis=1)
    res = nd.squeeze(data)
    assert res.shape == a.shape


def test_broadcast_div():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(LARGE_X, 1)) * 2
    res = a / b
    assert np.sum(res[-1].asnumpy() == 0.5) == a.shape[1]


def test_Dense(ctx=mx.cpu(0)):
    data = mx.nd.ones(shape=(50*1000*1000, 100))
    linear = gluon.nn.Dense(100)
    linear.initialize(ctx=ctx)
    res = linear(data)
    res.wait_to_read()
    assert res.shape == (50000000, 100)


def test_where():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
    res = nd.where(b > 100, a, b)
    assert np.sum(res[-1].asnumpy() == 1) == b.shape[1]
    csr_cond = nd.sparse.cast_storage(b < 10, 'csr')
    res = nd.sparse.where(csr_cond, a, b)
    assert np.sum(res[0].asnumpy() == 1) == 10


def test_pick():
    a = mx.nd.ones(shape=(256 * 35, 1024 * 1024))
    b = mx.nd.ones(shape=(256 * 35, ))
    res = mx.nd.pick(a, b)
    assert res.shape == b.shape


def test_depthtospace():
    def numpy_depth_to_space(x, blocksize):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
        return y

    shape_inp = (LARGE_X, 8, 4, 2)
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

    shape_inp = (LARGE_X, 2, 8, 4)
    data = rand_ndarray(shape_inp, 'default')
    data_np = data.asnumpy()
    expected = numpy_space_to_depth(data_np, 2)
    output = mx.nd.space_to_depth(data, 2)
    assert_almost_equal(output.asnumpy(), expected, atol=1e-3, rtol=1e-3)

@with_seed()
def test_diag():
    a_np = np.random.random((LARGE_X, SMALL_Y)).astype(np.float32)
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

    # random k
    k = np.random.randint(-min(LARGE_X, SMALL_Y) + 1, min(LARGE_X, SMALL_Y))
    r = mx.nd.diag(a, k=k)
    assert_almost_equal(r.asnumpy(), np.diag(a_np, k=k))


@with_seed()
def test_ravel_multi_index():
    x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, 10, SMALL_Y)
    x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, 9, SMALL_Y)
    x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, 8, SMALL_Y)
    indices_2d = [[x1, x2, x3], [y1, y2, y3]]
    idx = mx.nd.ravel_multi_index(mx.nd.array(indices_2d, dtype=np.int64), shape=(LARGE_X, SMALL_Y))
    idx_numpy = np.ravel_multi_index(indices_2d, (LARGE_X, SMALL_Y))
    assert np.sum(1 for i in range(idx.size) if idx[i] == idx_numpy[i]) == 3


@with_seed()
def test_unravel_index():
    x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, 10, SMALL_Y)
    x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, 9, SMALL_Y)
    x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, 8, SMALL_Y)
    original_2d_indices = [[x1, x2, x3], [y1, y2, y3]]
    idx_numpy = np.ravel_multi_index(original_2d_indices, (LARGE_X, SMALL_Y))
    indices_2d = mx.nd.unravel_index(mx.nd.array(idx_numpy, dtype=np.int64), shape=(LARGE_X, SMALL_Y))
    assert (indices_2d.asnumpy() == np.array(original_2d_indices)).all()


def create_2d_tensor(rows, columns):
    a = np.arange(0, rows).reshape(rows, 1)
    b = np.broadcast_to(a, shape=(a.shape[0], columns))
    return nd.array(b, dtype=np.int64)


def test_transpose():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    t = b.T
    assert t.shape == (SMALL_Y, LARGE_X)
    assert np.sum(t[:, -1].asnumpy() == (LARGE_X - 1)) == b.shape[1]


def test_swapaxes():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    t = nd.swapaxes(b, dim1=0, dim2=1)
    assert t.shape == (SMALL_Y, LARGE_X)
    assert np.sum(t[:, -1].asnumpy() == (LARGE_X - 1)) == b.shape[1]


def test_flip():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    t = nd.flip(b, axis=0)
    assert t.shape == (LARGE_X, SMALL_Y)
    assert np.sum(t[-1, :].asnumpy() == 0) == b.shape[1]


def test_softmax():
    input_data = mx.nd.ones((SMALL_Y, LARGE_X))
    true_output = np.full((SMALL_Y, LARGE_X), (1 / SMALL_Y))
    output = nd.softmax(input_data, axis=0)
    assert_almost_equal(output.asnumpy(), true_output, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    import nose
    nose.runmodule()
