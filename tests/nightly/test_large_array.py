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

import math
import numpy as np
import mxnet as mx

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, default_context, check_symbolic_forward
from mxnet import gluon, nd
from tests.python.unittest.common import with_seed

# dimension constants
MEDIUM_X = 10000
LARGE_X = 100000000
SMALL_X = 100
SMALL_Y = 50
LARGE_SIZE = LARGE_X * SMALL_Y


def create_2d_tensor(rows, columns, dtype=np.int64):
    a = nd.arange(0, rows, dtype=dtype).reshape(rows, 1)
    b = nd.broadcast_to(a, shape=(a.shape[0], columns))
    return nd.array(b, dtype=dtype)


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
    assert a[-1][0].dtype == np.int64


@with_seed()
def test_ndarray_random_exponential():
    scale_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    a = nd.random.exponential(scale=scale_array, shape=(SMALL_X, SMALL_Y))
    assert a[-1][0][0][0] >= 0
    assert a.shape == (MEDIUM_X, SMALL_Y, SMALL_X, SMALL_Y)


@with_seed()
def test_ndarray_random_gamma():
    alpha_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    beta_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    a = nd.random.gamma(alpha=alpha_array, beta=beta_array,
                        shape=(SMALL_X, SMALL_Y))
    assert a[-1][0][0][0] >= 0
    assert a.shape == (MEDIUM_X, SMALL_Y, SMALL_X, SMALL_Y)


@with_seed()
def test_ndarray_random_multinomial():
    # test 1 shape dimension
    probs = nd.random.uniform(shape=(LARGE_X, SMALL_Y))
    a = nd.random.multinomial(probs)
    assert a[-1] >= 0
    assert a.shape == (LARGE_X,)
    # test for NDArray multi-dimension shape
    a = nd.random.multinomial(probs, shape=(SMALL_X, SMALL_Y))
    assert a[-1][0][0] >= 0
    assert a.shape == (LARGE_X, SMALL_X, SMALL_Y)
    # test log_likelihood output shape
    a = nd.random.multinomial(probs, shape=(SMALL_X, SMALL_Y), get_prob=True)
    assert a[-1][0][0] >= 0
    assert a[0].shape == (LARGE_X, SMALL_X, SMALL_Y) and a[0].shape == a[1].shape


@with_seed()
def test_ndarray_random_generalized_negative_binomial():
    alpha_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    mu_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    a = nd.random.generalized_negative_binomial(mu=mu_array, alpha=alpha_array,
                                                shape=(SMALL_X, SMALL_Y))
    assert a[-1][0][0][0] >= 0
    assert a.shape == (MEDIUM_X, SMALL_Y, SMALL_X, SMALL_Y)


@with_seed()
def test_ndarray_random_negative_binomial():
    k_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    p_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    a = nd.random.negative_binomial(k=k_array, p=p_array,
                                    shape=(SMALL_X, SMALL_Y))
    assert a[-1][0][0][0] >= 0
    assert a.shape == (MEDIUM_X, SMALL_Y, SMALL_X, SMALL_Y)


@with_seed()
def test_ndarray_random_normal():
    scale_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    loc_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    a = nd.random.normal(loc=loc_array, scale=scale_array,
                         shape=(SMALL_X, SMALL_Y))
    a.wait_to_read()
    assert a.shape == (MEDIUM_X, SMALL_Y, SMALL_X, SMALL_Y)


@with_seed()
def test_ndarray_random_poisson():
    lambda_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_Y))
    a = nd.random.poisson(lam=lambda_array, shape=(SMALL_X, SMALL_Y))
    assert a[-1][0][0][0] >= 0
    assert a.shape == (MEDIUM_X, SMALL_Y, SMALL_X, SMALL_Y)


@with_seed()
def test_ndarray_random_randn():
    a = nd.random.randn(LARGE_X, SMALL_Y)
    a.wait_to_read()
    assert a.shape == (LARGE_X, SMALL_Y)
    # TODO: Once PR #15772 for randn ndarray dtype for loc,scale param merged
    # Add check for (x,y,m,n) where x,y shape of loc,scale and m,n input shape


@with_seed()
def test_ndarray_random_shuffle():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    a[-1] == 3  # assign 3 to entire last row
    a = nd.random.shuffle(a)
    # slice first column from shuffled array
    # pass LARGE_X values to numpy instead of LARGE_X*SMALL_Y
    # could have assigned to last column (so as to pass SMALL_Y)
    # but shuffle operation is performed along first axis
    unique_a = np.unique(a[:, 0].asnumpy())
    assert len(unique_a) == 2  # only 2 unique values
    assert unique_a[0] == 1  # first unique value is 1
    assert unique_a[1] == 3  # second unique value is 3
    assert a.shape[0] == (LARGE_X, SMALL_Y)


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
        tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h,
                         w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize,
                       w * blocksize])
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
        tmp = np.reshape(x, [b, c, h // blocksize, blocksize, w // blocksize,
                         blocksize])
        tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
        y = np.reshape(tmp, [b, c * (blocksize**2), h // blocksize,
                       w // blocksize])
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
    idx = mx.nd.ravel_multi_index(mx.nd.array(indices_2d, dtype=np.int64),
                                  shape=(LARGE_X, SMALL_Y))
    idx_numpy = np.ravel_multi_index(indices_2d, (LARGE_X, SMALL_Y))
    assert np.sum(1 for i in range(idx.size) if idx[i] == idx_numpy[i]) == 3


@with_seed()
def test_unravel_index():
    x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, 10, SMALL_Y)
    x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, 9, SMALL_Y)
    x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, 8, SMALL_Y)
    original_2d_indices = [[x1, x2, x3], [y1, y2, y3]]
    idx_numpy = np.ravel_multi_index(original_2d_indices, (LARGE_X, SMALL_Y))
    indices_2d = mx.nd.unravel_index(mx.nd.array(idx_numpy, dtype=np.int64),
                                     shape=(LARGE_X, SMALL_Y))
    assert (indices_2d.asnumpy() == np.array(original_2d_indices)).all()


def test_transpose():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    t = b.T
    assert np.sum(t[:, -1].asnumpy() == (LARGE_X - 1)) == b.shape[1]
    assert t.shape == (SMALL_Y, LARGE_X)


def test_swapaxes():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    t = nd.swapaxes(b, dim1=0, dim2=1)
    assert np.sum(t[:, -1].asnumpy() == (LARGE_X - 1)) == b.shape[1]
    assert t.shape == (SMALL_Y, LARGE_X)


def test_flip():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    t = nd.flip(b, axis=0)
    assert np.sum(t[-1, :].asnumpy() == 0) == b.shape[1]
    assert t.shape == (LARGE_X, SMALL_Y)


def test_softmax():
    input_data = mx.nd.ones((SMALL_Y, LARGE_X))
    true_output = np.full((SMALL_Y, LARGE_X), (1 / SMALL_Y))
    output = nd.softmax(input_data, axis=0)
    assert_almost_equal(output.asnumpy(), true_output, rtol=1e-5, atol=1e-5)


def test_argsort():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    s = nd.argsort(b, axis=0, is_ascend=False, dtype=np.int64)
    mx.nd.waitall()
    assert (s[0].asnumpy() == (LARGE_X - 1)).all()


def test_sort():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    s = nd.sort(b, axis=0, is_ascend=False)
    assert np.sum(s[-1][SMALL_Y//2:SMALL_Y].asnumpy() == 0).all()
    s = nd.sort(b, is_ascend=False)
    assert np.sum(s[0].asnumpy() == 0).all()


def test_topk():
    b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
    k = nd.topk(b, k=10, axis=0, dtype=np.int64)
    assert np.sum(k.asnumpy() == (LARGE_X - 1)) == SMALL_Y
    ind, val = mx.nd.topk(b, k=3, axis=0, dtype=np.int64, ret_typ="both",
                          is_ascend=False)
    assert np.all(ind == val)
    b = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    l = nd.topk(b, k=1, axis=-1, dtype=np.int64, ret_typ="value")
    assert l.sum() == np.sum(np.arange(0, SMALL_Y))


def test_exponent_logarithm_operators():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    # exponent
    result = nd.exp(a)
    assert result[0][-1] == 7.389056
    assert result.shape == a.shape

    # exponent minus 1
    result = nd.expm1(a)
    assert result[0][-1] == 6.389056
    assert result.shape == a.shape

    # log2
    result = nd.log2(a)
    assert result[0][-1] == 1
    assert result.shape == a.shape

    # log10
    result = nd.log10(a)
    assert result[0][-1] == 0.30103
    assert result.shape == a.shape

    # log1p
    result = nd.log1p(a)
    assert result[0][-1] == 1.0986123
    assert result.shape == a.shape

    # log
    result = nd.log(a)
    assert result[0][-1] == 0.6931472
    assert result.shape == a.shape


def test_power_operators():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    # sqrt
    result = nd.sqrt(a)
    assert result[0][-1] == 1.4142135
    assert result.shape == a.shape

    # rsqrt
    result = nd.rsqrt(a)
    assert result[0][-1] == 0.70710677
    assert result.shape == a.shape

    # cbrt
    result = nd.cbrt(a)
    assert result[0][-1] == 1.2599211
    assert result.shape == a.shape

    # rcbrt
    result = nd.rcbrt(a)
    assert result[0][-1] == 0.7937005
    assert result.shape == a.shape

    # square
    result = nd.square(a)
    assert result[0][-1] == 4
    assert result.shape == a.shape

    # reciprocal
    result = nd.reciprocal(a)
    assert result[0][-1] == 0.5
    assert result.shape == a.shape


def test_sequence_mask():
    # Sequence Mask input [max_sequence_length, batch_size, other_feature_dims]
    # test with input batch_size = 2
    a = nd.arange(0, LARGE_X * SMALL_Y * 2).reshape(LARGE_X, 2, SMALL_Y)

    # test as identity operator
    b = nd.SequenceMask(a)
    assert b[-1][0][1] == a[-1][0][1]
    assert b.shape == a.shape

    # test with default mask
    b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                        use_sequence_length=True)
    assert b[0][1][-1] == a[0][1][-1]  # first sequence of each batch kept
    assert b[-1][-1][-1] != a[-1][-1][-1]  # rest sequences masked
    assert b[-1][-1][-1] == 0

    # test with mask value
    b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                        use_sequence_length=True, value=-1)
    assert b[-1][-1][-1] == -1


def test_sequence_reverse():
    a = nd.arange(0, LARGE_X * SMALL_Y * 2).reshape(LARGE_X, 2, SMALL_Y)
    # test as reverse operator
    b = nd.SequenceReverse(a)
    assert b[-1][0][0] == a[0][0][0]
    assert b.shape == a.shape

    # test with sequence length
    # 2 rows of batch 1 and 3 rows of batch 2 reversed
    b = nd.SequenceReverse(a, sequence_length=nd.array([2, 3]),
                           use_sequence_length=True)
    assert b[1][0][0] == a[0][0][0]  # check if reversed
    assert b[-1][0][0] == a[-1][0][0]  # check if intact
    assert b.shape == a.shape


def test_sequence_last():
    a = nd.arange(0, LARGE_X * SMALL_Y * 2).reshape(LARGE_X, 2, SMALL_Y)

    # test if returns last sequence
    b = nd.SequenceLast(a)
    assert_almost_equal(b, a[-1])  # only checks for (2,SMALL_Y) tensor
    assert b.shape == (2, SMALL_Y)

    # test with sequence length
    # parameter sequence_length - NDArray with shape (batch_size)
    # (2,3) indicates 2nd sequence from batch 1 and 3rd sequence from batch 2
    b = nd.SequenceLast(a, sequence_length=mx.nd.array([2, 3]),
                        use_sequence_length=True)
    # check if it takes 2nd sequence from the first batch
    assert b[0][-1] == a[1][0][-1]


def test_softmax_cross_entropy():
    # dtype of input data, mxnet cross entropy set explicitly to float64
    # numpy implicitly takes care of double precision
    batch_size = SMALL_Y
    num_labels = LARGE_X
    input_data = mx.nd.ones((batch_size, num_labels), dtype="float64")
    input_label = mx.nd.zeros((batch_size,), dtype="float64")

    true_softmax = np.full((batch_size, num_labels), (1 / num_labels))
    # use 1/batch_size when softmax axis=0
    # here 1/num_labels since softmax_cross_entropy uses default axis
    # by default axis=1
    np_one_hot_label = np.zeros((batch_size, num_labels))
    np_one_hot_label[:, 0] = 1

    true_softmax_cross_entropy = np.sum(-np.log(true_softmax) *
                                        np_one_hot_label)
    mx_softmax_cross_entropy = mx.nd.softmax_cross_entropy(input_data,
                                                           input_label,
                                                           dtype="float64")
    assert_almost_equal(mx_softmax_cross_entropy.asnumpy(),
                        true_softmax_cross_entropy, rtol=1e-3, atol=1e-5)


def test_index_copy():
    x = mx.nd.zeros((LARGE_X, SMALL_Y))
    t = mx.nd.arange(1, SMALL_Y + 1).reshape((1, SMALL_Y))
    index = mx.nd.array([LARGE_X - 1])

    x = mx.nd.contrib.index_copy(x, index, t)
    assert x[-1][-1] == t[0][-1]


def testSoftmaxOutput():
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.ones((LARGE_X, SMALL_Y))
    grad_x = mx.nd.zeros((LARGE_X, SMALL_Y))
    label_nd = mx.nd.ones((LARGE_X))

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0,
                               use_ignore=False)
    ex = sym.bind(ctx=default_context(), args={'x': x_nd, 'label': label_nd},
                  args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0][0].asnumpy()
    expected_softmax_out = (1/SMALL_Y)*mx.nd.ones((SMALL_Y)).asnumpy()
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0][0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((SMALL_Y,))
    expected_grad_out[k] = -1
    assert np.isclose(grad_out - softmax_out, expected_grad_out).all()


# TODO: correctness of prelu (currently flaky)
def test_leaky_relu():
    a = -1*mx.nd.ones((LARGE_X, SMALL_Y))

    def test_leaky():
        res = mx.nd.LeakyReLU(a, act_type="leaky", slope=0.3)
        assert res[-1][-1].asnumpy() == 0.3*a[-1][-1].asnumpy()

    def test_elu():
        res = mx.nd.LeakyReLU(a, act_type="elu", slope=0.3)
        assert res[-1][-1].asnumpy() == 0.3*(np.exp(a[-1][-1].asnumpy())-1)

    def test_selu():
        lam = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        res = mx.nd.LeakyReLU(a, act_type="selu")
        assert res[-1][-1].asnumpy() == (lam * alpha * (np.exp(a[-1][-1].asnumpy())-1))

    def test_rrelu():
        lower = 0.125
        upper = 0.333999991
        res = mx.nd.LeakyReLU(a, act_type="rrelu")
        assert res[-1][-1].asnumpy() == (lower + upper) / 2 * a[-1][-1].asnumpy()

    test_leaky()
    test_elu()
    test_selu()
    test_rrelu()


def test_pooling():
    a = mx.nd.ones((MEDIUM_X, MEDIUM_X, SMALL_Y, SMALL_Y))

    def test_avg_pooling():
        res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='avg')
        assert res[-1][-1][-1][-1] == 1.0000001
        assert res.shape == SMALL_Y - 5 + 1

    def test_max_pooling():
        res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='max')
        assert res[-1][-1][-1][-1] == 1.
        assert res.shape == SMALL_Y - 5 + 1

    def test_sum_pooling():
        res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='sum')
        assert res[-1][-1][-1][-1] == 25
        assert res.shape == SMALL_Y - 5 + 1

    def test_lp_pooling():
        res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='lp', p_value=2)
        assert res[-1][-1][-1][-1] == 5.
        assert res.shape == SMALL_Y - 5 + 1

        res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='lp', p_value=1)
        assert res[-1][-1][-1][-1] == 25.
        assert res.shape == SMALL_Y - 5 + 1

    test_avg_pooling()
    test_max_pooling()
    test_sum_pooling()
    test_lp_pooling()


def test_layer_norm():
    dtype = np.float32
    forward_check_eps = 1E-3
    axis = 1
    eps = 1E-5
    in_shape = (LARGE_X, SMALL_Y)
    ctx = mx.cpu()

    def npy_layer_norm(data, gamma, beta, axis=1, eps=1E-5):
        if axis < 0:
            axis += data.ndim
        broadcast_shape = [1 for _ in range(data.ndim)]
        broadcast_shape[axis] = data.shape[axis]
        mean = data.mean(axis=axis, keepdims=True).astype(dtype)
        var = data.var(axis=axis, keepdims=True).astype(dtype)
        std = np.sqrt(var + dtype(eps)).astype(dtype)
        out = np.reshape(gamma, broadcast_shape) * (data - mean) / std + \
              np.reshape(beta, broadcast_shape)
        return out
    data = np.random.normal(0, 1, in_shape).astype(dtype)
    gamma = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
    beta = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
    data_s = mx.symbol.Variable('data')
    gamma_s = mx.symbol.Variable('gamma')
    beta_s = mx.symbol.Variable('beta')
    out_s = mx.symbol.LayerNorm(data=data_s, gamma=gamma_s, beta=beta_s,
                                axis=axis, eps=eps)
    exe = out_s.simple_bind(ctx, data=in_shape)
    exe.arg_dict['data'][:] = data
    exe.arg_dict['gamma'][:] = gamma
    exe.arg_dict['beta'][:] = beta
    out_nd = exe.forward()[0]
    out = npy_layer_norm(data, gamma, beta, axis, eps)
    assert_almost_equal(out, out_nd.asnumpy(), forward_check_eps,
                        forward_check_eps)


# TODO: correctness of dropout
# currently only test for dropout to work
# since testing for correctness involves flakiness issue #14288
def test_dropout():
    shape = (LARGE_X, SMALL_Y)
    x = mx.sym.var('data')
    y = mx.sym.Dropout(x, p=1, cudnn_off=True)
    exe = y.simple_bind(ctx=default_context(), data=shape)
    exe.arg_arrays[0][:] = 1
    out = exe.forward(is_train=True)
    out[0].wait_to_read()


def test_activation():
    a = mx.nd.ones((LARGE_X, SMALL_Y))
    test_x = -2
    a[-1, -1] = test_x

    # Hyperbolic tangent (tanh)
    # y = (exp(x)-exp(-x))/(exp(x)+exp(-x))
    a = mx.nd.Activation(a, act_type="tanh")
    tanh_x = (np.exp(test_x)-np.exp(-test_x))/(np.exp(test_x)+np.exp(-test_x))
    assert a[-1][-1] == tanh_x

    # Recitified Linear Unit (relu)
    # y = max(x,0)
    a = mx.nd.Activation(a, act_type="relu")
    assert a[-1][-1] == 0

    # Sigmoid
    # y = x/(1+abs(x))
    a = mx.nd.Activation(a, act_type="sigmoid")
    sigmoid_x = 1/(1+math.exp(-test_x))
    assert a[-1][-1] == sigmoid_x

    # Soft Sign
    # y = 1/(1+exp(-x))
    a = mx.nd.Activation(a, act_type="softsign")
    softsign_x = test_x/(1+abs(test_x))
    assert a[-1][-1] == softsign_x


# TODO: correctness of batchnorm
# in future, we could test if mean, var of output
# matches target output's mean, var
def test_batchnorm():
    shape = (LARGE_X, SMALL_Y)
    axis = 1  # default

    nch = shape[axis]
    data = mx.nd.ones(shape=shape)
    bn_gamma = mx.nd.random.uniform(shape=(nch,))
    bn_beta = mx.nd.random.uniform(shape=(nch,))
    bn_running_mean = mx.nd.zeros(nch)
    bn_running_var = mx.nd.ones(nch)

    output = mx.nd.BatchNorm(data, bn_gamma, bn_beta,
                             bn_running_mean, bn_running_var)
    output.wait_to_read()


def test_add():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__add__(a)
    assert c[0][-1] == 2
    assert c.shape == a.shape


def test_sub():
    a = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__sub__(a)
    assert c[0][-1] == -2
    assert c.shape == a.shape


def test_rsub():
    a = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__rsub__(a)
    assert c[0][-1] == 2
    assert c.shape == a.shape


def test_neg():
    a = nd.ones(shape=(LARGE_X, SMALL_Y))
    c = a
    c = c.__neg__()
    assert c[0][-1] == -1
    assert c.shape == a.shape


def test_mul():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__mul__(a)
    assert c[0][-1] == 6
    assert c.shape == a.shape


def test_div():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__div__(a)
    assert c[0][-1] == 3/2
    assert c.shape == a.shape


def test_rdiv():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__rdiv__(a)
    assert c[0][-1] == 2/3
    assert c.shape == a.shape


def test_mod():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__mod__(a)
    assert c[0][-1] == 1
    assert c.shape == a.shape


def test_rmod():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__rmod__(a)
    assert c[0][-1] == 2
    assert c.shape == a.shape


def test_imod():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__imod__(a)
    assert c[0][-1] == 1
    assert c.shape == a.shape


def test_pow():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__pow__(a)
    assert c[0][-1] == 9
    assert c.shape == a.shape


def test_rpow():
    a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
    b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
    c = b
    c = c.__rpow__(a)
    assert c[0][-1] == 8
    assert c.shape == a.shape


def test_shape():
    b = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    mx.nd.waitall()
    assert b.shape == (SMALL_Y, LARGE_X)


def test_size():
    b = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    mx.nd.waitall()
    assert b.size == LARGE_SIZE


def test_copy():
    a = nd.ones((SMALL_Y, LARGE_X))
    b = a.copy()
    nd.waitall()
    assert b.shape == a.shape
    assert b.size == LARGE_SIZE


def test_copy_to():
    a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    b = nd.array(np.zeros((SMALL_Y, LARGE_X)))
    c = a.copyto(b)
    assert c is b
    print(b)
    assert b[0][-1] == LARGE_X-1


def test_zeros_like():
    a = nd.array(np.ones((SMALL_Y, LARGE_X)))
    b = nd.zeros_like(a)
    assert b[-1][-1] == 0
    assert b.shape == a.shape


def test_ones_like():
    a = nd.array(np.zeros((SMALL_Y, LARGE_X)))
    b = nd.ones_like(a)
    assert b[-1][-1] == 1
    assert b.shape == a.shape


def test_reshape_like():
    a = nd.array(np.zeros((SMALL_Y, LARGE_X)))
    b = nd.array(np.zeros((SMALL_Y//2, LARGE_X*2)))
    c = nd.reshape_like(a, b)
    assert c.shape == (SMALL_Y//2, LARGE_X*2)


def test_flatten():
    a = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y).reshape((LARGE_X//2, 2, SMALL_Y))
    b = nd.flatten(a)
    assert b[-1][-1] == (LARGE_X-1)
    assert b[-1][0] == (LARGE_X-2)
    assert b.shape == (LARGE_X//2, SMALL_Y*2)


def test_expand_dims():
    a = nd.array(np.ones((SMALL_Y, LARGE_X)))
    b = nd.expand_dims(a, axis=1)
    nd.waitall()
    assert b.shape == (SMALL_Y, 1, LARGE_X)


def test_concat():
    a = nd.array(np.ones((SMALL_Y, LARGE_X)))
    b = nd.array(np.zeros((SMALL_Y, LARGE_X)))
    c = nd.concat(a,b, dim=0)
    assert c.shape == (b.shape[0]*2, LARGE_X)


def test_stack():
    a = nd.array(np.ones((SMALL_Y, LARGE_X)))
    b = nd.array(np.zeros((SMALL_Y, LARGE_X)))
    c = nd.stack(a,b, axis=1)
    assert c.shape == (b.shape[0], 2, LARGE_X)


def test_broadcast_axes():
    a = create_2d_tensor(rows=1, columns=LARGE_X)
    b = nd.broadcast_axis(a, axis=[0], size=2)
    assert b.shape == (a.shape[0]*2, a.shape[1])


def test_sum():
    a = nd.array(np.ones((SMALL_Y, LARGE_X)))
    b = nd.sum(a, axis=1)
    assert b.shape[0] == SMALL_Y


def test_prod():
    a = nd.array(np.ones((SMALL_Y, LARGE_X)))
    b = nd.prod(a, axis=1)
    assert b.shape[0] == SMALL_Y


def test_mean():
    a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    b = nd.mean(a, axis=0)
    assert b[0] == (SMALL_Y/2-1)


def test_min():
    a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    b = nd.min(a, axis=0)
    assert b[0] == 0
    assert b[-1] == 0


def test_max():
    a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    b = nd.max(a, axis=0)
    assert b[0] == (SMALL_Y-1)
    assert b[-1] == (SMALL_Y-1)


def test_norm():
    a = np.array(np.full((1, LARGE_X), 3))
    b = np.array(np.full((1, LARGE_X), 4))
    c = nd.array(np.concatenate((a,b), axis=0))
    d = nd.norm(c, ord=2, axis=0)
    e = nd.norm(c, ord=1, axis=0)
    assert d.shape[0] == LARGE_X
    assert e.shape[0] == LARGE_X
    assert d[-1] == 5
    assert e[-1] == 7


def test_argmax():
    a = np.ones((SMALL_Y, LARGE_X))
    b = np.zeros((SMALL_Y, LARGE_X))
    c = nd.array(np.concatenate((a,b), axis=0))
    d = nd.argmax(c, axis=0)
    assert d.shape[0] == LARGE_X
    assert d[-1] == d[0] == 0


def test_relu():
    def frelu(x):
        return np.maximum(x, 0.0)
    def frelu_grad(x):
        return 1.0 * (x > 0.0)
    shape = (SMALL_Y, LARGE_X)
    x = mx.symbol.Variable("x")
    y = mx.sym.relu(x)
    xa = np.random.uniform(low=-1.0,high=1.0,size=shape)
    eps = 1e-4
    xa[abs(xa) < eps] = 1.0
    ya = frelu(xa)
    ga = frelu_grad(xa)
    check_symbolic_forward(y, [xa], [ya])


def test_sigmoid():
    def fsigmoid(a):
        return np.divide(1.0, (1.0 + np.exp(-a)))
    shape = (SMALL_Y, LARGE_X)
    x = mx.symbol.Variable("x")
    y = mx.sym.sigmoid(x)
    xa = np.random.uniform(low=-1.0,high=1.0,size=shape)
    ya = fsigmoid(xa)
    check_symbolic_forward(y, [xa], [ya])


def np_softmax(x, axis=-1, temperature=1.0):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x/temperature)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


def test_log_softmax():
    ndim = 2
    shape = (SMALL_Y, LARGE_X)
    axis = np.random.randint(0, ndim)
    data = np.random.uniform(-2, 2, size=shape)
    sym = mx.sym.log_softmax(axis=axis-ndim)
    check_symbolic_forward(sym, [data], [np.log(np_softmax(data, axis=axis)+1e-20)])


def test_iadd():
    a = nd.array(np.ones((SMALL_Y, LARGE_X)))
    b = nd.array(np.ones((SMALL_Y, LARGE_X)))
    c = b
    c += a
    assert c.shape == a.shape
    assert c[0][-1] == 2


def test_isub():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    b = nd.array(np.ones((SMALL_Y, LARGE_X)))
    c = a
    c -= b
    assert c.shape == a.shape
    assert c[0][-1] == 2


def test_imul():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    b = nd.array(np.ones((SMALL_Y, LARGE_X)))
    c = b
    c *= a
    assert c.shape == a.shape
    assert c[0][-1] == 3


def test_idiv():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 4)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    c = a
    c /= b
    assert c.shape == a.shape
    assert c[0][-1] == 2


def test_imod():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    c = a
    c %= b
    assert c.shape == a.shape
    assert c[0][-1] == 1


def test_eq():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    c = (a == b)
    assert np.sum(c[0].asnumpy() == 1).all()


def test_neq():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    c = (a != b)
    assert np.sum(c[0].asnumpy() == 1).all()


def test_lt():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    d = (a <= b)
    assert np.sum(d[0].asnumpy() == 1).all()


def test_lte():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    c = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    d = (a <= b)
    e = (a <= c)
    assert np.sum(d[0].asnumpy() == 1).all()
    assert np.sum(e[0].asnumpy() == 1).all()


def test_gt():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    d = (a >= b)
    assert np.sum(d[0].asnumpy() == 1).all()


def test_gte():
    a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
    c = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
    d = (a >= b)
    e = (a >= c)
    assert np.sum(d[0].asnumpy() == 1).all()
    assert np.sum(e[0].asnumpy() == 1).all()


def test_slice_like():
    a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    b = nd.array(np.ones((SMALL_Y//2, LARGE_X//2)))
    c = nd.slice_like(a, b)
    d = nd.slice_like(a, b, axes=(0))
    e = nd.slice_like(a, b, axes=(-1))
    assert c.shape == b.shape
    assert d.shape[0] == b.shape[0]
    assert e.shape[-1] == b.shape[-1]
    assert c[0][-1] == 0
    assert d[-1][0] == (SMALL_Y//2-1)
    assert e[-1][-1] == (SMALL_Y-1)


def test_slice_axis():
    a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
    c = nd.slice_axis(a, axis=0, begin=0, end=SMALL_Y//2)
    d = nd.slice_axis(a, axis=1, begin=0, end=LARGE_X//2)
    assert c.shape[0] == a.shape[0]//2
    assert d.shape[1] == a.shape[1]//2
    assert c[-1][0] == (SMALL_Y//2-1)
    assert d[-1][-1] == (SMALL_Y-1)


def test_one_hot():
    a = nd.array(np.zeros(SMALL_Y))
    a[0] = 1
    a[-1] = 1
    b = nd.one_hot(a, LARGE_X)
    b[0][1] == 1
    b[-1][1] == 1


def test_full():
    a = nd.full((SMALL_Y, LARGE_X), 3)
    assert a.shape == (SMALL_Y, LARGE_X)
    assert a[SMALL_Y//2][LARGE_X//2] == 3
    assert a[-1][-1] == 3


if __name__ == '__main__':
    import nose
    nose.runmodule()
