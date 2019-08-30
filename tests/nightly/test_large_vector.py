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

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, create_vector
from mxnet import gluon, nd
from tests.python.unittest.common import with_seed, teardown

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
    assert a >= low and a < high


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
    mx.nd.waitall()
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


@with_seed()
def test_ndarray_random_exponential():
    a = nd.random.exponential(shape=LARGE_X)
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_gamma():
    a = nd.random.gamma(shape=LARGE_X)
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_generalized_negative_binomial():
    a = nd.random.generalized_negative_binomial(shape=LARGE_X)
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_multinomial():
    a = nd.random.multinomial(nd.random.uniform(shape=LARGE_X))
    assert a[-1] >= 0.
    assert a.shape[0] == 1


@with_seed()
def test_ndarray_random_negative_binomial():
    a = nd.random.negative_binomial(shape=LARGE_X)
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_normal():
    a = nd.random.normal(shape=LARGE_X)
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_poisson():
    a = nd.random.poisson(shape=LARGE_X)
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_randn():
    a = nd.random.randn(LARGE_X)
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_shuffle():
    a = nd.ones(shape=LARGE_X)
    a[-1] = 3
    a = nd.random.shuffle(a)
    unique_a = np.unique(a.asnumpy())
    assert len(unique_a) == 2  # only 2 unique values
    assert unique_a[0] == 1  # first unique value is 1
    assert unique_a[1] == 3  # second unique value is 3
    assert a.shape[0] == LARGE_X


def test_exponent_logarithm_operators():
    a = 2*nd.ones(shape=LARGE_X)
    # exponent
    result = nd.exp(a)
    assert result[-1] == 7.389056
    assert result.shape == a.shape

    # exponent minus 1
    result = nd.expm1(a)
    assert result[-1] == 6.389056
    assert result.shape == a.shape

    # log2
    result = nd.log2(a)
    assert result[-1] == 1
    assert result.shape == a.shape

    # log10
    result = nd.log10(a)
    assert result[-1] == 0.30103
    assert result.shape == a.shape

    # log1p
    result = nd.log1p(a)
    assert result[-1] == 1.0986123
    assert result.shape == a.shape

    # log
    result = nd.log(a)
    assert result[-1] == 0.6931472
    assert result.shape == a.shape


def test_power_operators():
    a = 2*nd.ones(shape=LARGE_X)
    # sqrt
    result = nd.sqrt(a)
    assert result[-1] == 1.4142135
    assert result.shape == a.shape

    # rsqrt
    result = nd.rsqrt(a)
    assert result[-1] == 0.70710677
    assert result.shape == a.shape

    # cbrt
    result = nd.cbrt(a)
    assert result[-1] == 1.2599211
    assert result.shape == a.shape

    # rcbrt
    result = nd.rcbrt(a)
    assert result[-1] == 0.7937005
    assert result.shape == a.shape

    # square
    result = nd.square(a)
    assert result[-1] == 4
    assert result.shape == a.shape

    # reciprocal
    result = nd.reciprocal(a)
    assert result[-1] == 0.5
    assert result.shape == a.shape


def test_sequence_mask():
    # Sequence Mask input [max_sequence_length, batch_size]
    # test with input batch_size = 2
    a = nd.arange(0, LARGE_X * 2).reshape(LARGE_X, 2)

    # test as identity operator
    b = nd.SequenceMask(a)
    assert b[-1][0] == a[-1][0]
    assert b.shape == a.shape

    # test with default mask
    b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                        use_sequence_length=True)
    assert b[0][1] == a[0][1]  # first sequence of each batch kept
    assert b[-1][-1] != a[-1][-1]  # rest sequences masked
    assert b[-1][-1] == 0

    # test with mask value
    b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                        use_sequence_length=True, value=-1)
    assert b[-1][-1] == -1


def test_sequence_reverse():
    a = nd.arange(0, LARGE_X * 2).reshape(LARGE_X, 2)
    # test as reverse operator
    b = nd.SequenceReverse(a)
    assert b[-1][0] == a[0][0]
    assert b.shape == a.shape

    # test with sequence length
    b = nd.SequenceReverse(a, sequence_length=nd.array([2, 3]),
                           use_sequence_length=True)
    assert b[1][0] == a[0][0]  # check if reversed
    assert b[-1][0] == a[-1][0]  # check if intact
    assert b.shape == a.shape


def test_sequence_last():
    a = nd.arange(0, LARGE_X * 2).reshape(LARGE_X, 2)

    # test if returns last sequence
    b = nd.SequenceLast(a)
    assert_almost_equal(b.asnumpy(), a[-1].asnumpy())
    assert b.shape == (2,)

    # test with sequence length
    # parameter sequence_length - NDArray with shape (batch_size)
    # (2,3) indicates 2nd sequence from batch 1 and 3rd sequence from batch 2
    b = nd.SequenceLast(a, sequence_length=mx.nd.array([2, 3]),
                        use_sequence_length=True)
    # check if it takes 2nd sequence from the first batch
    assert b[0] == a[1][0]


# TODO: correctness of layernorm
# numpy implementation for large vector is flaky
def test_layer_norm():
    axis = 0
    eps = 1E-5
    in_shape = LARGE_X

    data = nd.random.normal(0, 1, in_shape)
    gamma = nd.random.normal(0, 1, in_shape)
    beta = nd.random.normal(0, 1, in_shape)
    mx_out = nd.LayerNorm(data, gamma, beta, axis, eps)
    assert mx_out.shape == (in_shape,)


# TODO: correctness of batchnorm
# in future, we could test if mean, var of output
# matches target output's mean, var
def test_batchnorm():
    shape = LARGE_X
    axis = 0  # since vector

    data = mx.nd.ones(shape=shape)
    bn_gamma = mx.nd.random.uniform(shape=shape)
    bn_beta = mx.nd.random.uniform(shape=shape)
    bn_running_mean = mx.nd.zeros(shape)
    bn_running_var = mx.nd.ones(shape)

    output = mx.nd.BatchNorm(data, bn_gamma, bn_beta,
                             bn_running_mean, bn_running_var, axis=axis)
    assert output.shape == (shape,)


def test_add():
    a = nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X)
    c = b
    c = c.__add__(a)
    assert c[-1] == 2
    assert c.shape == a.shape


def test_sub():
    a = 3*nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X)
    c = b
    c = c.__sub__(a)
    assert c[-1] == -2
    assert c.shape == a.shape


def test_rsub():
    a = 3*nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X)
    c = b
    c = c.__rsub__(a)
    assert c[-1] == 2
    assert c.shape == a.shape


def test_neg():
    a = nd.ones(shape=LARGE_X)
    c = a
    c = c.__neg__()
    assert c[-1] == -1
    assert c.shape == a.shape


def test_mul():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__mul__(a)
    assert c[-1] == 6
    assert c.shape == a.shape


def test_div():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__div__(a)
    assert c[-1] == 3/2
    assert c.shape == a.shape


def test_rdiv():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__rdiv__(a)
    assert c[-1] == 2/3
    assert c.shape == a.shape


def test_mod():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__mod__(a)
    assert c[-1] == 1
    assert c.shape == a.shape


def test_rmod():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__rmod__(a)
    assert c[-1] == 2
    assert c.shape == a.shape


def test_imod():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__imod__(a)
    assert c[-1] == 1
    assert c.shape == a.shape


def test_pow():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__pow__(a)
    assert c[-1] == 9
    assert c.shape == a.shape


def test_rpow():
    a = 2*nd.ones(shape=LARGE_X)
    b = 3*nd.ones(shape=LARGE_X)
    c = b
    c = c.__rpow__(a)
    assert c[-1] == 8
    assert c.shape == a.shape


if __name__ == '__main__':
    import nose
    nose.runmodule()
