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


def create_large_vector(size, dtype="int64"):
    a = nd.arange(0, size, dtype=dtype)
    # Implicitly calling nd.waitall()
    assert a[0] == 0
    return a


def test_slice():
    a = nd.ones(LARGE_X)
    res = nd.slice(a, begin=(LARGE_X - MEDIUM_X), end=LARGE_X)
    assert res.shape[0] == MEDIUM_X


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
    a = nd.random.generalized_negative_binomial(probs=create_large_vector(LARGE_X))
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
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_poisson():
    a = nd.random.poisson(shape=LARGE_X)
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_randint():
    a = nd.random.randint(1500, 9000, shape=LARGE_X, dtype="int64")
    assert a[-1] >= 1500 and a[-1] < 9000
    assert a[-1] == np.int64
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_randn():
    a = nd.random.randn(LARGE_X)
    assert a[-1] >= 0.
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_uniform():
    a = nd.random.uniform(1500, 9000, shape=LARGE_X)
    assert a[-1] >= 1500 and a[-1] < 9000
    assert a.shape[0] == LARGE_X


@with_seed()
def test_ndarray_random_shuffle():
    a = nd.ones(shape=LARGE_X)
    a[-1] == 3
    a = nd.random.shuffle(a)
    unique_a = np.unique(a.asnumpy())
    assert len(unique_a) == 2  # only 2 unique values
    assert unique_a[0] == 1  # first unique value is 1
    assert unique_a[1] == 3  # second unique value is 3
    assert a.shape[0] == LARGE_X


def test_exponent_logarithm_operators():
    a = 2*nd.ones(shape=(LARGE_X))
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
    a = 2*nd.ones(shape=(LARGE_X))
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
    assert_almost_equal(b, a[-1])
    assert b.shape == (2,)

    # test with sequence length
    # parameter sequence_length - NDArray with shape (batch_size)
    # (2,3) indicates 2nd sequence from batch 1 and 3rd sequence from batch 2
    b = nd.SequenceLast(a, sequence_length=mx.nd.array([2, 3]),
                        use_sequence_length=True)
    # check if it takes 2nd sequence from the first batch
    assert b[0] == a[1][0]


if __name__ == '__main__':
    import nose
    nose.runmodule()
