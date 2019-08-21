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

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, default_context
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
    a = nd.random.multinomial(create_large_vector(LARGE_X))
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
    assert_almost_equal(b, a[-1])
    assert b.shape == (2,)

    # test with sequence length
    # parameter sequence_length - NDArray with shape (batch_size)
    # (2,3) indicates 2nd sequence from batch 1 and 3rd sequence from batch 2
    b = nd.SequenceLast(a, sequence_length=mx.nd.array([2, 3]),
                        use_sequence_length=True)
    # check if it takes 2nd sequence from the first batch
    assert b[0] == a[1][0]


def test_softmax_cross_entropy():
    # SoftmaxCrossEntropy only accepts 2D data
    # dtype of input data, mxnet cross entropy set explicitly to float64
    # numpy implicitly takes care of double precision
    batch_size = 2
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
    x = mx.nd.zeros(LARGE_X)
    t = mx.nd.array([-1])
    index = mx.nd.array([LARGE_X - 1])

    x = mx.nd.contrib.index_copy(x, index, t)
    assert x[-1] == t[-1]


# TODO: correctness of prelu (currently flaky)
def test_leaky_relu():
    a = -1*mx.nd.ones(LARGE_X)

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


def test_layer_norm():
    forward_check_eps = 1E-3
    axis = 0
    eps = 1E-5
    in_shape = LARGE_X

    def npy_layer_norm(data, gamma, beta, axis=0, eps=1E-5):
        broadcast_shape = [1 for _ in range(data.ndim)]
        broadcast_shape[axis] = data.shape[axis]
        mean = data.mean(axis=axis, keepdims=True)
        var = data.var(axis=axis, keepdims=True)
        std = np.sqrt(var + dtype(eps))
        out = np.reshape(gamma, broadcast_shape) * (data - mean) / std + \
              np.reshape(beta, broadcast_shape)
        return out
    data = nd.random.normal(0, 1, in_shape)
    gamma = np.random.normal(0, 1, in_shape)
    beta = np.random.normal(0, 1, in_shape)
    mx_out = nd.LayerNorm(data, gamma, beta, axis, eps)
    np_out = npy_layer_norm(data.asnumpy(), gamma.asnumpy(), beta.asnumpy(), axis, eps)
    assert_almost_equal(np_out, mx_out.asnumpy(), forward_check_eps,
                        forward_check_eps)


# TODO: correctness of dropout
# currently only test for dropout to work
# since testing for correctness involves flakiness issue #14288
def test_dropout():
    shape = LARGE_X
    x = mx.sym.var('data')
    y = mx.sym.Dropout(x, p=1, cudnn_off=True)
    exe = y.simple_bind(ctx=default_context(), data=shape)
    exe.arg_arrays[0][:] = 1
    out = exe.forward(is_train=True)
    out[0].wait_to_read()


def test_activation():
    a = mx.nd.ones(LARGE_X)
    test_x = -2
    a[-1] = test_x

    # Hyperbolic tangent (tanh)
    # y = (exp(x)-exp(-x))/(exp(x)+exp(-x))
    a = mx.nd.Activation(a, act_type="tanh")
    tanh_x = (np.exp(-2) - np.exp(2)) / (np.exp(-2) + np.exp(2))
    assert a[-1] == tanh_x

    # Recitified Linear Unit (relu)
    # y = max(x,0)
    a = mx.nd.Activation(a, act_type="relu")
    assert a[-1] == 0

    # Sigmoid
    # y = x/(1+abs(x))
    a = mx.nd.Activation(a, act_type="sigmoid")
    sigmoid_x = 1 / (1 + math.exp(-test_x))
    assert a[-1] == sigmoid_x

    # Soft Sign
    # y = 1/(1+exp(-x))
    a = mx.nd.Activation(a, act_type="softsign")
    softsign_x = test_x / (1 + abs(test_x))
    assert a[-1] == softsign_x


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
    output.wait_to_read()


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
