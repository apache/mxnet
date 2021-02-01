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

import os
import sys
import tempfile
import math
import numpy as _np
import mxnet as mx

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../python/unittest/'))

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, default_context, check_symbolic_forward, create_2d_np_tensor, use_np
from mxnet import gluon, np, npx
import pytest
from tests.python.unittest.common import assertRaises
from mxnet.base import MXNetError

# dimension constants
MEDIUM_X = 10000
LARGE_X = 100000000
SMALL_X = 100
SMALL_Y = 50
INT_OVERFLOW = 2**31
HALF_INT_OVERFLOW = 2**30
DOUBLE_INT_OVERFLOW = 2**32


@use_np
def test_gluon_embedding():
    m = gluon.nn.Embedding(SMALL_Y, MEDIUM_X)
    m.initialize()
    a = np.zeros((MEDIUM_X, SMALL_Y))
    b = m(a)
    assert b.shape == (MEDIUM_X, SMALL_Y, MEDIUM_X)
    assert b.asnumpy().size == MEDIUM_X * SMALL_Y * MEDIUM_X


@use_np
def test_fully_connected():
    a = np.ones(shape=(LARGE_X, SMALL_Y))
    b = np.ones(shape=(SMALL_Y, SMALL_Y))
    c = np.ones(shape=(b.shape[0],))

    # w/o bias
    res = mx.npx.fully_connected(a, b, num_hidden=b.shape[0], no_bias=True)
    assert np.sum(res[-1] == a.shape[1]) == b.shape[0]

    # w/ bias
    res = mx.npx.fully_connected(a, b, c, num_hidden=b.shape[0], no_bias=False)
    assert np.sum(res[-1] == a.shape[1] + 1) == b.shape[0]


@use_np
def test_dense():
    data = np.ones(shape=(LARGE_X, SMALL_X))
    linear = gluon.nn.Dense(SMALL_Y)
    linear.initialize()
    res = linear(data)
    assert res.shape == (LARGE_X, SMALL_Y)


@use_np
def test_softmax():
    input_data = np.ones((SMALL_Y, LARGE_X))
    for axis in [0, 1]:
        true_output = np.full((SMALL_Y, LARGE_X), (1 / input_data.shape[axis]))
        output = npx.softmax(input_data, axis=axis)
        assert_almost_equal(output.asnumpy(), true_output, rtol=1e-5, atol=1e-5)


'''
  _ _ _  _ _ __  _ __ _  _
 | ' \ || | '  \| '_ \ || |
 |_||_\_,_|_|_|_| .__/\_, |
                |_|   |__/
'''

@use_np
def test_ones():
    A = np.ones((INT_OVERFLOW, 2))
    assert A.shape == (INT_OVERFLOW, 2)
    assert A[0][0] == 1


@use_np
def test_zeros():
    A = np.zeros((INT_OVERFLOW, 2))
    assert A.shape == (INT_OVERFLOW, 2)
    assert A[0][0] == 0


@use_np
def test_ones_like():
    inp = np.ones((2, INT_OVERFLOW))
    out = np.ones_like(inp)
    assert out.shape == inp.shape
    assert out[0, 0] == 1 and out[-1, -1] == 1


@use_np
def test_zeros_like():
    inp = np.ones((INT_OVERFLOW, 2))
    out = np.zeros_like(inp)
    assert out.shape == inp.shape
    assert out[0, 0] == 0 and out[-1, -1] == 0

@use_np
def test_abs():
    # abs absolute and fabs are the same thing
    inp = np.zeros((INT_OVERFLOW, 2))
    inp[-1, -1] = -1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.abs(inp)
        out.backward()
    assert out.shape == (INT_OVERFLOW, 2)
    assert out[-1, -1] == 1
    assert inp.grad.shape == (INT_OVERFLOW, 2)
    assert inp.grad[-1, -1] == -1


@use_np
def test_binary_broadcast():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 1))
    C = np.add(A, B)
    assert C.shape == (INT_OVERFLOW, 2)
    assert C[0][0] == 2


@use_np
def test_all():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.all(A)
    assert B == True
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0 


@use_np
def test_amin():
    inp = np.ones((INT_OVERFLOW, 2))
    inp[-1, -1] = -1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.amin(inp)
        out.backward()
    assert out == -1.0
    assert inp.grad.shape == (INT_OVERFLOW, 2)
    assert inp.grad[0, 0] == 0 and inp.grad[-1, -1] == 1


@use_np
def test_amax():
    inp = np.zeros((INT_OVERFLOW, 2))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.amax(inp)
        out.backward()
    assert out == 1.0
    assert inp.grad.shape == (INT_OVERFLOW, 2)
    assert inp.grad[0, 0] == 0 and inp.grad[-1, -1] == 1


@use_np
def test_argmin():
    A = np.ones((INT_OVERFLOW, 2))
    A[10][1] = -1
    A.attach_grad()
    with mx.autograd.record():
        B = np.argmin(A)
    assert B == 21
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0


@use_np
def test_argmax():
    A = np.zeros((INT_OVERFLOW, 2))
    A[10][1] = 1
    A.attach_grad()
    with mx.autograd.record():
        B = np.argmax(A)
    assert B == 21
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0


@use_np
@pytest.mark.skip(reason='times out (20 mins)')
def test_trigonometric_family():
    def batch_check(x, funcs):
        for f in funcs:
            one = np.ones((1))
            x.attach_grad()
            one.attach_grad()
            with mx.autograd.record():
                y = f(x)
                _ = f(one)
            assert y.shape == (INT_OVERFLOW, 2)
            assert y[0][0] == _
            y.backward()
            _.backward()
            assert x.grad.shape == (INT_OVERFLOW, 2)
            assert x.grad[0][0] == one.grad
    A = np.ones((INT_OVERFLOW, 2))
    batch_check(A, [np.arccos, np.arccosh, np.arcsin, \
        np.arcsin, np.arctan, np.arctanh, np.sin, np.cos, \
        np.tan, np.sinh, np.cosh, np.tanh])


@use_np
def test_any():
    A = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.any(A)
    assert B == False
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0


@use_np
def test_append():
    A = np.ones((1, INT_OVERFLOW))
    B = np.ones((2, INT_OVERFLOW))
    A.attach_grad() 
    with mx.autograd.record():
        C = np.append(A, B, axis=0)
    assert C.shape == (3, INT_OVERFLOW)
    assert C[2][0] == 1
    C.backward()
    assert A.grad.shape == (1, INT_OVERFLOW)
    assert A[0][0] == 1


@use_np
def test_arange():
    A = np.arange(INT_OVERFLOW, dtype='int32')
    assert A.shape == (INT_OVERFLOW, )
    assert A[100] == 100


@use_np
def test_argsort():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.argsort(A)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 0
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A[0][0] == 1


@use_np
def test_atleast_xd_family():
    def batch_check(x, funcs, shapes):
        for f, s in zip(funcs, shapes):
            x.attach_grad()
            with mx.autograd.record():
                y = f(x)
            assert y.shape == s
            y.backward()
            assert x.grad.shape == (INT_OVERFLOW, )
            assert x.grad[0] == 0
    A = np.zeros((INT_OVERFLOW))
    batch_check(A, [np.atleast_1d, np.atleast_2d, np.atleast_3d], \
            [(INT_OVERFLOW, ), (1, INT_OVERFLOW), (1, INT_OVERFLOW, 1)])


@use_np
def test_average():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.average(A)
    assert B == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert_almost_equal(A.grad[0][0], np.array([1.0 / DOUBLE_INT_OVERFLOW]), \
            rtol=1e-3, atol=1e-5)


@use_np
def test_bincount():
    A = np.ones((INT_OVERFLOW), dtype='int32')
    A[0] = 0
    A.attach_grad()
    with mx.autograd.record():
        B = np.bincount(A)
    assert B.shape == (2,)
    assert B[-1] == INT_OVERFLOW - 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, )
    assert A.grad[0] == 0 


@use_np
def test_bitwise_family():
    def batch_check(x1, x2, funcs):
        x1.attach_grad()
        for f in funcs:
            with mx.autograd.record():
                y = f(x1, x2)
                y.backward()
            one = np.ones((1), dtype='int32')
            assert y.shape == (INT_OVERFLOW, 2)
            assert y[-1, -1] == f(one, one)
            assert x1.grad.shape == x1.shape
            assert x1.grad[-1, -1] == 0
    # test on broadcast input
    inp1 = np.ones((INT_OVERFLOW, 1), dtype='int32')
    inp2 = np.ones((INT_OVERFLOW, 2), dtype='int32')
    batch_check(inp1, inp2, [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
    out = np.bitwise_not(inp1)
    assert out.shape == (INT_OVERFLOW, 1)
    assert out[0] == np.bitwise_not(np.ones((1), dtype='int32')) 


@use_np
def test_blackman():
    data = np.blackman(INT_OVERFLOW)
    ind = int(INT_OVERFLOW / 6)
    ref = 0.42 - 0.5*math.cos(2*math.pi*ind/INT_OVERFLOW) \
        + 0.08*math.cos(4*math.pi*ind/INT_OVERFLOW)
    assert_almost_equal(data[ind], ref, rtol=1e-3, atol=1e-5)


@use_np
def test_broadcast_to():
    A = np.ones((2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.broadcast_to(A, (INT_OVERFLOW, 2))
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (2, )
    with mx.autograd.record():
        B = np.broadcast_to(A.reshape(2, 1), (2, INT_OVERFLOW))
    assert B.shape == (2, INT_OVERFLOW)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (2, )


@use_np
def test_root_family():
    def batch_check(x, funcs, grads):
        for f, g in zip(funcs, grads):
            x.attach_grad()
            with mx.autograd.record():
                y = f(x)
            assert y.shape == (INT_OVERFLOW, 2)
            assert y[0][0] == 1
            y.backward()
            assert x.grad.shape == (INT_OVERFLOW, 2)
            assert_almost_equal(A.grad[0][0], np.array(g), \
                rtol=1e-3, atol=1e-5)
    A = np.ones((INT_OVERFLOW, 2))
    batch_check(A, [np.sqrt, np.cbrt], [0.5, 1.0 / 3])


@use_np
def test_ceil_floor():
    def batch_check(x, funcs):
        for f in funcs:
            x.attach_grad()
            with mx.autograd.record():
                y = f(x)
            assert y.shape == (INT_OVERFLOW, 2)
            assert y[0][0] == 1
            y.backward()
            assert x.grad.shape == (INT_OVERFLOW, 2)
            assert x.grad[0][0] == 0
    A = np.ones((INT_OVERFLOW, 2))
    batch_check(A, [np.ceil, np.floor])


@use_np
def test_clip():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.clip(A, 1, 1)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 1


@use_np
def test_column_stack():
    A = np.ones(INT_OVERFLOW)
    A.attach_grad()
    with mx.autograd.record():
        B = np.column_stack((A, A))
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, )
    assert A.grad[0] == 2


@use_np
def test_concatenate():
    def batch_check(x1, x2, axises, shapes):
        for a, s in zip(axises, shapes):
            x1.attach_grad()
            with mx.autograd.record():
                y = np.concatenate((x1, x2), axis=a)
            assert y.shape == s
            y.backward()
            assert x1.grad.shape == (2, INT_OVERFLOW)
            assert x1.grad[0][0] == 1
    A = np.ones((2, INT_OVERFLOW))
    B = np.ones((1, INT_OVERFLOW))
    batch_check(A, B, [0, None], \
            [(3, INT_OVERFLOW), (int(INT_OVERFLOW * 3), )])

@use_np
def test_copysign():
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp1[-1, -1] = 2
    inp1.attach_grad()
    inp2 = np.array([-1])
    with mx.autograd.record():
        out = np.copysign(inp1, inp2)
        out.backward()
    assert out.shape == (INT_OVERFLOW, 2)
    assert out[-1 ,-1] == -2
    assert inp1.grad.shape == (INT_OVERFLOW, 2)
    assert inp1.grad[-1, -1] == -1


@use_np
def test_random_uniform():
    A = np.random.uniform(low=0, high=1.0, size=(INT_OVERFLOW))
    assert A[0] <= 1 and A[0] >= 0


@use_np
def test_random_normal():
    A = np.random.normal(loc=0, scale=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'

@use_np
@pytest.mark.skip(reason='times out (20 mins)')
def test_random_gamma():
    A = np.random.gamma(shape=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_exponential():
    A = np.random.exponential(size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_laplace():
    A = np.random.laplace(loc=0, scale=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_choice():
    A = np.random.choice(a=10, size=(INT_OVERFLOW))
    assert A[0] <= 10 and A[0] >= 0


@use_np
def test_random_gumbel():
    A = np.random.gumbel(loc=0, scale=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_logistic():
    A = np.random.logistic(loc=0, scale=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'

@use_np
@pytest.mark.skip(reason='times out (20 mins)')
def test_random_multinomial():
    A = np.random.multinomial(pvals=np.zeros(INT_OVERFLOW), n=1)
    assert A[-1] == 1

@use_np
def test_random_pareto():
    A = np.random.pareto(a=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_power():
    A = np.random.power(a=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_rayleigh():
    A = np.random.rayleigh(scale=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_weibull():
    A = np.random.weibull(a=1.0, size=(INT_OVERFLOW))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_shuffle():
    A = np.ones((INT_OVERFLOW, 2))
    np.random.shuffle(A)
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_lognormal():
    A = np.random.lognormal(mean=0, sigma=1.0, size=(2**31))
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_random_randint():
    A = np.random.randint(low=0, high=5, size=(2, 2**31))
    assert A[0][0] < 5 and A[0][0] >= 0


@use_np
def test_slice_assign():
    # test _slice_assign
    A = np.zeros((INT_OVERFLOW, 2))
    A[-1] = np.ones((1))
    assert A[-1, 0] == 1 and A[-1, 1] == 1
    # test _slice_assign_scalar
    B = np.zeros((INT_OVERFLOW, 2))
    B[-1] = 2
    assert B[-1, 0] == 2 and B[-1, 1] == 2

@use_np
def test_flatnonzero():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.flatnonzero(inp)
        out.backward()
    assert out.shape == (1, )
    assert out[0] == int(2 * INT_OVERFLOW - 1)
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0

@use_np
def test_ravel():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[0, -1], inp[-1, -1] = 1, 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.ravel(inp)
        out.backward()
    assert out.shape == (DOUBLE_INT_OVERFLOW, )
    assert out[INT_OVERFLOW-1] == 1 and out[-1] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1

@use_np
def test_mean():
    inp = np.arange(DOUBLE_INT_OVERFLOW).reshape((2, INT_OVERFLOW))
    inp.attach_grad()
    with mx.autograd.record():
        out = np.mean(inp, axis=1)
        out.backward()
    assert out.shape == (2, )
    assert_almost_equal(out[0], np.array((HALF_INT_OVERFLOW-0.5)), \
                rtol=1e-3, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert_almost_equal(inp.grad[-1, -1], np.array((1.0/INT_OVERFLOW)), \
                rtol=1e-3, atol=1e-5)

@use_np
def test_median():
    inp = np.arange(DOUBLE_INT_OVERFLOW).reshape((2, INT_OVERFLOW))
    inp.attach_grad()
    with mx.autograd.record():
        out = np.median(inp, axis=1)
        out.backward()
    assert out.shape == (2, )
    assert_almost_equal(out[0], np.array((HALF_INT_OVERFLOW-0.5)), \
                rtol=1e-3, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0

@use_np
def test_percentile():
    # np.percentile and np.quantile share the same implementation
    inp = np.arange(DOUBLE_INT_OVERFLOW).reshape((2, INT_OVERFLOW))
    inp.attach_grad()
    with mx.autograd.record():
        out = np.percentile(inp, 50, axis=1)
        out.backward()
    assert out.shape == (2, )
    assert_almost_equal(out[0], np.array((HALF_INT_OVERFLOW-0.5)), \
                rtol=1e-3, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0

@use_np
def test_shares_memory():
    # np.shares_memory and np.may_share_memory share the same implementation
    inp = np.ones((2, INT_OVERFLOW))
    out = np.shares_memory(inp[0,:100], inp[0,100:])
    out2 = np.shares_memory(inp[1,:101], inp[1,100:])
    assert out == False and out2 == True

@use_np
def test_where():
    inp1 = np.zeros((2, INT_OVERFLOW))
    inp1[-1, -1] = 1
    inp2 = inp1 + 1
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.where(inp1==0, inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[0, 0] == 0 and out[-1, -1] == 2
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[0, 0] == 1 and inp1.grad[-1, -1] == 0
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[0, 0] == 0 and inp2.grad[-1, -1] == 1
    # one side is scalar
    with mx.autograd.record():
        out = np.where(inp1==0, inp1, 2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[0, 0] == 0 and out[-1, -1] == 2
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[0, 0] == 1 and inp1.grad[-1, -1] == 0
    # both sides ar scalar
    with mx.autograd.record():
        out = np.where(inp1==0, 0, 2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[0, 0] == 0 and out[-1, -1] == 2
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 0

@use_np
def test_logical_family():
    def batch_check(x1, x2, funcs):
        x1.attach_grad()
        for f in funcs:
            with mx.autograd.record():
                y = f(x1, x2)
                y.backward()
            assert y.shape == x1.shape
            assert y[0] == f(x1[0], x2[0])
            assert x1.grad.shape == x1.shape
            assert x1.grad[0] == 0

    inp1 = np.zeros((INT_OVERFLOW), dtype='int32')
    inp2 = np.ones((INT_OVERFLOW), dtype='int32')
    batch_check(inp1, inp2, [np.logical_and, np.logical_or, np.logical_xor])
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.logical_not(inp2)
        out.backward()
    assert out.shape == inp2.shape
    assert out[0] == 0
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[0] == 0


@use_np
def test_deg_rad():
    # deg2rad is the same thing as radians
    # rad2deg is the same thing as degrees
    inp = np.zeros((INT_OVERFLOW, 2))
    inp[-1, -1] = 180
    inp.attach_grad()
    with mx.autograd.record():
        out = np.deg2rad(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, 0] == 0
    assert_almost_equal(out[-1, -1], np.array([np.pi]), rtol=1e-5, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert_almost_equal(inp.grad[0, 0], np.array([1.0 / 180 * np.pi]), rtol=1e-5, atol=1e-5)
    out.attach_grad()
    with mx.autograd.record():
        out2 = np.rad2deg(out)
        out2.backward()
    assert out2.shape == out.shape
    assert out2[0, 0] == 0 and out2[-1, -1] == 180
    assert out.grad.shape == out.shape
    assert_almost_equal(out.grad[0, 0], np.array([180.0 / np.pi]), rtol=1e-5, atol=1e-5)


@use_np
def test_divide():
    # np.divide and np.true_divide are the same thing
    inp = np.ones((INT_OVERFLOW, 2))
    inp[-1, -1] = 10
    inp.attach_grad()
    with mx.autograd.record():
        out = np.divide(inp, np.array([2, 3]))
        out.backward()
    assert out.shape == inp.shape
    assert_almost_equal(out[-1, -1], np.array([10 / 3]), rtol=1e-5, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert_almost_equal(inp.grad[-1, -1], np.array([1.0 / 3]), rtol=1e-5, atol=1e-5)


@use_np
def test_minimum():
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp1[-1, -1] = -1
    inp2 = np.zeros((INT_OVERFLOW, 1))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.minimum(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == -1
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 1 and inp1.grad[0, 0] == 0
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == 1 and inp2.grad[0] == 2


@use_np
def test_maximum():
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp1[-1, -1] = -1
    inp2 = np.zeros((INT_OVERFLOW, 1))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.maximum(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == 0
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 0 and inp1.grad[0, 0] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == 1 and inp2.grad[0] == 0


@use_np
def test_eye():
    N = 2**16
    data1 = np.eye(N)
    assert data1.shape == (N, N)
    for i in range(N):
        assert data1[i, i] == 1
    assert data1[-1, -2] == 0 and data1[0, 1] == 0
    data2 = np.eye(N, M=N-1, k=-1)
    assert data2.shape == (N, N-1)
    for i in range(1, N):
        assert data2[i, i-1] == 1
    assert data2[0, 0] == 0 and data2[-1, -2] == 0


@use_np
def test_fix():
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = -2.9
    inp[0, 0] = 2.9
    inp.attach_grad()
    with mx.autograd.record():
        out = np.fix(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, 0] == 2 and out[-1, -1] == -2
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0


@use_np
def test_flip():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[0, 0] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.flip(inp, axis=0)
        out.backward()
    assert out.shape == inp.shape
    assert out[1, 0] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0] == 1
    out2 = np.flip(inp, axis=1)
    assert out2[0, -1] == 2


@use_np
def test_fliplr():
    inp = np.zeros((1, 2, INT_OVERFLOW))
    inp[0, 0, 0] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.fliplr(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, 1, 0] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0, 0] == 1


@use_np
def test_flipud():
    inp = np.zeros((2, 1, INT_OVERFLOW))
    inp[0, 0, 0] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.flipud(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[1, 0, 0] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0, 0] == 1


@use_np
def test_full():
    data1 = np.full((INT_OVERFLOW, 2), np.array([1, 2]))
    assert data1.shape == (INT_OVERFLOW, 2)
    assert data1[-1, 0] == 1 and data1[-1, 1] == 2
    data2 = np.full((2, INT_OVERFLOW), 3)
    assert data2.shape == (2, INT_OVERFLOW)
    assert data2[-1, -1] == 3


@use_np
def test_full_like():
    inp = np.zeros((INT_OVERFLOW, 2))
    out = np.full_like(inp, 2)
    assert out.shape == inp.shape
    assert out[-1, -1] == 2


@use_np
def test_comparison_family():
    def batch_check(funcs, exp):
        inp1.attach_grad()
        for f, e in zip(funcs, exp):
            with mx.autograd.record():
                out = f(inp1, inp2)
                out.backward()
            assert out.shape == inp1.shape
            assert (out[0, 0], out[-1, -1]) == e
            assert inp1.grad.shape == inp1.shape
            assert inp1.grad[-1, -1] == 0
    
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp2 = np.zeros((INT_OVERFLOW, 2))
    inp2[-1, -1] = 1
    batch_check([np.greater, np.greater_equal, \
        np.less, np.less_equal, np.equal, np.not_equal], \
        [(True, False), (True, True), \
        (False, False), (False, True), (False, True), (True, False)])


@use_np
def test_lcm():
    inp1 = np.ones((2, INT_OVERFLOW), dtype='int32')
    inp2 = np.ones((2, INT_OVERFLOW), dtype='int32')
    inp1[-1, -1] = 3
    inp2[-1, -1] = 5
    inp1.attach_grad()
    with mx.autograd.record():
        out = np.lcm(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == 15
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 0


@use_np
def test_gcd():
    inp1 = np.ones((2, INT_OVERFLOW), dtype='int32')
    inp2 = np.ones((2, INT_OVERFLOW), dtype='int32')
    inp1[-1, -1] = 12
    inp2[-1, -1] = 20
    inp1.attach_grad()
    with mx.autograd.record():
        out = np.gcd(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == 4
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 0


@use_np
def test_log_family():
    def batch_check(funcs, exp):
        inp.attach_grad()
        for f, e in zip(funcs, exp):
            with mx.autograd.record():
                out = f(inp)
                out.backward()
            assert out.shape == inp.shape
            assert_almost_equal(out[-1, -1], np.array([e[0]]), \
                rtol=1e-5, atol=1e-5)
            assert inp.grad.shape == inp.shape
            assert_almost_equal(inp.grad[-1, -1], np.array([e[1]]), \
                rtol=1e-5, atol=1e-5)

    inp = np.ones((INT_OVERFLOW, 2))
    inp[-1, -1] = 100
    batch_check([np.log, np.log10, np.log2, np.log1p], \
        [(4.6051702, 0.01), (2, 0.00434294), \
        (6.643856, 0.01442695), (4.6151204, 0.00990099)])


@use_np
def test_expand_dims():
    inp = np.zeros((INT_OVERFLOW))
    inp[-1] = 1
    out1 = np.expand_dims(inp, axis=0)
    out2 = np.expand_dims(out1, axis=2)
    assert out1.shape == (1, INT_OVERFLOW)
    assert out2.shape == (1, INT_OVERFLOW, 1)
    assert out1[0, -1] == 1
    assert out2[0, -1, 0] == 1


@use_np
def test_hamming():
    data = np.hamming((INT_OVERFLOW))
    ind = int(INT_OVERFLOW / 6)
    ref = 0.54 - 0.46*math.cos(2*math.pi*ind/(INT_OVERFLOW-1))
    assert data.shape == (INT_OVERFLOW, )
    assert_almost_equal(data[ind], ref, rtol=1e-3, atol=1e-5)


@use_np
def test_hanning():
    data = np.hanning((INT_OVERFLOW))
    ind = int(INT_OVERFLOW / 6)
    ref = 0.5 - 0.5*math.cos(2*math.pi*ind/(INT_OVERFLOW-1))
    assert data.shape == (INT_OVERFLOW, )
    assert_almost_equal(data[ind], ref, rtol=1e-3, atol=1e-5)


@use_np
def test_fmax():
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp1[-1, -1] = -1
    inp2 = np.zeros((INT_OVERFLOW, 1))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.fmax(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == 0
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 0 and inp1.grad[0, 0] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == 1 and inp2.grad[0] == 0


@use_np
def test_fmin():
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp1[-1, -1] = -1
    inp2 = np.zeros((INT_OVERFLOW, 1))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.fmin(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == -1
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 1 and inp1.grad[0, 0] == 0
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == 1 and inp2.grad[0] == 2


@use_np
def test_fmod():
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp2 = np.ones((INT_OVERFLOW, 1))
    inp1[-1, -1], inp2[-1, -1] = 11, 7
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.fmod(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == 4
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[0, 0] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == -1 and inp2.grad[0] == -2


@use_np
def test_mod():
    # np.mod and np.remainder are the same thing
    inp1 = np.ones((INT_OVERFLOW, 2))
    inp2 = np.ones((INT_OVERFLOW, 1))
    inp1[-1, -1], inp2[-1, -1] = 11, 7
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.mod(inp1, inp2)
        out.backward()
    assert out.shape == inp1.shape
    assert out[-1, -1] == 4
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[0, 0] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == -1 and inp2.grad[0] == -2


@use_np
def test_value_check_family():
    def batch_check(funcs, ref):
        inp.attach_grad()
        for f, r in zip(funcs, ref):
            with mx.autograd.record():
                out = f(inp)
                out.backward()
            assert out.shape == inp.shape
            for i in range(4):
                assert out[i, -1] == r[i]
            assert inp.grad.shape == inp.shape
            assert inp.grad[-1, -1] == 0

    inp = np.zeros((4, INT_OVERFLOW))
    inp[1:, -1] = np.array([np.inf, -np.inf, np.nan])
    batch_check([np.isinf, np.isneginf, np.isposinf, np.isnan, np.isfinite], \
        [(False, True, True, False), (False, False, True, False), \
        (False, True, False, False), (False, False, False, True), \
        (True, False, False, False)])


@use_np
def test_rint():
    inp = np.zeros((INT_OVERFLOW, 2))
    inp[0, 0], inp[-1, -1] = 2.1,  2.9
    inp.attach_grad()
    with mx.autograd.record():
        out = np.rint(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, 0] == 2 and out[-1, -1] == 3
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0


@use_np
def test_invert():
    inp = np.zeros((2, INT_OVERFLOW), dtype='uint8')
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.invert(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, 0] == 255 and out[-1, -1] == 254
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0


@use_np
def test_exp():
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.exp(inp)
        out.backward()
    assert out.shape == inp.shape
    assert_almost_equal(out[0, 0], np.array(np.e**1), rtol=1e-5, atol=1e-5)
    assert_almost_equal(out[-1, -1], np.array(np.e**2), rtol=1e-5, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert_almost_equal(inp.grad[-1, -1], out[-1, -1], rtol=1e-5, atol=1e-5)


@use_np
def test_expm1():
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.expm1(inp)
        out.backward()
    assert out.shape == inp.shape
    assert_almost_equal(out[0, 0], np.array(np.e**1 - 1), rtol=1e-5, atol=1e-5)
    assert_almost_equal(out[-1, -1], np.array(np.e**2 - 1), rtol=1e-5, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert_almost_equal(inp.grad[-1, -1], np.array(np.e**2), rtol=1e-5, atol=1e-5)


@use_np
@pytest.mark.skip(reason='to be moved to new file and run separately as it takes lot of memory')
def test_frexp():
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = 9
    out1, out2 = np.frexp(inp)
    assert_almost_equal(inp[-1, -1], out1[-1, -1] * 2 ** out2[-1, -1], \
        rtol=1e-5, atol=1e-5)


@use_np
def test_reciprocal():
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = 3
    inp.attach_grad()
    with mx.autograd.record():
        out = np.reciprocal(inp)
        out.backward()
    assert out.shape == inp.shape
    assert_almost_equal(out[-1, -1], np.array([1.0/3]), rtol=1e-5, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert_almost_equal(inp.grad[-1, -1], np.array([-1.0/3**2]), \
        rtol=1e-5, atol=1e-5)


@use_np
def test_sum():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = 10
    inp.attach_grad()
    with mx.autograd.record():
        out1 = np.sum(inp, axis=1)
        out1.backward()
    assert out1.shape == (2, )
    assert out1[0] == 0 and out1[1] == 10
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1
    with mx.autograd.record():
        out2 = np.sum(inp, axis=0)
        out2.backward()
    assert out2.shape == (INT_OVERFLOW, )
    assert out2[0] == 0 and out2[-1] == 10
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1


@use_np
def test_negative():
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = -2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.negative(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, 0] == -1 and out[-1, -1] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == -1


@use_np
def test_identity():
    M = 2**16
    data = np.identity(M)
    assert data.shape == (M, M)
    assert data[0, 0] == 1 and data[-1, -1] == 1 and data[-1, -2] == 0


@use_np
def test_square():
    inp = np.ones((INT_OVERFLOW, 2))
    inp[-1, -1] = 3
    inp.attach_grad()
    with mx.autograd.record():
        out = np.square(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[-1, -1] == 9
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 6


@use_np
def test_sign():
    inp = np.zeros((INT_OVERFLOW, 2))
    inp[-1, -1], inp[-2, -1] = 2, -2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.sign(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, 0] == 0 and out[-1, -1] == 1 and out[-2, -1] == -1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0


@use_np
def test_prod():
    inp = np.ones((2, INT_OVERFLOW))
    inp[0, 0], inp[-1, -1] = 2, 10
    inp.attach_grad()
    with mx.autograd.record():
        out1 = np.prod(inp, axis=1)
        out1.backward()
    assert out1.shape == (2, )
    assert out1[0] == 2 and out1[1] == 10
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1
    with mx.autograd.record():
        out2 = np.prod(inp, axis=0)
        out2.backward()
    assert out2.shape == (INT_OVERFLOW, )
    assert out2[0] == 2 and out2[-1] == 10
    assert inp.grad.shape == inp.shape


@use_np
def test_add():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 2))
    A[-1, -1] = 2
    A.attach_grad()
    with mx.autograd.record():
        C = np.add(A, B)
        C.backward()
    assert C.shape == (INT_OVERFLOW, 2)
    assert C[-1, -1] == 3
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[-1, -1] == 1


@use_np
def test_hypot():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 2))
    A[-1, -1], B[-1, -1] = 3, 4
    A.attach_grad()
    with mx.autograd.record():
        C = np.hypot(A, B)
        C.backward()
    assert C.shape == A.shape
    assert C[-1, -1] == 5
    assert A.grad.shape == A.shape
    assert_almost_equal(A.grad[-1, -1], np.array([0.6]), rtol=1e-5, atol=1e-5)


@use_np
def test_power():
    A = np.full((2, INT_OVERFLOW), 2)
    B = np.ones((2, INT_OVERFLOW))
    B[-1, -1] = 3
    A.attach_grad()
    B.attach_grad()
    with mx.autograd.record():
        C = np.power(A, B)
        C.backward()
    assert C.shape == A.shape
    assert C[-1, -1] == 8
    assert A.grad.shape == A.shape
    assert A.grad[-1, -1] == 12
    assert B.grad.shape == B.shape
    assert_almost_equal(B.grad[-1, -1], 2**3 * np.log(2), rtol=1e-5, atol=1e-5)


@use_np
def test_ldexp():
    A = np.ones((2, INT_OVERFLOW))
    B = np.ones((2, INT_OVERFLOW))
    A[-1, -1], B[-1, -1] = 5, 2
    A.attach_grad()
    B.attach_grad()
    with mx.autograd.record():
        C = np.ldexp(A, B)
        C.backward()
    assert C.shape == A.shape
    assert C[-1, -1] == 20
    assert A.grad.shape == A.shape
    assert A.grad[-1, -1] == 4
    assert B.grad.shape == B.shape
    assert_almost_equal(B.grad[-1, -1], A[-1, -1] * 2**B[-1, -1] * np.log(2), \
        rtol=1e-5, atol=1e-5)


@use_np
def test_multiply():
    A = np.ones((2, INT_OVERFLOW))
    B = np.ones((2, INT_OVERFLOW))
    A[-1, -1], B[-1, -1] = 2, 3
    A.attach_grad()
    B.attach_grad()
    with mx.autograd.record():
        C = np.multiply(A, B)
        C.backward()
    assert C.shape == A.shape
    assert C[0, 0] == 1 and C[-1, -1] == 6
    assert A.grad.shape == A.shape
    assert A.grad[-1, -1] == B[-1, -1]
    assert B.grad.shape == B.shape
    assert B.grad[-1, -1] == A[-1, -1]


@use_np
def test_subtract():
    A = np.zeros((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 2))
    A[-1, -1] = 3
    A.attach_grad()
    B.attach_grad()
    with mx.autograd.record():
        C = np.subtract(A, B)
        C.backward()
    assert C.shape == (INT_OVERFLOW, 2)
    assert C[0, 0] == -1 and C[-1][-1] == 2
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 1
    assert B.grad.shape == (INT_OVERFLOW, 2)
    assert B.grad[0][0] == -1


@use_np
def test_diag():
    # test diag extraction
    inp = np.zeros((2, INT_OVERFLOW+2))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.diag(inp, k=INT_OVERFLOW)
        out.backward()
    assert out.shape == (2, )
    assert out[1] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[1, -1] == 1 and inp.grad[0, -2] == 1
    # now test mat generation
    N = 2**16
    inp = np.ones((N))
    inp[-1] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.diag(inp)
        out.backward()
    assert out.shape == (N, N)
    assert out[-1, -1] == 2 and out[0, 0] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1] == 1

    
@use_np
def test_diag_indices_from():
    N = 2**16
    inp = np.zeros((N, N))
    inp.attach_grad()
    with mx.autograd.record():
        dim1, dim2 = np.diag_indices_from(inp)
        dim1.backward()
    assert dim1.shape == (N, ) and dim2.shape == (N, )
    assert dim1[-1] == N-1 and dim2[-1] == N-1
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0] == 0

    
@use_np
def test_diagflat():
    N = 2**15
    inp = np.ones((2, N))
    inp[-1, -1] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.diagflat(inp)
        out.backward()
    assert out.shape == (N*2, N*2)
    assert out[-1, -1] == 2 and out[-1, -2] == 0
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1


@use_np
def test_diagonal():
    inp = np.zeros((2, INT_OVERFLOW+2))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.diagonal(inp, offset=INT_OVERFLOW)
        out.backward()
    assert out.shape == (2, )
    assert out[1] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[1, -1] == 1 and inp.grad[0, -2] == 1
    # now test with axes specified
    N = 2**16
    inp = np.zeros((N, N, 2))
    inp[-1, -1] = np.array([1, 2])
    inp.attach_grad()
    with mx.autograd.record():
        out = np.diagonal(inp, offset=0, axis1=0, axis2=1)
        out.backward()
    assert out.shape == (2, N)
    assert out[0, -1] == 1 and out[1, -1] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1, 0] == 1 and inp.grad[-1, -1, 1] == 1


def test_roll():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.roll(inp, 1)
        # equivalent but slower
        # out = np.roll(inp, shift=(1, 1), axis=(0, 1))
        out.backward()
    assert out.shape == (2, INT_OVERFLOW)
    assert out[0, 0] == 1, out[-1, -1] == 0
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1

    
def test_polyval():
    poly = np.array([1, 1, 5])
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = 2
    poly.attach_grad()
    inp.attach_grad()
    with mx.autograd.record():
        out = np.polyval(poly, inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[-1, -1] == 11 and out[0, 0] == 5
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 5
    assert poly.grad.shape == poly.shape
    assert poly.grad[0] == 4


@use_np
def test_rot90():
    inp = np.zeros((1, 2, INT_OVERFLOW))
    inp[-1, -1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.rot90(inp, axes=(1,2))
        out.backward()
    assert out.shape == (1, INT_OVERFLOW, 2)
    assert out[0, 0, 1] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1, -1] == 1


@use_np
def test_squeeze():
    inp = np.zeros((2, 1, INT_OVERFLOW))
    inp[-1, -1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.squeeze(inp, axis=1)
        out.backward()
    assert out.shape == (2, INT_OVERFLOW)
    assert out[-1, -1] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1, -1] == 1


@use_np
def test_tile():
    inp = np.array([[0, 1],[2, 3]])
    inp.attach_grad()
    with mx.autograd.record():
        out = np.tile(inp, (1, HALF_INT_OVERFLOW))
        out.backward()
    assert out.shape == (2, INT_OVERFLOW)
    assert out[-1, -1] == 3
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == HALF_INT_OVERFLOW


@use_np
def test_trace():
    N = 2**16
    inp = np.eye(N)
    inp.attach_grad()
    with mx.autograd.record():
        out = np.trace(inp)
        out.backward()
    assert out == N
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0] == 1 and inp.grad[-1, -1] == 1
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.trace(inp, offset=INT_OVERFLOW-2)
        out.backward()
    assert out == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, -2] == 1 and inp.grad[-1, -1] == 1


@use_np
def test_tri():
    N = 2**16
    data = np.tri(N)
    assert data.shape == (N, N)
    assert data[0, 0] == 1 and data[-1, -1] == 1
    assert data[0, -1] == 0 and data[-1, 0] == 1
    data = np.tri(2, INT_OVERFLOW, INT_OVERFLOW-2)
    assert data.shape == (2, INT_OVERFLOW)
    assert data[0, -1] == 0 and data[-1, -1] == 1


@use_np
def test_tril():
    N = 2**16
    inp = np.ones((N, N))
    inp.attach_grad()
    with mx.autograd.record():
        out = np.tril(inp)
        out.backward()
    assert out.shape == (N, N)
    assert out[-1, -1] == 1 and out[0, -1] == 0 and out[-1, 0] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1 and inp.grad[0, -1] == 0 and \
        inp.grad[-1, 0] == 1
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.tril(inp, k=INT_OVERFLOW-2)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, -1] == 0 and out[-1, -1] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, -1] == 0 and inp.grad[-1, -1] == 1


@use_np
def test_triu():
    N = 2**16
    inp = np.ones((N, N))
    inp.attach_grad()
    with mx.autograd.record():
        out = np.triu(inp)
        out.backward()
    assert out.shape == (N, N)
    assert out[-1, -1] == 1 and out[0, -1] == 1 and out[-1, 0] == 0
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1 and inp.grad[0, -1] == 1 and \
        inp.grad[-1, 0] == 0
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.triu(inp, k=INT_OVERFLOW-1)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, -1] == 1 and out[-1, -1] == 0
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, -1] == 1 and inp.grad[-1, -1] == 0


@use_np
def test_transpose():
    inp = np.zeros((1, 2, INT_OVERFLOW))
    inp[0, 0, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.transpose(inp, (2, 0, 1))
        out.backward()
    assert out.shape == (INT_OVERFLOW, 1, 2)
    assert out[-1, 0, 0] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1, -1] == 1


@use_np
def test_trunc():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[0, -1], inp[1, -1] = 1.9, -1.9
    inp.attach_grad()
    with mx.autograd.record():
        out = np.trunc(inp)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, -1] == 1 and out[1, -1] == -1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0


@use_np
def test_stack():
    inp1 = np.zeros((INT_OVERFLOW))
    inp2 = np.ones((INT_OVERFLOW))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out1 = np.stack([inp1, inp2])
        out1.backward()
    assert out1.shape == (2, INT_OVERFLOW)
    assert out1[0, -1] == 0 and out1[1, -1] == 1
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1] == 1
    with mx.autograd.record():
        out2 = np.stack([inp1, inp2], axis=1)
        out2.backward()
    assert out2.shape == (INT_OVERFLOW, 2)
    assert out2[-1, 0] == 0 and out2[-1, 1] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == 1


@use_np
def test_dstack():
    inp1 = np.zeros((INT_OVERFLOW, 1))
    inp2 = np.ones((INT_OVERFLOW, 1))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.dstack((inp1, inp2))
        out.backward()
    assert out.shape == (INT_OVERFLOW, 1, 2)
    assert out[0, -1, 0] == 0 and out[0, -1, 1] == 1
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 1


@use_np
def test_hstack():
    inp1 = np.zeros((INT_OVERFLOW, 1))
    inp2 = np.ones((INT_OVERFLOW, 1))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out1 = np.hstack((inp1, inp2))
        out1.backward()
    assert out1.shape == (INT_OVERFLOW, 2)
    assert out1[-1, 0] == 0 and out1[-1, 1] == 1
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 1
    with mx.autograd.record():
        out2 = np.hstack((inp1.flatten(), inp2.flatten()))
        out2.backward()
    assert out2.shape == (DOUBLE_INT_OVERFLOW, )
    assert out2[INT_OVERFLOW-1] == 0 and out2[-1] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1, -1] == 1


'''
                                     _               _
  _ _ _  _ _ __  _ __ _  _   _____ _| |_ ___ _ _  __(_)___ _ _
 | ' \ || | '  \| '_ \ || | / -_) \ /  _/ -_) ' \(_-< / _ \ ' \
 |_||_\_,_|_|_|_| .__/\_, | \___/_\_\\__\___|_||_/__/_\___/_||_|
                |_|   |__/
'''


@use_np
def test_activation():
    A = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.activation(A, act_type='sigmoid')
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 0.5
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert_almost_equal(A.grad[0][0], np.array([0.25]), \
                rtol=1e-3, atol=1e-5)


@use_np
def test_arange_like():
    A = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.arange_like(A)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[100][0] == 200
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0

@use_np
def test_batch_dot():
    inp1 = np.zeros((2, 1, INT_OVERFLOW))
    inp2 = np.zeros((2, INT_OVERFLOW, 1))
    inp1[-1, -1, -1] = 2
    inp2[-1, -1, -1] = 3
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = npx.batch_dot(inp1, inp2)
        out.backward()
    assert out.shape == (2, 1, 1)
    assert out[1, 0, 0] == 6
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1, -1] == 3
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1, -1, -1] == 2

@use_np
def test_cast():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.cast(A, dtype='float16')
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 1


@use_np
def test_broadcast_like():
    A = np.ones((1, 2))
    B = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        C = npx.broadcast_like(A, B)
    assert C.shape == (INT_OVERFLOW, 2)
    assert C[0][0] == 1
    C.backward()
    assert A.grad.shape == (1, 2)
    with mx.autograd.record():
        C = npx.broadcast_like(A.reshape(2, 1), B.T)
    assert C.shape == (2, INT_OVERFLOW)
    assert C[0][0] == 1
    C.backward()
    assert A.grad.shape == (1, 2)
    assert_almost_equal(A.grad[0][0], np.array([INT_OVERFLOW]), \
                            rtol=1e-3, atol=1e-5)


@use_np
def test_constraint_check():
    A = np.ones((2, INT_OVERFLOW))
    constraint = (A > 0)
    B = npx.constraint_check(constraint)
    assert B == True

# broken
@use_np
def test_batch_flatten():
    A = np.ones((2, 1, INT_OVERFLOW))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.batch_flatten(A)
    assert B.shape == (2, INT_OVERFLOW)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (2, 1, INT_OVERFLOW)
    assert A.grad[0][0][0] == 1


@use_np
def test_batch_norm():
    inp = np.zeros((2, INT_OVERFLOW))
    gamma = np.array([1.5, 2.5])
    beta = np.array([0.3, 0.6])
    mov_mean = np.array([0.4, 0.8])
    mov_var = np.array([0.6, 1.2])
    eps = 1e-5
    inp[0, -1], inp[1, -1] = 3, 6
    inp.attach_grad()
    with mx.autograd.record():
        out = npx.batch_norm(inp, gamma=gamma, beta=beta, moving_mean=mov_mean,\
            moving_var=mov_var, axis=0, eps=eps, use_global_stats=True)
        out.backward()
    assert out.shape == inp.shape
    ref0 = (inp[0, -1] - mov_mean[0]) / (mov_var[0] + eps)**0.5 * gamma[0] + beta[0]
    ref1 = (inp[1, -1] - mov_mean[1]) / (mov_var[1] + eps)**0.5 * gamma[1] + beta[1]
    assert_almost_equal(out[0, -1], ref0, rtol=1e-3, atol=1e-5)
    assert_almost_equal(out[1, -1], ref1, rtol=1e-3, atol=1e-5)
    assert inp.grad.shape == inp.shape
    grad_ref0 = gamma[0] / (mov_var[0] + eps)**0.5
    grad_ref1 = gamma[1] / (mov_var[1] + eps)**0.5
    assert_almost_equal(inp.grad[0, -1], grad_ref0, rtol=1e-3, atol=1e-5)
    assert_almost_equal(inp.grad[1, -1], grad_ref1, rtol=1e-3, atol=1e-5)


@use_np
def test_batch_norm_mean_var():
    N = 2**20
    inp = np.zeros((2, INT_OVERFLOW), dtype='float64')
    gamma = np.array([1, 1], dtype='float64')
    beta = np.array([0, 0], dtype='float64')
    mov_mean = np.array([0, 0], dtype='float64')
    mov_var = np.array([1, 1], dtype='float64')
    eps = 0
    inp[1, -1] = N
    with mx.autograd.record():
        out, mean, var = npx.batch_norm(inp, gamma=gamma, beta=beta, moving_mean=mov_mean,\
            moving_var=mov_var, axis=0, eps=eps, output_mean_var=True)
    assert out.shape == inp.shape
    mean_ref = float(N) / INT_OVERFLOW
    std_ref = ((INT_OVERFLOW-1) * (mean_ref-0)**2 + (mean_ref-N)**2) / INT_OVERFLOW
    out_ref = (N - mean_ref) / (std_ref**0.5)
    assert_almost_equal(mean[1], mean_ref, rtol=1e-3, atol=1e-5)
    assert_almost_equal(var[1], 1 / std_ref**0.5, rtol=1e-3, atol=1e-5)
    assert_almost_equal(out[1, -1], out_ref, rtol=1e-3, atol=1e-5)


@use_np
def test_nonzero():
    A = np.zeros((2, INT_OVERFLOW))
    A[0, 1] = 1
    A[0, -2] = 1
    A.attach_grad()
    with mx.autograd.record():
        B = npx.nonzero(A)
    assert B.shape == (2, 2)
    assert B[0, 0] == 0 and B[0, 1] == 1
    assert B[1, 0] == 0 and B[1, 1] == int(INT_OVERFLOW - 2)
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert A.grad[0][0] == 0


@use_np
def test_one_hot():
    A = np.zeros((INT_OVERFLOW))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.one_hot(A, 2)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, )
    assert A.grad[0] == 0


@use_np
def test_pick():
    INT_OVERFLOW = 2**31
    A = np.zeros((INT_OVERFLOW, 2))
    B = np.zeros((INT_OVERFLOW))
    A[-1, 0] = 3
    A.attach_grad()
    B.attach_grad()
    with mx.autograd.record():
        C = npx.pick(A, B)
    assert C.shape == (INT_OVERFLOW, )
    assert C[0] == 0
    assert C[-1] == 3
    C.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert B.grad.shape == (INT_OVERFLOW, )
    assert A.grad[0][0] == 1


@use_np
def test_scalar_poisson():
    A = npx.scalar_poisson(lam=4, shape=(2, INT_OVERFLOW))
    assert A.shape == (2, INT_OVERFLOW)


@use_np
def test_tensor_poisson():
    lam = np.array([2.0, 4.0])
    A = npx.tensor_poisson(lam, shape=(INT_OVERFLOW))
    assert A.shape == (2, INT_OVERFLOW)


@use_np
def test_reshape():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad() 
    with mx.autograd.record():
        B = npx.reshape(A, (-5))
    assert B.shape == (DOUBLE_INT_OVERFLOW, )
    assert B[0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 1


@use_np
def test_reshape_like():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.reshape_like(A, np.zeros((2, INT_OVERFLOW)))
    assert B.shape == (2, INT_OVERFLOW)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 1


@use_np
def test_sigmoid():
    A = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.sigmoid(A)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 0.5
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert_almost_equal(A.grad[0][0], np.array([0.25]), \
                rtol=1e-3, atol=1e-5)


@use_np
def test_shape_array():
    A = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.shape_array(A)
    assert B[0] == INT_OVERFLOW and B[1] == 2
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0


@use_np
def test_stop_gradient():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.stop_gradient(A * 3)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 3
    B.backward()
    # should be 3 if not for stop_gradient()
    assert A.grad[0][0] == 0


@use_np
def test_sequence_mask():
    A = np.ones((2, 2, INT_OVERFLOW))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.sequence_mask(A, sequence_length=np.array([1,1]), \
                use_sequence_length=True)
    assert B.shape == (2, 2, INT_OVERFLOW)
    assert B[0][0][0] == 1
    assert B[1][0][0] == 0
    B.backward()
    assert A.grad.shape == (2, 2, INT_OVERFLOW)
    assert A.grad[0][0][0] == 1


@use_np
def test_topk():
    A = np.ones((2, INT_OVERFLOW))
    A[0][100] = 2
    A[1][200] = 2
    A.attach_grad()
    with mx.autograd.record():
        B = npx.topk(A, k=2)
    assert B.shape == (2, 2)
    assert B[0][0] == 100 and B[1][0] == 200
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert A.grad[0][0] == 0


@use_np
def test_slice():
    A = np.ones((INT_OVERFLOW, 3))
    A[100][1] = 2
    B = npx.slice(A, begin=(100,1), end=(200,3))
    assert B.shape == (100, 2)
    assert B[0][0] == 2


@use_np
def test_smooth_l1():
    A = np.arange((INT_OVERFLOW))
    A.attach_grad() 
    with mx.autograd.record():
        B = npx.smooth_l1(A)
    assert B.shape == (INT_OVERFLOW, )
    assert B[1] == 0.5
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, )
    assert A.grad[0] == 0


@use_np
def test_gamma():
    A = np.ones((2, INT_OVERFLOW))
    A[0][0] = 5
    A.attach_grad()
    with mx.autograd.record():
        B = npx.gamma(A)
    assert B.shape == (2, INT_OVERFLOW)
    assert B[0][0] == 24
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert_almost_equal(A.grad[0][0], np.array([36.1428]), \
                rtol=1e-3, atol=1e-5)


@use_np
def test_gammaln():
    A = np.ones((2, INT_OVERFLOW))
    A[0][0] = 5
    A.attach_grad() 
    with mx.autograd.record():
        B = npx.gammaln(A)
    assert B.shape == (2, INT_OVERFLOW)
    assert_almost_equal(B[0][0], np.array([np.log(24)]), \
                rtol=1e-3, atol=1e-5)
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert_almost_equal(A.grad[0][0], np.array([1.5061178]), \
                rtol=1e-3, atol=1e-5)


@use_np
def test_digamma():
    A = np.ones((2, INT_OVERFLOW)) 
    A[0][0] = 5
    A.attach_grad()  
    with mx.autograd.record():
        B = npx.digamma(A)
    assert B.shape == (2, INT_OVERFLOW)
    assert_almost_equal(B[0][0], np.array([1.5061178]), \
                rtol=1e-3, atol=1e-5)
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert_almost_equal(A.grad[0][0], np.array([0.22132295]), \
                rtol=1e-3, atol=1e-5)


@use_np
@pytest.mark.skip(reason='to be moved to new file and run separately as it takes lot of memory')
def test_rnn_dim_check():
    L_SEQ, BAT, L_INP, L_STA = 2**31, 4, 2**10, 2
    data = np.random.uniform(-1, 1, (L_SEQ, BAT, L_INP))
    state = np.random.normal(0, 1, (1, BAT, L_STA))
    params = np.random.normal(0, 1, (2056,))
    assertRaises(ValueError, npx.rnn, data=data, parameters=params, \
        mode='rnn_relu', state=state, state_size=L_STA, num_layers=1)


@use_np
@pytest.mark.skip(reason='runs without MKLDNN, wtih is not default behavior')
def test_rnn_vanilla():
    L_SEQ, BAT, L_INP, L_STA = 2**20, 4, 2**10, 2
    def batch_check(x, modes, params):
        state = np.random.normal(0, 1, (1, BAT, L_STA))
        for m, p in zip(modes, params):
            x.attach_grad()
            with mx.autograd.record():
                y = npx.rnn(data=x, parameters=p, mode=m, \
                    state=state, state_size=L_STA, num_layers=1)
            assert y.shape == (L_SEQ, BAT, L_STA)
            y.backward()
            npx.waitall()
    data = np.random.uniform(-1, 1, (L_SEQ, BAT, L_INP))
    modes = ['rnn_tanh', 'rnn_relu']
    params = [np.random.normal(0, 1, (2056,)), \
        np.random.normal(0, 1, (2056,))]
    batch_check(data, modes, params)


@use_np
def test_rnn_gru():
    L_SEQ, BAT, L_INP, L_STA = 2**20, 4, 2**10, 2
    data = np.random.uniform(-1, 1, (L_SEQ, BAT, L_INP))
    state = np.random.normal(0, 1, (1, BAT, L_STA))
    params = np.random.normal(0, 1, (6168,))
    data.attach_grad()
    with mx.autograd.record():
        out = npx.rnn(data=data, parameters=params, mode='gru', \
            state=state, state_size=L_STA, num_layers=1)
    assert out.shape == (L_SEQ, BAT, L_STA)
    out.backward()
    npx.waitall()


@use_np
def test_rnn_lstm():
    L_SEQ, BAT, L_INP, L_STA= 2**20, 4, 2**10, 2
    data = np.random.uniform(-1, 1, (L_SEQ, BAT, L_INP))
    state = np.random.normal(0, 1, (1, BAT, L_STA))
    state_cell = np.random.normal(0, 1, (1, BAT, L_STA))
    params = np.random.normal(0, 1, (8224,))
    data.attach_grad()
    with mx.autograd.record():
        out = npx.rnn(data=data, parameters=params, mode='lstm', \
            state=state, state_size=L_STA, state_cell=state_cell, num_layers=1)
    assert out.shape == (L_SEQ, BAT, L_STA)
    out.backward()
    npx.waitall()



@use_np
def test_ctc_loss():
    def test_ctc_loss_size_check(A, label):
        assertRaises(ValueError, npx.ctc_loss, A, label)
    
    L_SEQ, L_ALP, L_LAB, BAT = 2**10, 2**20, 2**6, 2
    A = np.zeros((L_SEQ, BAT, L_ALP))
    label = np.random.randint(0, L_ALP, (BAT, L_LAB))
    # test for expected exception
    test_ctc_loss_size_check(A, label)
    # now we shrink the size a little bit and test for an allowed case
    L_ALP = 2**20 - 1
    A = np.zeros((L_SEQ, BAT, L_ALP))
    label = np.random.randint(0, L_ALP, (BAT, L_LAB))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.ctc_loss(A, label)
    assert B.shape == (BAT, )
    assert type(B[0]).__name__ == 'ndarray'
    B.backward()
    assert A.grad.shape == (L_SEQ, BAT, L_ALP)
    assert type(A[0]).__name__ == 'ndarray'


@use_np
def test_erf():
    A = np.ones((2, INT_OVERFLOW))
    A[0][0] = 10
    A.attach_grad()
    with mx.autograd.record():
        B = npx.erf(A)
    assert B.shape == (2, INT_OVERFLOW)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert_almost_equal(A.grad[0][0], np.array([4.2e-44]), \
                rtol=1e-3, atol=1e-5)


@use_np
def test_erfinv():
    A = np.ones((2, INT_OVERFLOW))
    A[0][0] = 0.5
    A.attach_grad()
    with mx.autograd.record():
        B = npx.erfinv(A)
    assert B.shape == (2, INT_OVERFLOW)
    assert_almost_equal(B[0][0], np.array([0.47693628]), \
                rtol=1e-3, atol=1e-5)
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert_almost_equal(A.grad[0][0], np.array([1.112585]), \
                rtol=1e-3, atol=1e-5)


@use_np
def test_index_add():
    A = np.zeros((2, INT_OVERFLOW))
    ind = np.array([[0, 0], [0, 1]], dtype='int32')
    val = np.array([100, 200])
    A.attach_grad() 
    with mx.autograd.record():
        B = npx.index_add(A, ind, val)
    assert B.shape == (2, INT_OVERFLOW)
    assert B[0][0] == 100 and B[0][1] == 200
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert A.grad[0][0] == 1


@use_np
def test_index_update():
    A = np.zeros((2, INT_OVERFLOW))
    ind = np.array([[0, 0], [0, 1]], dtype='int32')
    val = np.array([100, 200])
    A.attach_grad() 
    with mx.autograd.record():
        B = npx.index_update(A, ind, val) 
    assert B.shape == (2, INT_OVERFLOW) 
    assert B[0][0] == 100 and B[0][1] == 200
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert A.grad[0][0] == 0


@use_np
def test_layer_norm():
    A = np.ones((2, INT_OVERFLOW))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.layer_norm(A, gamma=np.ones((2)), beta=np.zeros((2)), axis=0)
    assert B.shape == (2, INT_OVERFLOW) 
    assert B[0][0] == 0
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert A.grad[0][0] == 0


@use_np
def test_dlpack():
    A = np.ones((2, INT_OVERFLOW))
    A[0][100] = 100
    B = npx.to_dlpack_for_read(A)
    assert type(B).__name__ == 'PyCapsule'
    C = npx.from_dlpack(B)
    assert type(C).__name__ == 'ndarray'
    assert C.shape == (2, INT_OVERFLOW)
    assert C[0][100] == 100
    B = npx.to_dlpack_for_write(A)
    assert type(B).__name__ == 'PyCapsule'
    C = npx.from_dlpack(B)
    C += 1
    assert type(C).__name__ == 'ndarray'
    assert C.shape == (2, INT_OVERFLOW)
    assert C[0][100] == 101


@use_np
def test_pooling():
    def test_pooling_large_dim():
        A = np.ones((1, 1, INT_OVERFLOW))
        assertRaises(MXNetError, npx.pooling, data=A, kernel=(2), stride=(2), \
                pool_type='max')
    
    test_pooling_large_dim()
    D, H, W = 2**12, 2**10, 2**10
    A = np.ones((1, 1, D, H ,W))
    A[0, 0, 0, 0, 2] = 100
    A.attach_grad()
    with mx.autograd.record():
        B = npx.pooling(data=A, kernel=(2, 2, 2), stride=(2, 2, 2), \
                pool_type='max')
    assert B.shape == (1, 1, int(D/2), int(H/2), int(W/2))
    assert B[0, 0, 0, 0, 1] == 100
    B.backward()
    assert A.grad.shape == (1, 1, D, H, W)
    assert A.grad[0, 0, 0, 0, 0] == 1


@use_np
def test_roi_pooling():
    def test_roi_pooling_large_dim():
        A = np.ones((1, 1, INT_OVERFLOW, 5))
        roi = np.array([[0, 0, 0, 5, 5]])
        assertRaises(MXNetError, npx.roi_pooling, A, roi, pooled_size=(3, 3), \
            spatial_scale=1)
    
    test_roi_pooling_large_dim()
    H, W = 2**16, 2**16
    A = np.ones((1, 1, H, W))
    A[0, 0, 0, 2] = 100
    roi = np.array([[0, 0, 0, 5, 5]])
    A.attach_grad()
    with mx.autograd.record():
        B = npx.roi_pooling(A, roi, pooled_size=(3, 3), spatial_scale=1)
    assert B.shape == (1, 1, 3, 3)
    assert B[0][0][0][1] == 100
    B.backward()
    assert A.grad.shape == (1, 1, H, W)
    assert A.grad[0][0][0][0] == 1


@use_np
@pytest.mark.skip(reason='times out on (generally speaking) large tensors')
def test_save_load():
    A = np.ones((2, INT_OVERFLOW), dtype='int8')
    A[0][100] = 100
    npx.save('my_tensor', A)
    B = np.array(npx.load('my_tensor'))
    assert B[0].shape == (2, INT_OVERFLOW)
    assert B[0][0][100] == 100


@use_np
def test_gather_nd():
    A = np.ones((1, 2, INT_OVERFLOW))
    A [0, 1, 100] = 100
    A.attach_grad()
    with mx.autograd.record():
        B = npx.gather_nd(data=A, \
            indices=np.array([[0, 0] , [0, 1], [INT_OVERFLOW-1, 100]], \
            dtype='int64'))
    assert B.shape == (2, )
    assert B[0] == 1 and B[1] == 100
    B.backward()
    assert A.grad.shape == (1, 2, INT_OVERFLOW)
    assert A.grad[0, 0, 0] == 0
    assert A.grad[0, 0, INT_OVERFLOW-1] == 1


@use_np
def test_random_bernoulli():
    prob = np.zeros((INT_OVERFLOW))
    prob[0] = 1
    A = npx.random.bernoulli(prob=prob, size=(INT_OVERFLOW))
    assert A.shape == (INT_OVERFLOW, )
    assert A[0] == 1
    assert A[1] == 0


@use_np
def test_cumsum():
    input = np.ones((INT_OVERFLOW, 3), dtype='int64')
    input.attach_grad()
    with mx.autograd.record():
        output = np.cumsum(input, axis=0, dtype='int64')
        output.backward()
    assert output.shape == input.shape
    assert output[-1, -1] == INT_OVERFLOW
    assert input.grad.shape == input.shape
    assert input.grad[0, 0] == INT_OVERFLOW
    assert input.grad[-1, -1] == 1


@use_np
def test_round():
    input = np.ones((INT_OVERFLOW, 2))
    input[INT_OVERFLOW-1][0] = 1.6
    output = np.round(input)
    assert output.shape == (INT_OVERFLOW, 2)
    assert output[-1][0] == 2


@use_np
def test_cross():
    inp = np.ones((INT_OVERFLOW, 3))
    inp2 = np.ones((INT_OVERFLOW, 2))
    inp[-1] = np.array([1, 2, 3])
    inp2[-1] = np.array([4, 5])
    inp.attach_grad()
    with mx.autograd.record():
        out = np.cross(inp, inp2)
        out.backward()
    assert out.shape == (INT_OVERFLOW, 3)
    assert out[0, 0] == -1 and out[0, 1] == 1 and out[0, 2] == 0
    assert out[-1, 0] == -15 and out[-1, 1] == 12 and out[-1, 2] == -3
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0] == 1 and inp.grad[0, 1] == -1 and inp.grad[0, 2] == 0
    assert inp.grad[-1, 0] == 5 and inp.grad[-1, 1] == -4 and inp.grad[-1, 2] == -1


@use_np
def test_array_split():
    inp = np.ones((INT_OVERFLOW, 2))
    inp[0][0] = 0
    inp[-1][-1] = 2
    out = np.array_split(inp, 2)
    assert out[0].shape ==(HALF_INT_OVERFLOW, 2)
    assert out[1].shape ==(HALF_INT_OVERFLOW, 2)
    assert out[0][0][0] == 0
    assert out[1][-1][-1] == 2


@use_np
def test_take():
    inp = np.zeros((INT_OVERFLOW, 2))
    inp[0], inp[-1] = 1, 2
    indices = np.array([[0],[INT_OVERFLOW-1]], dtype='int64')
    inp.attach_grad()
    indices.attach_grad()
    with mx.autograd.record():
        out = np.take(inp, indices, axis=0)
        out.backward()
    assert out.shape == (2, 1, 2)
    assert out[0, 0, 0] == 1 and out[1, 0, 0] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0] == 1 and inp.grad[-1, 0] == 1
    assert indices.grad.shape == indices.shape
    assert indices[0, 0] == 0


@use_np
def test_std():
    N = 2*20
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = N
    inp.attach_grad()
    with mx.autograd.record():
        out = np.std(inp, axis=1)
        out.backward()
    assert out.shape == (2, )
    ref = ((float(N)/INT_OVERFLOW)**2 * (INT_OVERFLOW-1))**0.5
    assert_almost_equal(out[1], ref, rtol=1e-5, atol=1e-5)
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0


@use_np
def test_var():
    N = 2*20
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = N
    inp.attach_grad()
    with mx.autograd.record():
        out = np.var(inp, axis=1)
        out.backward()
    assert out.shape == (2, )
    ref = (float(N)/INT_OVERFLOW)**2 * (INT_OVERFLOW-1)
    assert_almost_equal(out[1], ref, rtol=1e-5, atol=1e-5)

    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0

@use_np
def test_rollaxis():
    inp = np.zeros((1, 1, 2, INT_OVERFLOW, 1))
    inp[-1, -1, -1, -1, -1] = 1
    inp.attach_grad()
    with mx.autograd.record():
        out = np.rollaxis(inp, 3)
        out.backward()
    assert out.shape == (INT_OVERFLOW, 1, 1, 2, 1)
    assert out[-1, -1, -1, -1, -1] == 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1, -1, -1, -1] == 1


@use_np
def test_vstack():
    inp1 = np.zeros((INT_OVERFLOW, 1))
    inp2 = np.ones((INT_OVERFLOW, 1))
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out1 = np.vstack((inp1, inp2))
        out1.backward()
    assert out1.shape == (DOUBLE_INT_OVERFLOW, 1)
    assert out1[INT_OVERFLOW-1, 0] == 0 and out1[-1, 0] == 1
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 1
    with mx.autograd.record():
        out2 = np.vstack((inp1.flatten(), inp2.flatten()))
        out2.backward()
    assert out2.shape == (2, INT_OVERFLOW)
    assert out2[0, -1] == 0 and out2[1, -1] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1, -1] == 1


@use_np
def test_ediff1d():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[0, -1], inp[1, 0] = 1, 3
    inp.attach_grad()
    with mx.autograd.record():
        out = np.ediff1d(inp, to_begin=-99, to_end=np.array([88, 99]))
        out.backward()
    assert out.shape == (2 * INT_OVERFLOW - 1 + 1 + 2, )
    assert out[INT_OVERFLOW-1] == 1 and out[INT_OVERFLOW] == 2 and\
            out[INT_OVERFLOW+1] == -3
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, 0] == -1 and inp.grad[-1, -1] == 1


@use_np
def test_split():
    inp = np.ones((INT_OVERFLOW, 2))
    inp[INT_OVERFLOW // 2] = 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.split(inp, 2, axis = 0)
        out[1].backward()
    assert out[0].shape == (INT_OVERFLOW // 2, 2)
    assert out[1].shape == (INT_OVERFLOW // 2, 2)
    assert out[0][0, 0] == 1
    assert out[1][0, 0] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[0][0] == 0 and inp.grad[-1][-1] == 1


@use_np
def test_hsplit():
    inp = create_2d_np_tensor(rows=INT_OVERFLOW, columns=4)
    inp.attach_grad()
    with mx.autograd.record():
        out = np.hsplit(inp, 2)
        out[1].backward()
    assert out[1].shape == (INT_OVERFLOW, 2)
    assert out[0].shape == (INT_OVERFLOW, 2)
    assert out[1][-1][0] == INT_OVERFLOW-1
    assert out[0][-1][1] == INT_OVERFLOW-1
    assert inp.grad.shape == inp.shape
    assert inp.grad[0][0] == 0 and inp.grad[-1][-1] == 1


@use_np
def test_vsplit():
    inp = create_2d_np_tensor(rows=INT_OVERFLOW, columns=4)
    inp.attach_grad()
    with mx.autograd.record():
        out = np.vsplit(inp, 2)
        out[1].backward()
    assert out[1].shape == (INT_OVERFLOW//2, 4)
    assert out[0].shape == (INT_OVERFLOW//2, 4)
    assert out[0][-1][0] == INT_OVERFLOW // 2 -1
    assert out[1][-1][1] == INT_OVERFLOW - 1
    assert inp.grad.shape == inp.shape
    assert inp.grad[INT_OVERFLOW//2 - 1][-1] == 0 and inp.grad[-1][-1] == 1


@use_np
def test_dsplit():
    inp = np.arange(INT_OVERFLOW, dtype=np.int64).reshape(INT_OVERFLOW, 1, 1)
    inp = np.broadcast_to(inp, shape=(inp.shape[0], 2, 2))
    inp.attach_grad()
    with mx.autograd.record():
        out = np.dsplit(inp, 2)
        out[1].backward()
    assert out[1].shape == (INT_OVERFLOW, 2, 1)
    assert out[0].shape == (INT_OVERFLOW, 2, 1)
    assert out[0][-1][0][0] == INT_OVERFLOW - 1
    assert out[1][0][1][0] == 0
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1][-1][0] == 0 and inp.grad[0][1][1] == 1


@use_np
def test_unique():
    inp = np.zeros((2, HALF_INT_OVERFLOW))
    assertRaises(ValueError, np.unique, inp, axis=1)


@use_np
def test_repeat():
    inp = np.ones((2, HALF_INT_OVERFLOW))
    assertRaises(ValueError, np.repeat, inp, repeats=2, axis=1)


@use_np
def test_indices():
    assertRaises(ValueError, np.indices, (2, HALF_INT_OVERFLOW))


@use_np    
def test_tril_indices():
    N = 2**16
    data = np.tril_indices(N, -1)
    assert data[0].shape == (((1 + (N-1)) * (N-1) / 2), )
    assert data[0][-1] == N - 1 and data[1][-1] == N - 2


@use_np
def test_tril_indices_extreme():
    data = np.tril_indices(n=2, m=INT_OVERFLOW+2, k=INT_OVERFLOW)
    assert data[0].shape == (INT_OVERFLOW + 1 + INT_OVERFLOW + 2, )
    assert data[0][-1] == 1 and data[1][-1] == INT_OVERFLOW + 1
    assert data[0][INT_OVERFLOW] == 0 and data[1][INT_OVERFLOW] == INT_OVERFLOW


@use_np
def test_diff():
    inp = np.zeros((2, INT_OVERFLOW+1))
    inp[-1, -1] = 100
    inp.attach_grad()
    with mx.autograd.record():
        out1 = np.diff(inp)
        out1.backward()
    assert out1.shape == (2, INT_OVERFLOW)
    assert out1[-1, -1] == 100
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 1
    with mx.autograd.record():
        out2 = np.diff(inp, axis=0)
        out2.backward()
    assert out2.shape == (1, INT_OVERFLOW+1)
    assert out2[-1, -1] == 100
    assert inp.grad.shape == inp.shape
    assert inp.grad[1, -1] == 1, inp.grad[0, -1] == 1


@use_np
def test_kron():
    # tensor tensor case
    inp1 = np.array([5, 10], dtype="float64")
    inp2 = np.ones((INT_OVERFLOW), dtype = 'float64')
    inp2[-1] = 3
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out1 = np.kron(inp1, inp2)
        out1.backward()
    assert out1.shape == (DOUBLE_INT_OVERFLOW, )
    assert out1[INT_OVERFLOW-1] == 15 and out1[-1] == 30
    assert inp1.grad.shape == inp1.shape and inp2.grad.shape == inp2.shape
    assert inp1.grad[0] == INT_OVERFLOW + 2
    assert inp2.grad[-1] == 15
    # scalar tensor case
    inp3 = np.array([3], dtype='float64')
    inp3.attach_grad()
    with mx.autograd.record():
        out2 = np.kron(inp3, inp2)
        out2.backward()
    assert out2.shape == (INT_OVERFLOW, )
    assert out2[INT_OVERFLOW-1] == 9
    assert inp3.grad.shape == inp3.shape and inp2.grad.shape == inp2.shape
    assert inp3.grad[0] == INT_OVERFLOW + 2
    assert inp2.grad[-1] == 3


@use_np
def test_logspace():
    data = np.logspace(1.0, 10.0, INT_OVERFLOW)
    assert data.shape == (INT_OVERFLOW, )
    assert data[0] == 10 and data[-1] == 10000000000
    assert_almost_equal(data[HALF_INT_OVERFLOW], np.array(10**5.5), \
                rtol=1e-3, atol=1e-5)

@use_np
def test_linspace():
    data = np.linspace(0, 1000, INT_OVERFLOW)
    assert data.shape == (INT_OVERFLOW, )
    assert data[0] == 0 and data[-1] == 1000
    assert data[HALF_INT_OVERFLOW] == 500


@use_np
def test_histogram():
    inp = np.ones((INT_OVERFLOW, 2))
    inp[-1, -1] = 2
    hist, _ = np.histogram(inp, np.array([0.5, 1.5, 2.5, 3.5]))
    assert hist.shape == (3, )
    assert hist[0] == int(2 * INT_OVERFLOW - 1) and hist[1] == 1


@use_np
def test_nan_to_num():
    inp = np.zeros((3, INT_OVERFLOW))
    inp[:, -1] = np.array([np.nan, np.inf, -np.inf])
    inp.attach_grad()
    with mx.autograd.record():
        out = np.nan_to_num(inp, nan=0, posinf=1, neginf=-1)
        out.backward()
    assert out.shape == inp.shape
    assert out[0, -1] == 0 and out[1, -1] == 1 and out[2, -1] == -1
    assert inp.grad.shape == inp.shape
    assert inp.grad[0, -1] == 0 and inp.grad[1, -1] == 0
    assert inp.grad[0, 0] == 1 and inp.grad[2, -1] == 0


@use_np
def test_interp():
    xp = np.array([1, 2, 3])
    fp = np.array([3, 2, 1])
    inp = np.ones((2, INT_OVERFLOW))
    inp[-1, -1] = 2.5
    inp.attach_grad()
    with mx.autograd.record():
        B = np.interp(inp, xp, fp)
        B.backward()
    assert B.shape == inp.shape
    assert B[-1, -1] == 1.5
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1] == 0


@use_np
@pytest.mark.skip(reason='times out (20 mins)')
def test_edge_padding():
    inp = create_2d_np_tensor(rows=INT_OVERFLOW, columns=4, dtype=np.int64)
    out = np.pad(inp, ((1, 1), (1, 1)), "edge")
    assert out[0][0] == 0
    assert out[-1][-1] == INT_OVERFLOW - 1
    assert out.shape == (INT_OVERFLOW + 2, 4 + 2)


@use_np
def test_constant_padding():
    inp = create_2d_np_tensor(rows=INT_OVERFLOW, columns=4, dtype=np.int64)
    out = np.pad(inp, ((1, 1), (1, 1)), "constant")
    assert out[0][0] == 0
    assert out[-1][-1] == 0
    assert out.shape == (INT_OVERFLOW + 2, 4 + 2)


@use_np
@pytest.mark.skip(reason='times out (20 mins)')
def test_minimum_padding():
    inp = create_2d_np_tensor(rows=INT_OVERFLOW, columns=4, dtype=np.int64)
    out = np.pad(inp, ((1, 1), (1, 1)), "minimum")
    assert out[0][-1] == 0
    assert out[-1][-1] == 0
    assert out.shape == (INT_OVERFLOW + 2, 4 + 2)


@use_np
@pytest.mark.skip(reason='times out (20 mins)')
def test_reflection_padding():
    inp = create_2d_np_tensor(rows=INT_OVERFLOW, columns=4, dtype=np.int64)
    out = np.pad(inp, ((1, 1), (1, 1)), "reflect")
    assert out[0][-1] == 0 + 1
    assert out[-1][0] == INT_OVERFLOW - 1 - 1
    assert out.shape == (INT_OVERFLOW + 2, 4 + 2)


@use_np
@pytest.mark.skip(reason='times out (20 mins)')
def test_symmetric_padding():
    inp = create_2d_np_tensor(rows=INT_OVERFLOW, columns=4, dtype=np.int64)
    out = np.pad(inp, ((1, 1), (1, 1)), "symmetric")
    assert out[0][0] == 0
    assert out[-1][-1] == INT_OVERFLOW - 1
    assert out.shape == (INT_OVERFLOW + 2, 4 + 2)


@use_np
def test_fill_diagonal():
    # test 2d square matrix case
    N = 2**16
    data1 = np.zeros((N, N))
    np.fill_diagonal(data1, [1, 2, 3, 4])
    assert data1[0, 0] == 1 and data1[-1, -1] == 4
    # test 2d long matrix case with wrap
    data2 = np.zeros((INT_OVERFLOW, 2))
    np.fill_diagonal(data2, [1, 2], wrap=True)
    assert data2[0, 0] == 1 and data2[-1, -1] == 2


@use_np
def test_insert():
    inp = np.zeros((INT_OVERFLOW, 2))
    inp2 = np.ones((INT_OVERFLOW))
    inp2[-1] = 2
    inp3 = inp.flatten()
    out = np.insert(inp, 1, inp2, axis=1)
    out2 = np.insert(inp3, slice(1, 2), np.array([5, 6]))
    assert out.shape == (INT_OVERFLOW, 3)
    assert out2.shape == (INT_OVERFLOW * 2 + 2,)
    assert out[0, 1] == 1 and out[-1, 1] == 2
    assert out2[1] == 5 and out2[2] == 6
    assertRaises(MXNetError, np.insert, arr=inp3, obj=np.array([2, 2], dtype=np.int64), values=np.array([5, 6]))


@use_np
def test_moveaxis():
    inp = np.zeros((2, 1, INT_OVERFLOW))
    inp[0, 0, -1], inp[1, 0, -1] = 1, 2
    inp.attach_grad()
    with mx.autograd.record():
        out = np.moveaxis(inp, 2, 0)
        out.backward()
    assert out.shape == (INT_OVERFLOW, 2, 1)
    assert out[-1, 0, 0] == 1 and out[-1, 1, 0] == 2
    assert inp.grad.shape == inp.shape
    assert inp.grad[-1, -1, -1] == 1


@use_np
def test_newaxis():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[-1, -1] = 1
    out1 = inp[np.newaxis, :, :]
    assert out1.shape == (1, 2, INT_OVERFLOW)
    assert out1[0, -1, -1] == 1
    out1 = out1[:, :, :, np.newaxis]
    assert out1.shape == (1, 2, INT_OVERFLOW, 1)
    assert out1[0, -1, -1, 0] == 1


@use_np
def test_triu_indices():
    N = 2**16
    data = np.triu_indices(N, 1)
    assert data[0].shape == (((1 + (N-1)) * (N-1) / 2), )
    assert data[0][-1] == N - 2 and data[1][-1] == N - 1


@use_np
def test_triu_indices_from():
    N = 2**16
    arr = np.zeros((N, N))
    data = np.triu_indices_from(arr, 1)
    assert data[0].shape == (((1 + (N-1)) * (N-1) / 2), )
    assert data[0][-1] == N - 2 and data[1][-1] == N - 1


@use_np
def test_empty():
    data = np.empty((2, INT_OVERFLOW), dtype='float64')
    data = data + 1
    assert data.shape == (2, INT_OVERFLOW)
    assert data[-1, -1] == 1


@use_np
def test_shape_reshape():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[0, -1] = 1
    assert np.shape(inp) == (2, INT_OVERFLOW)
    out = np.reshape(inp, (INT_OVERFLOW, 2))
    assert np.shape(inp) == (2, INT_OVERFLOW)
    assert np.shape(out) == (INT_OVERFLOW, 2)
    assert out[HALF_INT_OVERFLOW-1, 1] == 1


@use_np
def test_copy():
    inp = np.zeros((2, INT_OVERFLOW))
    inp[1, -1] = 2
    out = np.copy(inp)
    out[0, -1] = 3
    assert out.shape == inp.shape
    assert inp[0, -1] == 0 and inp[1, -1] == 2
    assert out[0, -1] == 3 and inp[1, -1] == 2


@use_np
def test_broadcast_arrays():
    inp1 = np.ones((INT_OVERFLOW))
    inp1[-1] = 2
    inp2 = np.array([[3], [4]])
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.broadcast_arrays(inp1, inp2)
        out[0].backward()
        out[1].backward()
    assert out[0].shape == (2, INT_OVERFLOW)
    assert out[0][-1, -1] == 2
    assert out[1].shape == (2, INT_OVERFLOW)
    assert out[1][0, -1] == 3 and out[1][1, -1] == 4
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1] == 2
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1, -1] == INT_OVERFLOW


@use_np
def test_inner():
    inp1 = np.ones((INT_OVERFLOW))
    inp2 = np.zeros((INT_OVERFLOW))
    inp2[-1] = 3
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.inner(inp1, inp2)
        out.backward()
    assert out.shape == ()
    assert out == 3
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1] == 3
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1] == 1


@use_np
def test_matmul():
    inp1 = np.ones((1, 2, INT_OVERFLOW), dtype='float64')
    inp2 = np.ones((INT_OVERFLOW, 1), dtype='float64')
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.matmul(inp1, inp2)
        out.backward()
    assert out.shape == (1, 2, 1)
    assert out[0, 0, 0] == INT_OVERFLOW
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1, -1] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1, -1] == 2


@use_np
def test_outer():
    inp1 = np.ones((INT_OVERFLOW), dtype='float64')
    inp1[-1] = 2
    inp2 = np.ones((2), dtype='float64')
    inp2[-1] = 3
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.outer(inp1, inp2)
        out.backward()
    assert out.shape == (INT_OVERFLOW, 2)
    assert out[-1, 0] == 2 and out[0, -1] == 3 and out[-1, -1] == 6
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[0] == 2 + 3 - 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[0] == INT_OVERFLOW + 2 - 1


@use_np
def test_tensordot():
    inp1 = np.ones((1, INT_OVERFLOW, 2), dtype='float64')
    inp1[0, -1, 1] = 2
    inp2 = np.ones((INT_OVERFLOW, 1, 1), dtype='float64')
    inp2[-1, 0, 0] = 3
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.tensordot(inp1, inp2, axes=[[0, 1], [1, 0]])
        out.backward()
    assert out.shape == (2, 1)
    assert out[0] == INT_OVERFLOW + 3 - 1 and out[1] == INT_OVERFLOW + 6 - 1
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1, -1] == 3 and inp1.grad[0, 0, 0] == 1
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1, -1, -1] == 3 and inp2.grad[0, 0, 0] == 2


@use_np
def test_vdot():
    inp1 = np.ones((2, INT_OVERFLOW))
    inp2 = np.zeros((INT_OVERFLOW, 2))
    inp1[0, -1] = 2
    inp2[HALF_INT_OVERFLOW-1, 1] = 3
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.vdot(inp1, inp2)
        out.backward()
    assert out.shape == ()
    assert out == 6
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[0, -1] == 3 and inp1.grad[-1, -1] == 0
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[HALF_INT_OVERFLOW-1, 1] == 2 and inp2.grad[-1, -1] == 1


@use_np
def test_dot():
    inp1 = np.zeros((1, INT_OVERFLOW))
    inp2 = np.zeros((INT_OVERFLOW, 1))
    inp1[-1, -1] = 2
    inp2[-1, -1] = 3
    inp1.attach_grad()
    inp2.attach_grad()
    with mx.autograd.record():
        out = np.dot(inp1, inp2)
        out.backward()
    assert out.shape == (1, 1)
    assert out[0, 0] == 6
    assert inp1.grad.shape == inp1.shape
    assert inp1.grad[-1, -1] == 3
    assert inp2.grad.shape == inp2.shape
    assert inp2.grad[-1, -1] == 2


@use_np
def test_convolution():
    dim = 2
    batch_size = 1
    channel = 3
    height = SMALL_Y
    width = LARGE_X // 3
    num_filter = 4
    kernel = (3,) * dim   # => shape = (3, 3)

    inp=mx.np.ones(shape=(batch_size, channel, height, width))
    weight = mx.np.ones(shape=(num_filter, channel, kernel[0], kernel[1]))
    bias = mx.np.array(num_filter,)
    inp.attach_grad()
    with mx.autograd.record():
        out = mx.npx.convolution(data=inp, weight=weight, num_filter=num_filter, \
                                 kernel=kernel, stride=(SMALL_Y, SMALL_Y), no_bias=True)
    assert out.shape == (batch_size, channel + 1, 1, (width + SMALL_Y - 1)// SMALL_Y)
    assert out[0][0][0][0] == channel * kernel[0] * kernel[1]
    assert inp.grad.shape == inp.shape
    assert inp.grad[0][0][0][0] == 0


@use_np
def test_deconvolution():
    dim = 2
    batch_size = 1
    channel = 3
    height = SMALL_Y
    width = LARGE_X // 5
    num_filter = 4
    kernel = (4,) * dim   # => shape = (3, 3)

    inp=mx.np.ones(shape=(batch_size, channel, 1, width))
    weight = mx.np.ones(shape=(channel, num_filter, kernel[0], kernel[1]))
    bias = mx.np.array(num_filter,)
    inp.attach_grad()
    with mx.autograd.record():
        out = mx.npx.deconvolution(data=inp, weight=weight, num_filter=num_filter, \
                                   kernel=kernel, no_bias=True)
    assert out.shape == (batch_size, channel + 1, kernel[0], width + 3)
    assert out[0][0][kernel[0]//2][kernel[1]] == channel * num_filter
    assert inp.grad.shape == inp.shape
    assert inp.grad[0][0][0][0] == 0


@use_np
def test_dropout():
    shape = (LARGE_X, SMALL_Y)
    inp = mx.np.ones(shape=shape)
    inp.attach_grad()
    with mx.autograd.record():
        out = npx.dropout(inp, p=0.5, cudnn_off=True)
        out.backward()
    assert out.shape == shape
    assert _np.count_nonzero(out[0] == 2) != 0
    assert inp.grad.shape == shape
    assert _np.count_nonzero(inp.grad[0] == 2) != 0


@use_np
def test_log_softmax():
    LOG_SOFTMAX_VAL = -18.420681
    ndim = 2
    shape = (LARGE_X, SMALL_Y)
    axis = 1
    inp = np.ones(shape=shape)
    inp.attach_grad()
    with mx.autograd.record():
        out = npx.log_softmax(inp, axis=0)
        out.backward()
    assert out.shape == shape
    assert_almost_equal(out[-1][-1], LOG_SOFTMAX_VAL, atol=1e-3, rtol=1e-3)
    assert inp.grad.shape == shape
    assert_almost_equal(inp.grad[0][0], 0.0, atol=1e-3, rtol=1e-3)


@use_np
def test_relu():
    shape = (LARGE_X, SMALL_Y)
    inp = np.ones(shape)
    inp[:, shape[1] // 2:shape[1]] = -1
    inp.attach_grad()
    with mx.autograd.record():
        out = npx.relu(inp)
        out.backward()
    assert out.shape == shape
    assert out[0][0] == 1
    assert out[-1][-1] == 0
    assert inp.grad.shape == shape
    assert inp.grad[0][0] == 1


@use_np
def test_leaky_relu():
    inp = -1 * mx.np.ones(shape=(LARGE_X, SMALL_Y))
    inp.attach_grad()

    def check_leaky():
        with mx.autograd.record():
            res = mx.npx.leaky_relu(inp, act_type="leaky", slope=0.3)
            res.backward()
        assert_almost_equal(res[-1][-1], 0.3 * inp[-1][-1], atol=1e-3, rtol=1e-3)
        assert_almost_equal(inp.grad[0][-1], -0.3 * inp[-1][-1], atol=1e-3, rtol=1e-3)

    def check_elu():
        with mx.autograd.record():
            res = mx.npx.leaky_relu(inp, act_type="elu", slope=0.3)
            res.backward()
        assert_almost_equal(res[-1][-1], 0.3*(_np.exp(inp[-1][-1])-1), atol=1e-3, rtol=1e-3)
        assert_almost_equal(inp.grad[-1][0], 0.3*_np.exp(inp[-1][-1]), atol=1e-3, rtol=1e-3)

    def check_selu():
        lam = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        with mx.autograd.record():
            res = mx.npx.leaky_relu(inp, act_type="selu")
            res.backward()
        assert_almost_equal(res[-1][-1], (lam * alpha * (_np.exp(inp[-1][-1])-1)), atol=1e-3, rtol=1e-3)
        assert_almost_equal(inp.grad[0][0], 0.590423, atol=1e-3, rtol=1e-3)

    check_leaky()
    check_elu()
    check_selu()


@use_np
def test_norm():
    shape = (LARGE_X * 2, SMALL_Y // 2)
    inp = np.ones(shape)
    inp.attach_grad()
    with mx.autograd.record():
        out = npx.norm(inp, ord=2, axis=1)
        out.backward()
    assert out.shape == (shape[0],)
    assert out[shape[0] - 1] == 5
    assert inp.grad.shape == shape
    assert_almost_equal(inp.grad[0][0], 1/5, atol=1e-3, rtol=1e-3)


@use_np
def test_embedding():
    inp = np.arange(0, 4, dtype=np.int32).reshape(2, 2)
    vec = np.ones((4, INT_OVERFLOW))
    vec[0][INT_OVERFLOW-1] = 3
    vec[1][INT_OVERFLOW//2] = 2
    vec[2][1] = 2
    vec[3][0] = 0
    out = npx.embedding(inp, vec, 4, INT_OVERFLOW)
    assert out[0][0][INT_OVERFLOW-1] == 3
    assert out[0][0][0] == 1
    assert out[0][1][INT_OVERFLOW//2] == 2
    assert out[0][1][1] == 1
    assert out[1][0][1] == 2
    assert out[1][0][INT_OVERFLOW//2] == 1
    assert out[1][1][0] == 0
    assert out[1][1][INT_OVERFLOW-1] == 1
