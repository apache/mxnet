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

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, default_context, check_symbolic_forward, create_2d_tensor, use_np
from mxnet import gluon, np, npx
from common import with_seed
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
def test_abs():
    A = np.ones((INT_OVERFLOW, 2))
    A[0][0] = -1
    A.attach_grad()
    with mx.autograd.record():
        B = np.abs(A)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == -1

@use_np
def test_absolute():
    A = np.ones((INT_OVERFLOW, 2))
    A[0][0] = -1
    A.attach_grad()
    with mx.autograd.record():
        B = np.absolute(A)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == -1

@use_np
@pytest.mark.skip(reason='backward errors out on (2^30,2), gives wrong result \
    on (2^31, 2)')
def test_add():
    INT_OVERFLOW = 2**30
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        C = np.add(A, B)
    assert C.shape == (INT_OVERFLOW, 2)
    assert C[0][0] == 2
    C.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 1

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
    A = np.ones((INT_OVERFLOW, 2))
    A[100][1] = -1
    A.attach_grad()
    with mx.autograd.record():
        B = np.amin(A)
    assert B == -1.0
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0

@use_np
def test_amax():
    A = np.zeros((INT_OVERFLOW, 2))
    A[100][1] = 1
    A.attach_grad()
    with mx.autograd.record():
        B = np.amax(A)
    print(B)
    assert B == 1.0
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0

@use_np
def test_argmin():
    A = np.ones((INT_OVERFLOW, 2))
    A[10][1] = -1
    A.attach_grad()
    with mx.autograd.record():
        B = np.argmin(A)
    print(B)
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
    print(B)
    assert B == 21
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)
    assert A.grad[0][0] == 0

@use_np
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

# broken
# TODO add backward test after foward is fixed
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_round():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.round(A)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == 1

# broken
# TODO add backward test after forward is fixed
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_array_split():
    A = np.zeros((INT_OVERFLOW, 2))
    B = np.array_split(A, 2)
    print(B)
    assert B[0].shape ==(HALF_INT_OVERFLOW, 2)
    assert B[1].shape ==(HALF_INT_OVERFLOW, 2)
    assert B[0][0][0] == 0

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

# broken
# TODO add backward test after forward is fixed
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_bitwise_family():
    def batch_check(x1, x2, funcs):
        for f in funcs:
            y = f(x1, x2)
            one = np.ones((1), dtype='int32')
            assert y.shape == (INT_OVERFLOW, 2)
            assert y[0][0] == f(one, one)
    # test on broadcast input
    A = np.ones((INT_OVERFLOW, 1), dtype='int32')
    B = np.ones((INT_OVERFLOW, 2), dtype='int32')
    batch_check(A, B, [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
    C = np.bitwise_not(A)
    assert C.shape == (INT_OVERFLOW, 1)
    assert C[0] == np.bitwise_not(np.ones((1), dtype='int32')) 

@use_np
def test_blackman():
    A = np.blackman((INT_OVERFLOW))
    assert A.shape == (INT_OVERFLOW, )

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
# backward not working https://github.com/apache/incubator-mxnet/issues/18952
def test_copysign():
    A = np.ones((INT_OVERFLOW, 2))
    #A.attach_grad()
    #with mx.autograd.record():
    B = np.copysign(A, -1)
    assert B.shape == (INT_OVERFLOW, 2)
    assert B[0][0] == -1
    #B.backward()
    #assert A.grad.shape == (INT_OVERFLOW, 2)
    
@pytest.mark.skip(reason="CI hasn't switch to ILP64 OpenBLAS yet")
@use_np
def test_dot():
    A = np.ones((1, INT_OVERFLOW), dtype='float32')
    B = np.ones((INT_OVERFLOW, 1), dtype='float32')
    A.attach_grad()
    with mx.autograd.record():
        C = np.dot(A, B)
    assert_almost_equal(C, [INT_OVERFLOW], rtol=1e-5, atol=1e-5)
    C.backward()
    assert A.grad.shape == (1, INT_OVERFLOW)
    assert A.grad[0][0] == 1

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

# TODO implement this test after dot is fixed for large tensors and we have
# migrated to using ILP64 BLAS/LAPACK
@use_np
@pytest.mark.skip(reason='dot is known to not work on large tensors. PR to fix: https://github.com/apache/incubator-mxnet/pull/18925')
def test_batch_dot():
    assert False 

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

# broken
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_batch_norm():
    A = np.ones((2, INT_OVERFLOW))
    gamma = np.ones((2))
    beta = np.zeros((2))
    mov_mean = np.ones((2))
    mov_var = np.ones((2))
    A.attach_grad() 
    with mx.autograd.record():
        B = npx.batch_norm(A, gamma, beta, mov_mean, mov_var)
    assert B.shape == (2, INT_OVERFLOW)
    assert B[0][0] == 0
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
    assert A.grad[0][0] == 0

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
def test_rnn_dim_check():
    L_SEQ, BAT, L_INP, L_STA = 2**31, 4, 2**10, 2
    data = np.random.uniform(-1, 1, (L_SEQ, BAT, L_INP))
    state = np.random.normal(0, 1, (1, BAT, L_STA))
    params = np.random.normal(0, 1, (2056,))
    assertRaises(ValueError, npx.rnn, data=data, parameters=params, \
        mode='rnn_relu', state=state, state_size=L_STA, num_layers=1)


@use_np
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
