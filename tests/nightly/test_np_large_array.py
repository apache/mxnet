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


# dimension constants
MEDIUM_X = 10000
LARGE_X = 100000000
SMALL_X = 100
SMALL_Y = 50
INT_OVERFLOW = 2**31
HALF_INT_OVERFLOW = 2**30


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

@use_np
def test_zeros():
    A = np.zeros((INT_OVERFLOW, 2))
    assert A.shape == (INT_OVERFLOW, 2)

@use_np
def test_abs():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.abs(A)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_absolute():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.absolute(A)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_add():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        C = np.add(A, B)
    print(C)
    assert C.shape == (INT_OVERFLOW, 2)
    C.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

# this will fail; broadcast needs to be fixed
# TODO add backward test after forward is fixed
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_add_broadcast():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 1))
    C = np.add(A, B)
    print(C)
    assert C.shape == (INT_OVERFLOW, 2)

@use_np
def test_all():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.all(A)
    print(B)
    assert B.asnumpy() == True
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_amin():
    A = np.ones((INT_OVERFLOW, 2))
    A[100][1] = -1
    A.attach_grad()
    with mx.autograd.record():
        B = np.amin(A)
    print(B)
    assert B.asnumpy() == -1.0
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_amax():
    A = np.zeros((INT_OVERFLOW, 2))
    A[100][1] = 1
    A.attach_grad()
    with mx.autograd.record():
        B = np.amax(A)
    print(B)
    assert B.asnumpy() == 1.0
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_argmin():
    A = np.ones((INT_OVERFLOW, 2))
    A[10][1] = -1
    A.attach_grad()
    with mx.autograd.record():
        B = np.argmin(A)
    print(B)
    assert B.asnumpy() == 21
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_argmax():
    A = np.zeros((INT_OVERFLOW, 2))
    A[10][1] = 1
    A.attach_grad()
    with mx.autograd.record():
        B = np.argmax(A)
    print(B)
    assert B.asnumpy() == 21
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_trigonometric_family():
    def batch_check(x, funcs):
        for f in funcs:
            x.attach_grad()
            with mx.autograd.record():
                y = f(x)
            print(y)
            assert y.shape == (INT_OVERFLOW, 2)
            y.backward()
            assert x.grad.shape == (INT_OVERFLOW, 2)
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
    print(B)
    assert B.asnumpy() == False
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_append():
    A = np.ones((1, INT_OVERFLOW))
    B = np.ones((2, INT_OVERFLOW))
    A.attach_grad() 
    with mx.autograd.record():
        C = np.append(A, B, axis=0)
    print(C.shape)
    assert C.shape == (3, INT_OVERFLOW)
    C.backward()
    assert A.grad.shape == (1, INT_OVERFLOW)

@use_np
def test_arange():
    A = np.arange(INT_OVERFLOW)
    print(A)
    assert A.shape == (INT_OVERFLOW, )

@use_np
def test_argsort():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.argsort(A)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

# broken
# TODO add backward test after foward is fixed
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_round():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.round(A)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)

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

@use_np
def test_atleast_xd_family():
    def batch_check(x, funcs, shapes):
        for f, s in zip(funcs, shapes):
            x.attach_grad()
            with mx.autograd.record():
                y = f(x)
            print(y.shape)
            assert y.shape == s
            y.backward()
            assert x.grad.shape == (INT_OVERFLOW, )
    A = np.zeros((INT_OVERFLOW))
    batch_check(A, [np.atleast_1d, np.atleast_2d, np.atleast_3d], \
            [(INT_OVERFLOW, ), (1, INT_OVERFLOW), (1, INT_OVERFLOW, 1)])

@use_np
def test_average():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.average(A)
    print(B)
    assert B.asnumpy() == 1
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_bincount():
    A = np.ones((INT_OVERFLOW), dtype='int32')
    A[0] = 0
    A.attach_grad()
    with mx.autograd.record():
        B = np.bincount(A)
    print(B)
    assert B.shape == (2,)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, )

# broken
# TODO add backward test after forward is fixed
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_bitwise_family():
    def batch_check(x1, x2, funcs):
        for f in funcs:
            y = f(x1, x2)
            print(y)
            assert y.shape == (INT_OVERFLOW, 2)
    # test on broadcast input
    A = np.ones((INT_OVERFLOW, 1), dtype='int32')
    B = np.ones((INT_OVERFLOW, 2), dtype='int32')
    batch_check(A, B, [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
    C = np.bitwise_not(A)
    print(C)
    assert C.shape == (INT_OVERFLOW, 1)

@use_np
def test_blackman():
    A = np.blackman((INT_OVERFLOW))
    print(A)
    assert A.shape == (INT_OVERFLOW, )

@use_np
def test_broadcast_to():
    A = np.ones((2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.broadcast_to(A, (INT_OVERFLOW, 2))
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (2, )
    with mx.autograd.record():
        B = np.broadcast_to(A.reshape(2, 1), (2, INT_OVERFLOW))
    print(B)
    assert B.shape == (2, INT_OVERFLOW)
    B.backward()
    assert A.grad.shape == (2, )

@use_np
def test_root_family():
    def batch_check(x, funcs):
        for f in funcs:
            x.attach_grad()
            with mx.autograd.record():
                y = f(x)
            print(y)
            assert y.shape == (INT_OVERFLOW, 2)
            y.backward()
            assert x.grad.shape == (INT_OVERFLOW, 2)
    A = np.ones((INT_OVERFLOW, 2))
    batch_check(A, [np.sqrt, np.cbrt])

@use_np
def test_ceil_floor():
    def batch_check(x, funcs):
        for f in funcs:
            x.attach_grad()
            with mx.autograd.record():
                y = f(x)
            print(y)
            assert y.shape == (INT_OVERFLOW, 2)
            y.backward()
            assert x.grad.shape == (INT_OVERFLOW, 2)
    A = np.ones((INT_OVERFLOW, 2))
    batch_check(A, [np.ceil, np.floor])

@use_np
def test_clip():
    A = np.ones((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = np.clip(A, 1, 1)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_column_stack():
    A = np.ones(INT_OVERFLOW)
    A.attach_grad()
    with mx.autograd.record():
        B = np.column_stack((A, A))
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, )

@use_np
def test_concatenate():
    def batch_check(x1, x2, axises, shapes):
        for a, s in zip(axises, shapes):
            x1.attach_grad()
            with mx.autograd.record():
                y = np.concatenate((x1, x2), axis=a)
            print(y)
            assert y.shape == s
            y.backward()
            assert x1.grad.shape == (2, INT_OVERFLOW)
    A = np.ones((2, INT_OVERFLOW))
    B = np.ones((1, INT_OVERFLOW))
    batch_check(A, B, [0, None], \
            [(3, INT_OVERFLOW), (int(INT_OVERFLOW * 3), )])

@use_np
def test_copysign():
    A = np.ones((INT_OVERFLOW, 2))
    #A.attach_grad()
    #with mx.autograd.record():
    B = np.copysign(A, -1)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    #B.backward()
    #assert A.grad.shape == (INT_OVERFLOW, 2)
    
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
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_arange_like():
    A = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.arange_like(A)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    B.backward()
    assert A.grad.shape == (INT_OVERFLOW, 2)

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
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)
    A.grad.shape == (INT_OVERFLOW, 2)

@use_np
def test_broadcast_like():
    A = np.ones((1, 2))
    B = np.zeros((INT_OVERFLOW, 2))
    A.attach_grad()
    with mx.autograd.record():
        C = npx.broadcast_like(A, B)
    print(C)
    assert C.shape == (INT_OVERFLOW, 2)
    C.backward()
    assert A.grad.shape == (1, 2)
    with mx.autograd.record():
        C = npx.broadcast_like(A.reshape(2, 1), B.T)
    print(C)
    assert C.shape == (2, INT_OVERFLOW)
    C.backward()
    assert A.grad.shape == (1, 2)

@use_np
def test_constraint_check():
    A = np.ones((2, INT_OVERFLOW))
    constraint = (A > 0)
    B = npx.constraint_check(constraint)
    assert B.asnumpy() == True

# broken
@use_np
@pytest.mark.skip(reason='Does not support large tensor; to be fixed')
def test_batch_flatten():
    A = np.ones((2, 1, INT_OVERFLOW))
    A.attach_grad()
    with mx.autograd.record():
        B = npx.batch_flatten(A)
    print(B)
    assert B.shape == (2, INT_OVERFLOW)
    B.backward()
    assert A.grad.shape == (2, 1, INT_OVERFLOW)

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
    print(B)
    assert B.shape == (2, INT_OVERFLOW)
    B.backward()
    assert A.grad.shape == (2, INT_OVERFLOW)
