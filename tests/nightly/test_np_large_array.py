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

@use_np
def test_abs():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.abs(A)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)

@use_np
def test_absolute():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.abs(A)
    print(B)
    assert B.shape == (INT_OVERFLOW, 2)

def test_add():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 2))
    C = np.add(A, B)
    print(C)
    assert C.shape == (INT_OVERFLOW, 2)

def test_add_broadcast():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.ones((INT_OVERFLOW, 1))
    C = np.add(A, B)
    print(C)
    assert C.shape == (INT_OVERFLOW, 2)

def test_all():
    A = np.ones((INT_OVERFLOW, 2))
    B = np.all(A)
    print(B)
    assert B.asnumpy() == True

def test_amin():
    A = np.ones((INT_OVERFLOW, 2))
    A[100][1] = -1
    B = np.amin(A)
    print(B)
    assert B.asnumpy() == -1.0

def test_amax():
    A = np.zeros((INT_OVERFLOW, 2))
    A[100][1] = 1
    B = np.amax(A)
    print(B)
    assert B.asnumpy() == 1.0

def test_argmin():
    A = np.ones((INT_OVERFLOW, 2))
    A[10][1] = -1
    B = np.argmin(A)
    print(B)
    assert B.asnumpy() == 21
    
def test_argmax():
    A = np.zeros((INT_OVERFLOW, 2))
    A[10][1] = 1
    B = np.argmax(A)
    print(B)
    assert B.asnumpy() == 21

def test_trigonometric_family():
    def batch_check(x, funcs):
        for f in funcs:
            y = f(x)
            print(y)
            assert y.shape == (INT_OVERFLOW, 2)
    A = np.ones((INT_OVERFLOW, 2))
    batch_check(A, [np.arccos, np.arccosh, np.arcsin, \
        np.arcsin, np.arctan, np.arctanh, np.sin, np.cos, \
        np.tan, np.sinh, np.cosh, np.tanh])
