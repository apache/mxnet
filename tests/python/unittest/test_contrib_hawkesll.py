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

import mxnet as mx
import numpy as np
from mxnet import nd


def test_hawkesll_output_ok():
    T, N, K = 4, 4, 3

    mu = nd.array([1.5, 2.0, 3.0]).tile((N, 1))
    alpha = nd.array([0.2, 0.3, 0.4])
    beta = nd.array([1.0, 2.0, 3.0])

    lags = nd.array([[6, 7, 8, 9], [1, 2, 3, 4], [3, 4, 5, 6], [8, 9, 10, 11]])
    marks = nd.zeros((N, T)).astype(np.int32)
    states = nd.zeros((N, K))

    valid_length = nd.array([1, 2, 3, 4])
    max_time = nd.ones((N,)) * 100.0

    A = nd.contrib.hawkesll(
        mu, alpha, beta, states, lags, marks, valid_length, max_time
    )

    assert np.allclose(
        np.array([-649.79453489, -649.57118596, -649.38025115, -649.17811484]),
        A[0].asnumpy(),
    )


def test_hawkesll_output_multivariate_ok():
    T, N, K = 9, 2, 3

    mu = nd.array([1.5, 2.0, 3.0])
    alpha = nd.array([0.2, 0.3, 0.4])
    beta = nd.array([2.0, 2.0, 2.0])

    lags = nd.array([[6, 7, 8, 9, 3, 2, 5, 1, 7], [1, 2, 3, 4, 2, 1, 2, 1, 4]])
    marks = nd.array([[0, 1, 2, 1, 0, 2, 1, 0, 2], [1, 2, 0, 0, 0, 2, 2, 1, 0]]).astype(
        np.int32
    )

    states = nd.zeros((N, K))

    valid_length = nd.array([7, 9])
    max_time = nd.ones((N,)) * 100.0

    A = nd.contrib.hawkesll(
        mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
    )

    assert np.allclose(np.array([-647.01240372, -646.28617272]), A[0].asnumpy())


def test_hawkesll_backward_correct():
    ctx = mx.cpu()

    mu = nd.array([1.5, 2.0, 3.0])
    alpha = nd.array([0.2, 0.3, 0.4])
    beta = nd.array([2.0, 2.0, 2.0])

    T, N, K = 9, 2, 3
    lags = nd.array([[6, 7, 8, 9, 3, 2, 5, 1, 7], [1, 2, 3, 4, 2, 1, 2, 1, 4]])
    marks = nd.array([[0, 0, 0, 1, 0, 0, 1, 2, 0], [1, 2, 0, 0, 0, 2, 2, 1, 0]]).astype(
        np.int32
    )
    valid_length = nd.array([9, 9])
    states = nd.zeros((N, K))

    max_time = nd.ones((N,)) * 100.0

    mu.attach_grad()
    alpha.attach_grad()
    beta.attach_grad()

    with mx.autograd.record():
        A, _ = nd.contrib.hawkesll(
            mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
        )
    A.backward()

    dmu, dalpha, dbeta = (
        np.array([-193.33987481, -198.0, -198.66828681]),
        np.array([-9.95093892, -4.0, -3.98784892]),
        np.array([-1.49052169e-02, -5.87469511e-09, -7.29065224e-03]),
    )
    assert np.allclose(dmu, mu.grad.asnumpy())
    assert np.allclose(dalpha, alpha.grad.asnumpy())
    assert np.allclose(dbeta, beta.grad.asnumpy())


def test_hawkesll_forward_single_mark():
    _dtype = np.float32

    mu = nd.array([1.5]).astype(_dtype)
    alpha = nd.array([0.2]).astype(_dtype)
    beta = nd.array([1.0]).astype(_dtype)

    T, N, K = 7, 1, 1
    lags = nd.array([[6, 7, 8, 3, 2, 1, 7]]).astype(_dtype)
    marks = nd.array([[0, 0, 0, 0, 0, 0, 0]]).astype(np.int32)
    valid_length = nd.array([7]).astype(_dtype)

    states = nd.zeros((N, K)).astype(_dtype)
    max_time = nd.ones((N,)).astype(_dtype) * 100

    A, _ = nd.contrib.hawkesll(
        mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
    )

    assert np.allclose(A[0].asscalar(), -148.4815)


def test_hawkesll_backward_single_mark():
    _dtype = np.float32

    mu = nd.array([1.5]).astype(_dtype)
    alpha = nd.array([0.2]).astype(_dtype)
    beta = nd.array([1.0]).astype(_dtype)

    T, N, K = 7, 1, 1
    lags = nd.array([[6, 7, 8, 3, 2, 1, 7]]).astype(_dtype)
    marks = nd.array([[0, 0, 0, 0, 0, 0, 0]]).astype(np.int32)
    valid_length = nd.array([7]).astype(_dtype)

    states = nd.zeros((N, K)).astype(_dtype)
    max_time = nd.ones((N,)).astype(_dtype) * 40

    mu.attach_grad()
    beta.attach_grad()

    with mx.autograd.record():
        A, _ = nd.contrib.hawkesll(
            mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
        )

    A.backward()

    assert np.allclose(beta.grad.asnumpy().sum(), -0.05371582)


if __name__ == "__main__":
    import nose

    nose.runmodule()
