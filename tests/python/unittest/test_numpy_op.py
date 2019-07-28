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

# pylint: skip-file
from __future__ import absolute_import
import numpy as _np
import mxnet as mx
from mxnet import np, npx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.base import MXNetError
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient, use_np
from common import assertRaises, with_seed
import random
import collections


@with_seed()
@use_np
def test_np_svd():
    class TestSVD(HybridBlock):
        def __init__(self):
            super(TestSVD, self).__init__()

        def hybrid_forward(self, F, data):
            return F.np.linalg.svd(data)

    def get_grad(UT, L, V):
        m = V.shape[-2]
        n = V.shape[-1]
        E = _np.zeros_like(UT)
        dUT = _np.ones_like(UT)
        dV = _np.ones_like(V)
        for i in range(m):
            for j in range(i + 1, m):
                denom1 = _np.maximum(L[..., i] - L[..., j], 1e-20)
                denom2 = _np.maximum(L[..., i] + L[..., j], 1e-20)
                E[..., i, j] = 1.0 / denom1 / denom2
                E[..., j, i] = -E[..., i, j]
            E[..., i, i] = 0
        G1 = _np.matmul(1.0 / L[..., None] * dV, _np.swapaxes(V, -2, -1)) * L[..., None, :]
        G1 = G1 + _np.matmul(_np.swapaxes(dUT, -2, -1), UT)
        X = G1 * E
        G2 = _np.eye(m) + (X + _np.swapaxes(X, -2, -1)) * L[..., None, :] - 1.0 / L[..., None] * _np.matmul(dV, _np.swapaxes(V, -2, -1)) * _np.eye(m)
        dA = _np.matmul(UT, _np.matmul(G2, V) + 1.0 / L[..., None] * dV)
        return dA

    shapes = [
        (3, 3),
        (3, 5),
        (4, 4),
        (4, 5),
        (5, 5),
        (5, 6),
        (6, 6),
        (0, 1),
        (6, 5, 6),
        (2, 3, 3, 4),
        (4, 2, 1, 2),
        (0, 5, 3, 3),
        (5, 0, 3, 3),
        (3, 3, 0, 0),
    ]
    dtypes = ['float32', 'float64']
    for hybridize in [True, False]:
        for dtype in dtypes:
            for shape in shapes:
                rtol = 1e-3
                atol = 1e-3
                test_svd = TestSVD()
                if hybridize:
                    test_svd.hybridize()
                data_np = _np.random.uniform(-10.0, 10.0, shape)
                data_np = _np.array(data_np, dtype=dtype)
                data = np.array(data_np, dtype=dtype)
                data.attach_grad()
                with mx.autograd.record():
                    ret = test_svd(data)
                UT = ret[0].asnumpy()
                L = ret[1].asnumpy()
                V = ret[2].asnumpy()
                # check UT @ L @ V == A
                t = _np.matmul(UT * L[..., None, :], V)
                assert t.shape == data_np.shape
                assert_almost_equal(t, data_np, rtol=rtol, atol=atol)
                # check UT @ U == I
                I = _np.matmul(UT, _np.swapaxes(UT, -2, -1))
                I_np = _np.ones_like(UT) * _np.eye(shape[-2])
                assert I.shape == I_np.shape
                assert_almost_equal(I, I_np, rtol=rtol, atol=atol)
                # check U @ UT == I
                I = _np.matmul(_np.swapaxes(UT, -2, -1), UT)
                I_np = _np.ones_like(UT) * _np.eye(shape[-2])
                assert I.shape == I_np.shape
                assert_almost_equal(I, I_np, rtol=rtol, atol=atol)
                # check V @ VT == I
                I = _np.matmul(V, _np.swapaxes(V, -2, -1))
                I_np = _np.ones_like(UT) * _np.eye(shape[-2])
                assert I.shape == I_np.shape
                assert_almost_equal(I, I_np, rtol=rtol, atol=atol)
                # check descending singular values
                s = [L[..., i] - L[..., i + 1] for i in range(L.shape[-1] - 1)]
                s = _np.array(s)
                assert (s >= -1e-5).all()
                if L.size > 0:
                    assert (L[..., -1] >= -1e-5).all()
                # check backward
                mx.autograd.backward(ret)
                if ((s > 1e-5).all()):
                    backward_expected = get_grad(ret[0].asnumpy(), ret[1].asnumpy(), ret[2].asnumpy())
                    assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)


if __name__ == '__main__':
    import nose
    nose.runmodule()
