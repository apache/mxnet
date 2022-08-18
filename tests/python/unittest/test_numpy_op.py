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
from distutils.version import StrictVersion
import sys
import copy
import itertools
from mxnet.gluon.parameter import Parameter
import numpy as onp
import platform
import mxnet as mx
import scipy.stats as ss
import scipy.special as scipy_special
import pytest
import mxnet.ndarray.numpy._internal as _npi
from functools import reduce
from packaging.version import parse
from mxnet import np, npx
from mxnet.gluon import HybridBlock
from mxnet.base import MXNetError
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient, use_np, collapse_sum_like, effective_dtype
from mxnet.test_utils import new_matrix_with_real_eigvals_nd
from mxnet.test_utils import new_sym_matrix_with_real_eigvals_nd
from common import assertRaises, retry, xfail_when_nonstandard_decimal_separator
import random
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf
from mxnet.numpy_op_signature import _get_builtin_op
from mxnet.test_utils import is_op_runnable, has_tvm_ops, rand_shape_2d
from mxnet.operator import get_all_registered_operators
from common import assert_raises_cuda_not_satisfied
from numpy.testing import assert_allclose


@use_np
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('a_shape,b_shape,axes', [
    ((3, 5), (5, 4), 1),
    ((3,), (3,), 1),
    ((3, 4, 5, 3, 2), (5, 3, 2, 1, 2), 3),
    ((3, 5, 4, 3, 2), (2, 3, 5, 1, 2), [[1, 3, 4], [2, 1, 0]]),
    ((3, 5, 4), (5, 4, 3), [[1, 0, 2], [0, 2, 1]]),
    ((3, 5, 4), (5, 3, 4), [[2, 0], [-1, -2]]),
    ((2, 2), (2, 2), 2),
    ((3, 5, 4), (5, ), [[-2], [0]]),
    ((3, 5, 4), (5, ), [[1], [0]]),
    ((2,), (2, 3), 1),
    ((3,), (3,), 0),
    ((2,), (2, 3), 0),
    ((3, 5, 4), (5, ), 0),
    ((2, 3, 4), (4, 3, 2), [[], []]),
    ((3, 0), (0, 5), 1),
    ((3, 0), (0, 4), [[1], [0]]),
    ((0, 3), (3, 5), 1),
    ((0, 3), (5, 0), [[0], [1]])
])
def test_np_tensordot(a_shape, b_shape, axes, hybridize, dtype):
    class TestTensordot(HybridBlock):
        def __init__(self, axes):
            super(TestTensordot, self).__init__()
            self._axes = axes

        def forward(self, a, b):
            return np.tensordot(a, b, self._axes)

    def tensordot_backward(out_grad, a, b, axes=2):
        if (a.ndim < 1) or (b.ndim < 1):
            raise ValueError('An input is zero-dim')

        if onp.isscalar(axes):
            a_axes_summed = [i + a.ndim - axes for i in range(axes)]
            b_axes_summed = [i for i in range(axes)]
        else:
            if len(axes) != 2:
                raise ValueError('Axes must consist of two arrays.')
            a_axes_summed, b_axes_summed = axes
            if onp.isscalar(a_axes_summed):
                a_axes_summed = a_axes_summed,
            if onp.isscalar(b_axes_summed):
                b_axes_summed = b_axes_summed,

            for i in range(len(a_axes_summed)):
                a_axes_summed[i] = (a_axes_summed[i] + a.ndim) % a.ndim

            for i in range(len(b_axes_summed)):
                b_axes_summed[i] = (b_axes_summed[i] + b.ndim) % b.ndim

        if len(a_axes_summed) != len(b_axes_summed):
            raise ValueError('Axes length mismatch')

        a_axes_remained = []
        for i in range(a.ndim):
            if not (i in a_axes_summed):
                a_axes_remained.append(i)
        a_axes = a_axes_remained[:] + a_axes_summed[:]

        b_axes_remained = []
        for i in range(b.ndim):
            if not (i in b_axes_summed):
                b_axes_remained.append(i)
        b_axes = b_axes_summed[:] + b_axes_remained[:]

        ad1 = onp.prod([a.shape[i] for i in a_axes_remained]) if len(a_axes_remained) > 0 else 1
        ad2 = onp.prod([a.shape[i] for i in a_axes_summed]) if len(a_axes_summed) > 0 else 1
        bd1 = onp.prod([b.shape[i] for i in b_axes_summed]) if len(b_axes_summed) > 0 else 1
        bd2 = onp.prod([b.shape[i] for i in b_axes_remained]) if len(b_axes_remained) > 0 else 1

        out_grad = out_grad.reshape((ad1, bd2))

        new_a = onp.transpose(a, a_axes)
        new_a_shape = new_a.shape[:]
        new_a = new_a.reshape((ad1, ad2))
        new_b = onp.transpose(b, b_axes)
        new_b_shape = new_b.shape[:]
        new_b = new_b.reshape((bd1, bd2))

        reverse_a_axes = [0 for i in a_axes]
        for i in range(len(a_axes)):
            reverse_a_axes[a_axes[i]] = i

        reverse_b_axes = [0 for i in b_axes]
        for i in range(len(b_axes)):
            reverse_b_axes[b_axes[i]] = i

        grad_b = onp.dot(new_a.T, out_grad).reshape(new_b_shape)
        grad_b = onp.transpose(grad_b, reverse_b_axes)
        grad_a = onp.dot(out_grad, new_b.T).reshape(new_a_shape)
        grad_a = onp.transpose(grad_a, reverse_a_axes)

        return [grad_a, grad_b]

    test_tensordot = TestTensordot(axes)
    if hybridize:
        test_tensordot.hybridize()
    a = rand_ndarray(shape = a_shape, dtype = dtype).as_np_ndarray()
    b = rand_ndarray(shape = b_shape, dtype = dtype).as_np_ndarray()
    a.attach_grad()
    b.attach_grad()

    np_out = onp.tensordot(a.asnumpy(), b.asnumpy(), axes)
    with mx.autograd.record():
        mx_out = test_tensordot(a, b)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)
    mx_out.backward()
    np_backward = tensordot_backward(onp.ones(np_out.shape), a.asnumpy(), b.asnumpy(), axes)
    assert_almost_equal(a.grad.asnumpy(), np_backward[0], rtol = 1e-3, atol=1e-5)
    assert_almost_equal(b.grad.asnumpy(), np_backward[1], rtol = 1e-3, atol=1e-5)

    # Test imperative once again
    mx_out = np.tensordot(a, b, axes)
    np_out = onp.tensordot(a.asnumpy(), b.asnumpy(), axes)
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    # test numeric gradient
    if (onp.prod(a_shape) > 0 and onp.prod(b_shape) > 0):
        a_sym = mx.sym.Variable("a").as_np_ndarray()
        b_sym = mx.sym.Variable("b").as_np_ndarray()
        mx_sym = mx.sym.np.tensordot(a_sym, b_sym, axes).as_nd_ndarray()
        check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
          rtol=1e-1, atol=1e-1, dtype = dtype)

    # General Gradient Test
    for a_grad_status in ['add', 'write']:
        for b_grad_status in ['add', 'write']:
            a = mx.np.random.normal(0, 1, a_shape)
            b = mx.np.random.normal(0, 1, b_shape)
            a.attach_grad(a_grad_status)
            b.attach_grad(b_grad_status)
            if a_grad_status == 'add':
                ori_a_grad = mx.np.random.normal(0, 1, a_shape)
                if a.ndim == 0:
                    a.grad[()] = ori_a_grad
                else:
                    a.grad[:] = ori_a_grad
            if b_grad_status == 'add':
                ori_b_grad = mx.np.random.normal(0, 1, b_shape)
                if b.ndim == 0:
                    b.grad[()] = ori_b_grad
                else:
                    b.grad[:] = ori_b_grad

            with mx.autograd.record():
                mx_out = mx.np.tensordot(a, b, axes)
                out_grad = mx.np.random.normal(0, 1, mx_out.shape)
                loss = (mx_out * out_grad).sum()
                loss.backward()

            gt_in_grad = tensordot_backward(out_grad.asnumpy(), a.asnumpy(), b.asnumpy(), axes)

            if(a_grad_status == 'add'):
                gt_in_grad[0] += ori_a_grad
            if(b_grad_status == 'add'):
                gt_in_grad[1] += ori_b_grad

            assert_almost_equal(a.grad.asnumpy(), gt_in_grad[0], rtol=1e-2, atol=1e-2)
            assert_almost_equal(b.grad.asnumpy(), gt_in_grad[1], rtol=1e-2, atol=1e-2)


@use_np
@pytest.mark.parametrize('shape_a,shape_b', [
    ((3, 0), (0, 4)),
    ((3,), (3,)),
    ((3, 4), (4, 5)),
    ((), ()),
    ((3, 4, 5), ()),
    ((), (3, 4, 5)),
    ((3, 4, 5), (5, )),
    ((3, 4, 5), (5, 2)),
    ((5,), (5, 2)),
    ((3, 5, 4), (5, 4, 3)),
    ((3, 4), (5, 4, 3)),
    ((4,), (5, 4, 3))
])
def test_np_dot(shape_a, shape_b):
    eps = 1e-3

    np_a = onp.random.uniform(-1.0, 1.0, shape_a)
    np_a[abs(np_a) < eps] = 2 * eps
    np_b = onp.random.uniform(-1.0, 1.0, shape_b)
    np_b[abs(np_b) < eps] = 2 * eps
    a = mx.nd.array(np_a)
    b = mx.nd.array(np_b)
    np_res = onp.dot(np_a, np_b)
    mx_res = np.dot(a.as_np_ndarray(), b.as_np_ndarray())
    assert mx_res.shape == np_res.shape
    assert_almost_equal(np_res, mx_res.asnumpy(), rtol=1e-5, atol=1e-5)
    mx_a = mx.sym.Variable("a")
    mx_b = mx.sym.Variable("b")
    mx_sym = mx.sym.np.dot(mx_a.as_np_ndarray(), mx_b.as_np_ndarray()).as_nd_ndarray()
    if (len(shape_a) > 0 and len(shape_b) > 0 and onp.prod(shape_a) > 0 and onp.prod(shape_b) > 0):
        check_numeric_gradient(mx_sym, {"a": a, "b": b}, numeric_eps=eps, rtol=1e-2, atol=1e-3)


@use_np
@pytest.mark.parametrize('shape_a,shape_b', [
    ((4, 5), (2, 3)),
    ((3, 4, 5), (6, ))
])
def test_np_dot_error(shape_a, shape_b):
    a = mx.nd.array(random.random()) if len(shape_a) == 0 else rand_ndarray(shape_a)
    b = mx.nd.array(random.random()) if len(shape_b) == 0 else rand_ndarray(shape_b)
    with pytest.raises(mx.base.MXNetError):
        mx_res = np.dot(a.as_np_ndarray(), b.as_np_ndarray())


@use_np
@pytest.mark.parametrize('shape', [(), (5,), (3, 3)])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_np_vdot(shape, dtype, hybridize):
    class TestVdot(HybridBlock):
        def __init__(self):
            super(TestVdot, self).__init__()

        def forward(self, a, b):
            return np.vdot(a, b)

    def vdot_backward(a, b):
        return [b, a]

    test_vdot = TestVdot()
    if hybridize:
        test_vdot.hybridize()
    a = rand_ndarray(shape=shape, dtype=dtype).as_np_ndarray()
    b = rand_ndarray(shape=shape, dtype=dtype).as_np_ndarray()
    a.attach_grad()
    b.attach_grad()

    np_out = onp.vdot(a.asnumpy(), b.asnumpy())
    with mx.autograd.record():
        mx_out = test_vdot(a, b)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)
    mx_out.backward()
    np_backward = vdot_backward(a.asnumpy(), b.asnumpy())
    assert_almost_equal(a.grad.asnumpy(), np_backward[0], rtol = 1e-2, atol=1e-2)
    assert_almost_equal(b.grad.asnumpy(), np_backward[1], rtol = 1e-2, atol=1e-2)

    # Test imperative once again
    mx_out = np.vdot(a, b)
    np_out = onp.vdot(a.asnumpy(), b.asnumpy())
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    # test numeric gradient
    if len(shape) > 0 and onp.prod(shape) > 0:
        a_sym = mx.sym.Variable("a").as_np_ndarray()
        b_sym = mx.sym.Variable("b").as_np_ndarray()
        mx_sym = mx.sym.np.vdot(a_sym, b_sym).as_nd_ndarray()
        check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
          rtol=1e-1, atol=1e-1, dtype=dtype)


@use_np
@pytest.mark.parametrize('a_shape,b_shape', [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((3,), (2, 3))
])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_np_inner(a_shape, b_shape, dtype, hybridize):
    class TestInner(HybridBlock):
        def __init__(self):
            super(TestInner, self).__init__()

        def forward(self, a, b):
            return np.inner(a, b)

    def inner_backward(a, b):
        a_axes_summed = [a.ndim - 1]
        b_axes_summed = [b.ndim - 1]

        a_axes_remained = []
        for i in range(a.ndim):
            if not (i in a_axes_summed):
                a_axes_remained.append(i)
        a_axes = a_axes_remained[:] + a_axes_summed[:]

        b_axes_remained = []
        for i in range(b.ndim):
            if not (i in b_axes_summed):
                b_axes_remained.append(i)
        b_axes = b_axes_summed[:] + b_axes_remained[:]

        ad1 = onp.prod([a.shape[i] for i in a_axes_remained]) if len(a_axes_remained) > 0 else 1
        ad2 = onp.prod([a.shape[i] for i in a_axes_summed]) if len(a_axes_summed) > 0 else 1
        bd1 = onp.prod([b.shape[i] for i in b_axes_summed]) if len(b_axes_summed) > 0 else 1
        bd2 = onp.prod([b.shape[i] for i in b_axes_remained]) if len(b_axes_remained) > 0 else 1

        out_grad = onp.ones((ad1, bd2))

        new_a = onp.transpose(a, a_axes)
        new_a_shape = new_a.shape[:]
        new_a = new_a.reshape((ad1, ad2))
        new_b = onp.transpose(b, b_axes)
        new_b_shape = new_b.shape[:]
        new_b = new_b.reshape((bd1, bd2))

        reverse_a_axes = [0 for i in a_axes]
        for i in range(len(a_axes)):
            reverse_a_axes[a_axes[i]] = i

        reverse_b_axes = [0 for i in b_axes]
        for i in range(len(b_axes)):
            reverse_b_axes[b_axes[i]] = i

        grad_b = onp.dot(new_a.T, out_grad).reshape(new_b_shape)
        grad_b = onp.transpose(grad_b, reverse_b_axes)
        grad_a = onp.dot(out_grad, new_b.T).reshape(new_a_shape)
        grad_a = onp.transpose(grad_a, reverse_a_axes)

        return [grad_a, grad_b]

    test_inner = TestInner()
    if hybridize:
        test_inner.hybridize()
    a = rand_ndarray(shape=a_shape, dtype=dtype).as_np_ndarray()
    b = rand_ndarray(shape=b_shape, dtype=dtype).as_np_ndarray()
    a.attach_grad()
    b.attach_grad()

    np_out = onp.inner(a.asnumpy(), b.asnumpy())
    with mx.autograd.record():
        mx_out = test_inner(a, b)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)
    mx_out.backward()
    np_backward = inner_backward(a.asnumpy(), b.asnumpy())
    assert_almost_equal(a.grad.asnumpy(), np_backward[0], rtol = 1e-2, atol=1e-2)
    assert_almost_equal(b.grad.asnumpy(), np_backward[1], rtol = 1e-2, atol=1e-2)

    # Test imperative once again
    mx_out = np.inner(a, b)
    np_out = onp.inner(a.asnumpy(), b.asnumpy())
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    # test numeric gradient
    a_sym = mx.sym.Variable("a").as_np_ndarray()
    b_sym = mx.sym.Variable("b").as_np_ndarray()
    mx_sym = mx.sym.np.inner(a_sym, b_sym).as_nd_ndarray()
    check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
      rtol=1e-1, atol=1e-1, dtype=dtype)


@use_np
@pytest.mark.parametrize('a_shape,b_shape', [
    ((3,), (3,)),
    ((2, 3), (6,)),
    ((6,), (2, 3))
])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_np_outer(a_shape, b_shape, dtype, hybridize):
    class TestOuter(HybridBlock):
        def __init__(self):
            super(TestOuter, self).__init__()

        def forward(self, a, b):
            return np.outer(a, b)

    test_outer = TestOuter()
    if hybridize:
        test_outer.hybridize()
    a = rand_ndarray(shape=a_shape, dtype=dtype).as_np_ndarray()
    b = rand_ndarray(shape=b_shape, dtype=dtype).as_np_ndarray()
    a.attach_grad()
    b.attach_grad()

    np_out = onp.outer(a.asnumpy(), b.asnumpy())
    with mx.autograd.record():
        mx_out = test_outer(a, b)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
    mx_out.backward()

    # Test imperative once again
    mx_out = np.outer(a, b)
    np_out = onp.outer(a.asnumpy(), b.asnumpy())
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    # test numeric gradient
    a_sym = mx.sym.Variable("a").as_np_ndarray()
    b_sym = mx.sym.Variable("b").as_np_ndarray()
    mx_sym = mx.sym.np.outer(a_sym, b_sym).as_nd_ndarray()
    check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                           rtol=1e-1, atol=1e-1, dtype=dtype)


@use_np
@pytest.mark.parametrize('shape_a,shape_b', [
    ((3,), (3,)),
    ((3, 4), (4, 5)),
    ((3, 0), (0, 4)),
    ((4, 5), (5,)),
    ((3, 4, 5), (5,)),
    ((5,), (5, 2)),
    ((2,), (4, 2, 3)),
    ((2, 1, 3, 4, 5), (5, 2)),
    ((1, 3, 5, 4), (1, 4, 3)),
    ((3, 5, 4), (2, 1, 4, 3)),
    ((3, 4), (1, 5, 4, 3))
])
@pytest.mark.parametrize('grad_req_a', ['write', 'add', 'null'])
@pytest.mark.parametrize('grad_req_b', ['write', 'add', 'null'])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_np_matmul(shape_a, shape_b, grad_req_a, grad_req_b,
                   dtype, hybridize):
    class TestMatmul(HybridBlock):
        def __init__(self):
            super(TestMatmul, self).__init__()

        def forward(self, a, b):
            return np.matmul(a, b)

    def matmul_backward(a, b):
        def ShapeInfer(mat_a, mat_b):
            if mat_a.ndim == 1:
                mat_a = mat_a.reshape((1, mat_a.size))
            if mat_b.ndim == 1:
                mat_b = mat_b.reshape((mat_b.size, 1))
            ndim = max(mat_a.ndim, mat_b.ndim)
            newshape_a = list(onp.array(mat_a, ndmin=ndim).shape)
            newshape_b = list(onp.array(mat_b, ndmin=ndim).shape)
            if ndim >= 3:
                pre_shape = onp.fmax(newshape_a[ndim - 3::-1], newshape_b[ndim - 3::-1])
                newshape_a[ndim - 3::-1] = pre_shape
                newshape_b[ndim - 3::-1] = pre_shape
            else:
                pre_shape = onp.array([])
            out_shape = onp.append(pre_shape[::-1].astype(onp.int64), [newshape_a[ndim - 2], newshape_b[ndim - 1]])
            return [ndim, newshape_a, newshape_b, out_shape]

        def ShapeReduce(mat, shape, is_b=False):
            ndim = mat.ndim
            if is_b and len(shape) == 1:
                rng = onp.arange(ndim - 2)
            else:
                pre_len = ndim - len(shape)
                in_pre = onp.array(mat.shape[pre_len : ndim - 2])
                out_pre = onp.array(shape[:len(shape) - 2])
                diff = onp.nonzero(in_pre != out_pre)[0] + pre_len
                rng = onp.append(onp.arange(ndim - len(shape)), diff)
            mat = onp.sum(mat, axis=tuple(rng))
            return mat.reshape(shape)

        a_shape = a.shape
        b_shape = b.shape
        [ndim, newshape_a, newshape_b, out_shape] = ShapeInfer(a, b)
        new_a = onp.broadcast_to(a, newshape_a)
        if len(b_shape) == 1:
            new_b = onp.broadcast_to(b.reshape((b.size, 1)), newshape_b)
        else:
            new_b = onp.broadcast_to(b, newshape_b)

        ad1 = new_a.shape[ndim - 2]
        ad2 = new_a.shape[ndim - 1]
        bd1 = new_b.shape[ndim - 2]
        bd2 = new_b.shape[ndim - 1]
        a_T = onp.moveaxis(new_a, [ndim - 2, ndim - 1], [ndim - 1, ndim - 2])
        b_T = onp.moveaxis(new_b, [ndim - 2, ndim - 1], [ndim - 1, ndim - 2])
        out_grad = onp.ones(out_shape)
        grad_b = onp.matmul(a_T, out_grad)
        grad_b = ShapeReduce(grad_b, b_shape, is_b=True)
        grad_a = onp.matmul(out_grad, b_T)
        grad_a = ShapeReduce(grad_a, a_shape)
        return [grad_a, grad_b]

    eps = 1E-4
    test_matmul = TestMatmul()
    if hybridize:
        test_matmul.hybridize()
    np_a = onp.random.uniform(-1.0, 1.0, shape_a).astype(dtype)
    np_a[abs(np_a) < eps] = 2 * eps
    np_b = onp.random.uniform(-1.0, 1.0, shape_b).astype(dtype)
    np_b[abs(np_b) < eps] = 2 * eps
    a = mx.np.array(np_a, dtype=dtype)
    a.attach_grad(grad_req=grad_req_a)
    b = mx.np.array(np_b, dtype=dtype)
    b.attach_grad(grad_req=grad_req_b)

    np_out = onp.matmul(np_a, np_b)
    with mx.autograd.record():
        mx_out = test_matmul(a, b)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(np_out, mx_out.asnumpy(), rtol=eps, atol=eps)

    if grad_req_a != 'null' or grad_req_b != 'null':
        mx_out.backward()
        np_backward = matmul_backward(np_a, np_b)
        if grad_req_a == 'null':
            assert a.grad is None
        else:
            assert_almost_equal(a.grad.asnumpy(), np_backward[0], rtol = eps, atol=eps)
        if grad_req_b == 'null':
            assert b.grad is None
        else:
            assert_almost_equal(b.grad.asnumpy(), np_backward[1], rtol = eps, atol=eps)

    mx_out = np.matmul(a, b)
    np_out = onp.matmul(np_a, np_b)
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=eps, atol=eps)


@pytest.mark.parametrize('shape_a,shape_b', [
    ((1,), (2,)),            # mismatched vector vector
    ((2, 1,), (2,)),         # mismatched matrix vector
    ((2,), (1, 2)),          # mismatched vector matrix
    ((1, 2), (3, 1)),        # mismatched matrix matrix
    ((1,), ()),              # vector scalar
    ((), (1,)),              # scalar vector
    ((1, 1), ()),            # matrix scalar
    ((), (1, 1)),            # scalar matrix
    ((2, 2, 1), (3, 1, 2)),  # cannot broadcast
])
def test_np_matmul_error(shape_a, shape_b):
    a = np.random.uniform(size=shape_a)
    b = np.random.uniform(size=shape_b)
    with pytest.raises(MXNetError):
        np.matmul(a, b)


@use_np
@pytest.mark.parametrize('a_shape,b_shape', [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((2, 3, 4), (2,)),
    ((3, 2), ())
])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_kron(a_shape, b_shape, dtype, hybridize):
    def np_kron_backward(ograd, a, b):
        ndim = ograd.ndim
        # Make ndim equal
        if ndim > a.ndim:
            a = a.reshape((1,)*(ndim - a.ndim) + a.shape)
        else:
            b = b.reshape((1,)*(ndim - b.ndim) + b.shape)
        assert(a.ndim == b.ndim)

        # Compute agrad
        agrad = onp.zeros(a.shape)
        for i in range(a.size):
            ia = onp.asarray(onp.unravel_index(i, a.shape))
            for j in range(b.size):
                jb = onp.asarray(onp.unravel_index(j, b.shape))
                k = ia * onp.asarray(b.shape) + jb
                agrad[tuple(ia)] += ograd[tuple(k)] * b[tuple(jb)]
        # Compute bgrad
        bgrad = onp.zeros(b.shape)
        for j in range(b.size):
            jb = onp.asarray(onp.unravel_index(j, b.shape))
            for i in range(a.size):
                ia = onp.asarray(onp.unravel_index(i, a.shape))
                k = ia * onp.asarray(b.shape) + jb
                bgrad[tuple(jb)] += ograd[tuple(k)] * a[tuple(ia)]
        return [agrad, bgrad]

    class TestKron(HybridBlock):
        def __init__(self):
            super(TestKron, self).__init__()

        def forward(self, a, b):
            return np.kron(a, b)

    test_kron = TestKron()
    if hybridize:
        test_kron.hybridize()
    a = rand_ndarray(shape=a_shape, dtype=dtype).as_np_ndarray()
    b = rand_ndarray(shape=b_shape, dtype=dtype).as_np_ndarray()
    a.attach_grad()
    b.attach_grad()

    np_out = onp.kron(a.asnumpy(), b.asnumpy())
    with mx.autograd.record():
        mx_out = test_kron(a, b)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)
    mx_out.backward()

    # Test imperative once again
    mx_out = np.kron(a, b)
    np_out = onp.kron(a.asnumpy(), b.asnumpy())
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)

    # test numeric gradient
    a_sym = mx.sym.Variable("a").as_np_ndarray()
    b_sym = mx.sym.Variable("b").as_np_ndarray()
    mx_sym = mx.sym.np.kron(a_sym, b_sym).as_nd_ndarray()
    check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                           rtol=1e-2, atol=1e-2, dtype=dtype)

    # test gradient via backward implemented by numpy
    np_backward = np_kron_backward(onp.ones(np_out.shape, dtype = dtype), a.asnumpy(), b.asnumpy())
    assert_almost_equal(a.grad.asnumpy(), np_backward[0], rtol=1e-2, atol=1e-2)
    assert_almost_equal(b.grad.asnumpy(), np_backward[1], rtol=1e-2, atol=1e-2)


@use_np
@pytest.mark.parametrize('shape', [rand_shape_nd(4, dim=4), (4, 0, 4, 0)])
@pytest.mark.parametrize('axis', [0, 1, 2, 3, (), None])
@pytest.mark.parametrize('keepdims', [True, False])
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int8', 'int32', 'int64'])
@pytest.mark.parametrize('itype,acc_type', [
    ('float16', 'float32'),
    ('float32', 'float64'),
    ('float64', 'float64'),
    ('int8', 'int32'),
    ('int32', 'int64'),
    ('int64', 'int64'),
    ('bool', 'int64')
])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_sum(shape, axis, keepdims, itype, acc_type, dtype, hybridize):
    class TestSum(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestSum, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def forward(self, a, *args, **kwargs):
            return np.sum(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    class TestSumConv(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestSumConv, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def forward(self, a, *args, **kwargs):
            return a.sum(axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' in dtype

    is_windows = sys.platform.startswith('win')
    if (is_int(dtype) and not is_int(itype)) or (is_windows and is_int(itype))\
            or (itype == 'bool' and\
                (dtype not in ('float32', 'float64', 'int32', 'int64') or is_windows)):
        return
    # test gluon
    test_sum = TestSum(axis=axis, dtype=dtype, keepdims=keepdims)
    test_sum_conv = TestSumConv(axis=axis, dtype=dtype, keepdims=keepdims)
    if hybridize:
        test_sum.hybridize()
        test_sum_conv.hybridize()
    if is_int(itype):
        x = onp.random.randint(-128, 128, shape, dtype=itype)
        x = np.array(x)
    elif itype == 'bool':
        x = onp.random.randint(0, 2, shape) < 1
        x = np.array(x, dtype='bool')
    else:
        x = np.random.uniform(-1.0, 1.0, size=shape, dtype=itype)
    expected_ret = onp.sum(x.asnumpy(), axis=axis, dtype=acc_type, keepdims=keepdims)
    expected_ret = expected_ret.astype(dtype)
    if itype == 'bool':
        if is_op_runnable() and (not is_windows):  # special handling of boolean ndarray
            y = test_sum(x)
            y_conv = test_sum_conv(x)
            assert y.dtype == expected_ret.dtype
            assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-4, atol=1e-5,
                                use_broadcast=False)
            assert y_conv.dtype == expected_ret.dtype
            assert_almost_equal(y_conv.asnumpy(), expected_ret, rtol=1e-4, atol=1e-5,
                                use_broadcast=False)
        return

    x.attach_grad()
    with mx.autograd.record():
        y = test_sum(x)
        y_conv = test_sum_conv(x)
    assert y.shape == expected_ret.shape
    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                        atol=1e-5 if dtype == 'float16' else 1e-5, use_broadcast=False)
    assert y_conv.shape == expected_ret.shape
    assert_almost_equal(y_conv.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                        atol=1e-5 if dtype == 'float16' else 1e-5, use_broadcast=False)
    y.backward()
    assert same(x.grad.asnumpy(), onp.ones(shape=x.shape, dtype=x.dtype))

    # test numeric
    if itype == 'float32' and dtype == 'float32' and shape != (4, 0, 4, 0):
        x_sym = mx.sym.Variable("x").as_np_ndarray()
        mx_sym = mx.sym.np.sum(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
        check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                numeric_eps=1e-3, rtol=1e-2, atol=1e-3, dtype=onp.float32)

    # test imperative
    mx_out = np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
    np_out = onp.sum(x.asnumpy(), axis=axis, dtype=acc_type, keepdims=keepdims).astype(dtype)
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@use_np
@pytest.mark.parametrize('bool_agg', ['all', 'any'])
@pytest.mark.parametrize('shape', [
    (), (5, ), (10, ), (2, 5), (5, 5), (10, 10),
    (4, 4, 4), (4, 6, 9), (6, 6, 6), (6, 0, 5),
    (7, 8, 9, 10), (7, 9, 11, 13), (0, 7, 7, 5)
])
@pytest.mark.parametrize('axis', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('keepdim', [True, False])
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool])
def test_np_bool_agg(bool_agg, shape, axis, keepdim, dtype, hybridize):
    class TestOp(HybridBlock):
        def __init__(self, axis=None, keepdims=False) :
            super(TestOp, self).__init__()
            self._axis = axis
            self._keepdims = keepdims

        def forward(self, a):
            return getattr(np, bool_agg)(a, axis=self._axis, keepdims=self._keepdims)

    ndim = len(shape)
    samples = random.randint(0, ndim)
    axis = None if not axis else tuple(random.sample([i for i in range(0, ndim)], samples))
    x = np.random.normal(0, 5.0, size=shape).astype(dtype)
    test_op = TestOp(axis=axis, keepdims=keepdim)
    if hybridize:
        test_op.hybridize()
    y = test_op(x)
    expected_ret = getattr(onp, bool_agg)(x.asnumpy(), axis=axis, keepdims=keepdim)
    assert_almost_equal(y.asnumpy(), expected_ret)

    # test imperative
    mx_outs = getattr(np, bool_agg)(x, axis=axis, keepdims=keepdim)
    np_outs = getattr(onp, bool_agg)(x.asnumpy(), axis=axis, keepdims=keepdim)
    assert_almost_equal(mx_outs.asnumpy(), np_outs)


@use_np
@pytest.mark.parametrize('func', ['max', 'min'])
@pytest.mark.parametrize('in_data_dim', [2, 3, 4])
@pytest.mark.parametrize('itype', ['float16', 'float32', 'float64', 'int'])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('keepdims', [True, False])
def test_np_max_min(func, in_data_dim, itype, keepdims, hybridize):
    class TestOp(HybridBlock):
        def __init__(self, axis=None, keepdims=False):
            super(TestOp, self).__init__()
            self._axis = axis
            self._keepdims = keepdims

        def forward(self, a, *args, **kwargs):
            return getattr(a, func)(axis=self._axis, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' == dtype

    def get_grad(axis, func_name):
        index = -1 if func_name == 'max' else 0
        if axis == ():
            return onp.ones((2,3,4,5))
        else:
            temp = onp.zeros((2,3,4,5))
            if axis == 0:
                temp[index,:,:,:] = 1
                return temp
            elif axis == 1:
                temp[:,index,:,:] = 1
                return temp
            elif axis == 2:
                temp[:,:,index,:] = 1
                return temp
            elif (axis == 3 or axis == -1):
                temp[:,:,:,index] = 1
                return temp
            elif not axis:
                temp[index,index,index,index] = 1
                return temp
            raise ValueError('axis should be int or None or ()')

    shape = rand_shape_nd(in_data_dim, dim=3)
    for axis in ([i for i in range(in_data_dim)] + [(), None] + [-1]):
        test_gluon = TestOp(axis=axis, keepdims=keepdims)
        if hybridize:
            test_gluon.hybridize()
        if is_int(itype):
            x = np.arange(120).reshape((2, 3, 4, 5))
        else:
            x = np.random.uniform(-1.0, 1.0, size=shape, dtype=itype)
        x.attach_grad()
        ref_op = getattr(onp, 'a'+func)
        expected_ret = ref_op(x.asnumpy(), axis=axis, keepdims=keepdims)
        with mx.autograd.record():
            y = test_gluon(x)
        assert y.shape == expected_ret.shape
        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if itype == 'float16' else 1e-3,
                            atol=1e-5 if itype == 'float16' else 1e-5)
        y.backward()
        # only check the gradient with hardcoded input
        if is_int(itype):
            assert same(x.grad.asnumpy(), get_grad(axis, func)), \
                'x={}\ny={}\nx.grad={}\nnumpy={}'.format(x.asnumpy(), y.asnumpy(), x.grad.asnumpy(), get_grad(axis))

        # test imperative
        mx_out = getattr(np, func)(x, axis=axis, keepdims=keepdims)
        np_out = ref_op(x.asnumpy(), axis=axis, keepdims=keepdims)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

@use_np
@pytest.mark.parametrize('func', ['max', 'min'])
@pytest.mark.parametrize('shape,exception', [
    ((), False),
    ((0), True),
    ((2, 0), True),
    ((0, 2, 1), True)
])
def test_np_max_min_error(func, shape, exception):
    # test zero and zero dim
    def _test_np_exception(func, shape, dim):
        x = np.random.uniform(-1.0, 1.0, shape)
        out = getattr(x, func)()
        assert out.ndim == dim, 'dimension mismatch, output.ndim={}, dim={}'.format(output.ndim, dim)
    dim = 0
    if exception:
        assertRaises(MXNetError, _test_np_exception, func, shape, dim)
    else:
        _test_np_exception(func, shape, dim)


@use_np
@pytest.mark.parametrize('a_shape,w_shape,axes', [
    ((3, 5), (3, 5), None),
    ((4, 5, 6), (4, 5, 6), (0, 2)),
    ((3,), (3,), 0),
    ((2, 3), (3,), 1),
    ((2, 3, 4), (2,), 0),
    ((2, 3, 4), (3,), 1),
    ((2, 3, 4), (4,), -1),
    ((2, 3, 4, 5), (5,), 3)
])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('is_weighted', [True, False])
@pytest.mark.parametrize('returned', [True, False])
@pytest.mark.parametrize('req_a', ['null', 'add', 'write'])
@pytest.mark.flaky
def test_np_average(a_shape, w_shape, axes, is_weighted, req_a,
                    hybridize, returned, dtype):
    class TestAverage(HybridBlock):
        def __init__(self, axis=None, returned=False):
            super(TestAverage, self).__init__()
            # necessary initializations
            self._axis = axis
            self._returned = returned

        def forward(self, a, weights):
            return np.average(a, weights=weights, axis=self._axis, returned=self._returned)

    def avg_backward(a, w, avg, axes, init_a_grad=None, init_w_grad=None):
        # avg = sum(a * w) / sum(w)
        if axes is not None and not isinstance(axes, tuple) and axes < 0:
            axes += a.ndim
        if w is None:
            a_grad = onp.ones(shape=a.shape, dtype=a.dtype)/(a.size/avg.size)
            if init_a_grad is not None:
                a_grad += init_a_grad.asnumpy()
            return [a_grad, None]
        onedim = a.ndim != w.ndim
        if onedim:
            new_shape = [a.shape[i] if i == axes else 1 for i in range(a.ndim)]
            w = w.reshape(new_shape)
            w = onp.broadcast_to(w, a.shape)

        # partial a = w / sum(w)
        # partial w = (a*sum(w) - sum(a*w)) / (sum(w) * sum(w))
        scl = onp.sum(w, axis=axes, keepdims=True)
        a_grad = onp.divide(w, scl)
        w_grad = onp.divide(a*scl-onp.sum(a*w, axis=axes, keepdims=True), scl*scl)

        if onedim:
            axis = list(range(a.ndim))
            axis.remove(axes)
            w_grad = onp.sum(w_grad, axis=tuple(axis))
        if init_a_grad is not None:
            a_grad += init_a_grad.asnumpy()
        if init_w_grad is not None:
            w_grad += init_w_grad.asnumpy()
        return [a_grad, w_grad]

    if req_a == 'null' and not is_weighted:
        return
    rtol, atol = 1e-3, 1e-4
    test_average = TestAverage(axes, returned)
    if hybridize:
        test_average.hybridize()
    a = np.random.uniform(-1.0, 1.0, size=a_shape, dtype=dtype)
    a.attach_grad(req_a)
    init_a_grad = np.random.uniform(-1.0, 1.0, size=a_shape, dtype=dtype) if req_a == 'add' else None
    init_w_grad = None
    req_w = req_a
    w, np_w = None, None
    if is_weighted:
        w = np.random.uniform(-1.0, 1.0, size=w_shape, dtype=dtype)
        if req_a == 'null':
            req_w = random.choice(['add', 'write'])
        w.attach_grad(req_w)
        if req_w == 'add':
            init_w_grad = np.random.uniform(-1.0, 1.0, size=w_shape, dtype=dtype)
        np_w = w.asnumpy()
    np_out = onp.average(a.asnumpy(), axis=axes, weights=np_w, returned=returned)
    with mx.autograd.record():
        mx_out = test_average(a, w)
    if returned:
        np_out, np_sum_of_weights = np_out
        mx_out, mx_sum_of_weights = mx_out
        assert_almost_equal(mx_sum_of_weights.asnumpy(), np_sum_of_weights, rtol=rtol, atol=atol)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
    if req_a == 'add':
        a.grad[:] = init_a_grad
    if is_weighted and req_w == 'add':
        w.grad[:] = init_w_grad
    mx_out.backward()
    # Code to get reference backward value
    a_grad, w_grad = avg_backward(a.asnumpy(), np_w, np_out, axes, init_a_grad, init_w_grad)
    if is_weighted:
        assert_almost_equal(w.grad.asnumpy(), w_grad, rtol=rtol*10, atol=atol*10)
    if req_a == 'null':
        assert a.grad is None
    else:
        assert_almost_equal(a.grad.asnumpy(), a_grad, rtol=rtol, atol=atol)

    # Test imperative once again
    np_out = onp.average(a.asnumpy(), weights=np_w, axis=axes, returned=returned)
    mx_out = np.average(a, weights=w, axis=axes, returned=returned)
    if returned:
        np_out, np_sum_of_weights = np_out
        mx_out, mx_sum_of_weights = mx_out
        assert_almost_equal(mx_sum_of_weights.asnumpy(), np_sum_of_weights, rtol=rtol, atol=atol)
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_mean():
    class TestMean(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestMean, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def forward(self, a, *args, **kwargs):
            return a.mean(axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' in dtype

    is_windows = sys.platform.startswith('win')
    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    acc_type = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64',
                'bool': 'int64', 'int8': 'int32', 'int32': 'int64', 'int64': 'int64'}
    ft_types = ['float16', 'float32', 'float64']
    it_types = ['bool', 'int8', 'int32', 'int64']
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype, dtype in itertools.product(ft_types, [None] + ft_types + it_types):
                    if dtype == 'bool':
                        continue
                    # test gluon
                    test_mean = TestMean(axis=axis, dtype=dtype, keepdims=keepdims)
                    if hybridize:
                        test_mean.hybridize()
                    x = np.random.uniform(-1.0, 1.0, size=shape).astype(itype)
                    x = x.as_np_ndarray()
                    x.attach_grad()

                    expected_ret = onp.mean(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims)
                    expected_ret = expected_ret.astype(dtype)
                    with mx.autograd.record():
                        y = test_mean(x)
                    assert y.shape == expected_ret.shape
                    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                        atol=1e-5 if dtype == 'float16' else 1e-5)

                    y.backward()
                    N = x.size / y.size
                    assert same(x.grad.asnumpy(), onp.ones(shape=x.shape, dtype=x.dtype) / N)

                    # test numeric
                    if itype == 'float32' and dtype == 'float32':
                        x_sym = mx.sym.Variable("x").as_np_ndarray()
                        mx_sym = mx.sym.np.mean(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                        check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                               numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=onp.float32)

                    # test imperative
                    mx_out = np.mean(x, axis=axis, dtype=dtype, keepdims=keepdims)
                    np_out = onp.mean(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                for itype, dtype in itertools.product(it_types, [None] + ft_types + it_types):
                    if dtype == 'bool':
                        continue
                    # test gluon
                    test_mean = TestMean(axis=axis, dtype=dtype, keepdims=keepdims)
                    if hybridize:
                        test_mean.hybridize()

                    if itype == 'bool':
                        x = np.array(onp.random.uniform(size=shape) > 0.5)
                    else:
                        x = np.random.uniform(-128, 127, size=shape).astype(itype)

                    expected_ret = onp.mean(x.asnumpy(), axis=axis, dtype=dtype, keepdims=keepdims)

                    if itype == 'bool':
                        if is_op_runnable() and (not is_windows) and dtype not in ['float16', 'int8']:  # special handling of boolean ndarray
                            y = test_mean(x)
                            assert y.shape == expected_ret.shape
                            assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                                atol=1e-5 if dtype == 'float16' else 1e-5)
                        continue

                    y = test_mean(x)
                    assert y.shape == expected_ret.shape
                    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                        atol=1e-5 if dtype == 'float16' else 1e-5)

                    # test imperative
                    mx_out = np.mean(x, axis=axis, dtype=dtype, keepdims=keepdims)
                    np_out = onp.mean(x.asnumpy(), axis=axis, dtype=dtype, keepdims=keepdims).astype(dtype)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_moment():
    class TestMoment(HybridBlock):
        def __init__(self, name, axis=None, dtype=None, keepdims=False, ddof=0):
            super(TestMoment, self).__init__()
            self._moment_name = name
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims
            self._ddof = ddof

        def forward(self, a, *args, **kwargs):
            return getattr(a, self._moment_name)(axis=self._axis, dtype=self._dtype,
                                                 keepdims=self._keepdims, ddof=self._ddof)

    def is_int(dtype):
        return 'int' in dtype

    def legalize_shape(shape):
        shape_ = list(shape)
        for i in range(len(shape_)):
            shape_[i] += 1
        return tuple(shape_)

    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    shape = legalize_shape(shape)
    acc_type = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64',
                'int8': 'float64', 'int32': 'float64', 'int64': 'float64'}

    for name in ['var', 'std']:
        for hybridize in [False, True]:
            for ddof in [0, 1]:
                for keepdims in [True, False]:
                    for axis in ([i for i in range(in_data_dim)] + [(), None]):
                        for itype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                            for dtype in ['float16', 'float32', 'float64']:
                                if is_int(dtype) and not is_int(itype) or is_int(itype) and is_int(dtype):
                                    continue
                                atol = 3e-4 if itype == 'float16' or dtype == 'float16' else 1e-5
                                rtol = 1e-2 if itype == 'float16' or dtype == 'float16' else 1e-3
                                # test gluon
                                test_moment = TestMoment(name, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
                                if hybridize:
                                    test_moment.hybridize()
                                if is_int(itype):
                                    x = onp.random.randint(-16, 16, shape, dtype=itype)
                                    x = mx.nd.array(x)
                                else:
                                    x = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype=itype)
                                x = x.as_np_ndarray()
                                x.attach_grad()
                                expected_ret = getattr(onp, name)(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims, ddof=ddof)
                                expected_ret = expected_ret.astype(dtype)
                                y = test_moment(x)
                                assert y.shape == expected_ret.shape
                                assert_almost_equal(y.asnumpy(), expected_ret, rtol=rtol, atol=atol, use_broadcast=False, equal_nan=True)

                                # test imperative
                                mx_out = getattr(np, name)(x, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
                                np_out = getattr(onp, name)(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims, ddof=ddof).astype(dtype)
                                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol, use_broadcast=False, equal_nan=True)


@use_np
def test_np_shape():
    shapes = [
        (),
        (0, 1),
        (2, 3),
        (2, 3, 4),
    ]

    for shape in shapes:
        mx_a = np.random.uniform(size=shape)
        np_a = onp.random.uniform(size=shape)

        mx_shape = np.shape(mx_a)
        np_shape = onp.shape(np_a)

        assert mx_shape == np_shape


@use_np
@pytest.mark.parametrize('config', [
    (0.0, 1.0, 10),
    (-2, 4, 30),
    (5.234324, 8.98324, 324),
    (2, 10, 100)
])
@pytest.mark.parametrize('dtype', ['int32', 'float16', 'float32', 'float64', None])
@pytest.mark.parametrize('endpoint', [True, False])
@pytest.mark.parametrize('retstep', [True, False])
def test_np_linspace(config, dtype, endpoint, retstep):
    if isinstance(config, tuple):
        mx_ret = np.linspace(*config, endpoint=endpoint, retstep=retstep, dtype=dtype)
        np_ret = onp.linspace(*config, endpoint=endpoint, retstep=retstep, dtype=dtype)
    else:
        mx_ret = np.linspace(config, endpoint=endpoint, retstep=retstep, dtype=dtype)
        np_ret = onp.linspace(config, endpoint=endpoint, retstep=retstep, dtype=dtype)
    if retstep:
        assert_almost_equal(mx_ret[0].asnumpy(), np_ret[0], atol=1e-3, rtol=1e-5)
        assert same(mx_ret[1], np_ret[1])
    else:
        assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-3, rtol=1e-5)

@use_np
@pytest.mark.parametrize('config', [
    (0.0, 1.0, 10),
    (-2, 4, 30),
    (5.234324, 8.98324, 324),
    (2, 10, 100)
])
@pytest.mark.parametrize('dtype', ['int32', 'float16', 'float32', 'float64', None])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('endpoint', [True, False])
def test_np_linspace_gluon(config, dtype, endpoint, hybridize):
    class TestLinspace(HybridBlock):
        def __init__(self, start, stop, num=50, endpoint=None, retstep=False, dtype=None, axis=0):
            super(TestLinspace, self).__init__()
            self._start = start
            self._stop = stop
            self._num = num
            self._endpoint = endpoint
            self._retstep = retstep
            self._dtype = dtype

        def forward(self, x):
            if self._retstep:
                raise ValueError("linspace didn't support retstep = True inside HybridBlock")
            else:
                return x + np.linspace(self._start, self._stop, num=self._num, \
                endpoint=self._endpoint, retstep=self._retstep, dtype=self._dtype)

    x = np.zeros(shape=(), dtype=dtype)
    if isinstance(config, tuple):
        net = TestLinspace(*config, endpoint=endpoint, dtype=dtype)
        np_out = onp.linspace(*config, endpoint=endpoint, dtype=dtype)
    else:
        net = TestLinspace(config, endpoint=endpoint, dtype=dtype)
        np_out = onp.linspace(config, endpoint=endpoint, dtype=dtype)
    if hybridize:
        net.hybridize()
    mx_out = net(x)
    assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-3, rtol=1e-5)

@use_np
@pytest.mark.parametrize('config', [
    (0, 10, -1),
    (0, 1, 2.5)
])
def test_np_linspace_error(config):
    with pytest.raises(MXNetError):
        np.linspace(*config)


@use_np
def test_np_linspace_arange():
    # check linspace equivalent to arange
    for test_index in range(1000):
        assert_almost_equal(mx.np.linspace(0, test_index, test_index + 1).asnumpy(), onp.arange(test_index + 1))


@use_np
@pytest.mark.parametrize('config', [
    (0.0, 1.0, 20),
    (2, 8, 0),
    (22, 11, 1),
    (2.22, 9.99, 11),
    (4.99999, 12.11111111, 111)
])
@pytest.mark.parametrize('dtype', ['float32', 'float64', None])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('endpoint', [True, False])
@pytest.mark.parametrize('base', [0, 1, 5, 8, 10, 33])
def test_np_logspace(config, dtype, endpoint, hybridize, base):
    class TestLogspace(HybridBlock):
        def __init__(self, start, stop, num=50, endpoint=None, base=50.0, dtype=None, axis=0):
            super(TestLogspace, self).__init__()
            self._start = start
            self._stop = stop
            self._num = num
            self._endpoint = endpoint
            self._base = base
            self._dtype = dtype
            self.axis = axis

        def forward(self, x):
            return x + np.logspace(self._start, self._stop, self._num, self._endpoint, self._base, self._dtype, self.axis)

    x = np.zeros(shape=(), dtype=dtype)
    net = TestLogspace(*config, endpoint=endpoint, base=base, dtype=dtype)
    np_out = onp.logspace(*config, endpoint=endpoint, base=base, dtype=dtype)
    if hybridize:
        net.hybridize()
    mx_out = net(x)
    assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-3, rtol=1e-5)
    if dtype is not None:
        assert mx_out.dtype == np_out.dtype

    # Test imperative once again
    mx_ret = np.logspace(*config, endpoint=endpoint, base=base, dtype=dtype)
    np_ret = onp.logspace(*config, endpoint=endpoint, base=base, dtype=dtype)
    assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-3, rtol=1e-5)
    if dtype is not None:
        assert mx_out.dtype == np_out.dtype


@use_np
@pytest.mark.parametrize('start,end,step', [
    ([], [], None),
    ([], [], []),
    ([1], [4], None),
    ([1], [10], [3]),
    ([10], [0], [-2]),
    ([None], [None], [None]),
    ([None], [None], [-1]),
    ([10], [None], [-1]),
    ([1, 0, 3], [-2, 10, -4], [None, 2, 3]),
    ([-2, -3, -5, -6], [1, 3, 4, 5], None),
    ([-2, -3, -5, -6], [1, 3, 4, 5], [-1, -2, -3, -4]),
    ([2, -3, -5, -6], [2, 3, 4, 5], None),
    ([2, -3, -5, 5], [3, 3, 4, 5], None),
])
@pytest.mark.parametrize('hybridize', [True, False])
def test_npx_slice(start, end, step, hybridize):
    class TestSlice(HybridBlock):
        def __init__(self, begin, end, step):
            super(TestSlice, self).__init__()
            self._begin = begin
            self._end = end
            self._step = step

        def forward(self, a):
            return npx.slice(a, begin=self._begin, end=self._end, step=self._step)

    shape = (8, 16, 9, 9)
    np_array = onp.arange(onp.prod(shape), dtype='int32').reshape(shape)

    test_slice = TestSlice(begin=start, end=end, step=step)
    if hybridize:
        test_slice.hybridize()

    a = np.array(np_array, dtype=np_array.dtype)
    a.attach_grad()
    basic_index = tuple([
        slice(start[i], end[i], step[i]) if step is not None else slice(start[i], end[i])
        for i in range(len(start))
    ])
    expected_ret = np_array[basic_index]
    with mx.autograd.record():
        y = test_slice(a)

    assert same(y.asnumpy(), expected_ret)

    # test backward
    mx.autograd.backward(y)
    expected_grad = onp.zeros(shape)
    expected_grad[basic_index] = 1
    assert same(a.grad.asnumpy(), expected_grad)

@use_np
def test_npx_index_add():
    class TestIndexAdd(HybridBlock):
        def __init__(self):
            super(TestIndexAdd, self).__init__()

        def forward(self, a, ind, val):
            return npx.index_add(a, ind, val)

    def index_add_forward(a, ind, val, ind_ndim, ind_num):
        if val.dtype != a.dtype:
            val = val.astype(a.dtype)
        ind_arr = ind.transpose()
        if ind_arr.ndim == 0:
            ind_arr = onp.array([ind_arr])
        for i in range(ind_arr.shape[0]):
            t_ind = ind_arr[i]
            t_ind = tuple(t_ind.tolist()) if type(t_ind) is onp.ndarray else t_ind.tolist()
            if val.ndim + ind_ndim > a.ndim:
                t_val = val[tuple([0 if val.shape[0]==1 else i])]
                if type(t_val) is onp.ndarray and t_val.shape[0] == 1:
                    a[t_ind] += onp.squeeze(t_val, axis=0)
                else:
                    a[t_ind] += t_val
            else:
                a[t_ind] += val
        return a

    def index_add_bwd(out_grad, a_grad, ind, val_grad, ind_ndim, ind_num, grad_req_a, grad_req_val):
        if grad_req_a == 'add':
            init_a_grad = onp.array(a_grad)
        if grad_req_val == 'add':
            init_val_grad = onp.array(val_grad)
        a_grad = onp.zeros(a_grad.shape) + out_grad
        a_grad = a_grad.astype(a_grad.dtype)
        val_grad = onp.zeros(val_grad.shape).astype(val_grad.dtype)

        ind_arr = ind.transpose()
        if ind_arr.ndim == 0:
            ind_arr = onp.array([ind_arr])
        for i in range(ind_arr.shape[0]):
            t_ind = ind_arr[i]
            t_ind = tuple(ind_arr[i].tolist()) if type(ind_arr[i]) is onp.ndarray else ind_arr[i].tolist()
            if val_grad.ndim + ind_ndim > a_grad.ndim:
                idx = 0 if val_grad.shape[0]==1 else i
                t_grad = out_grad[t_ind]
                t_grad_shape = onp.array(t_grad.shape)
                val_grad_shape = onp.array(val_grad[idx].shape)
                if type(val_grad[idx]) is not onp.ndarray:
                    t_grad = onp.sum(t_grad)
                else:
                    is_not_equal = t_grad_shape - val_grad_shape
                    if onp.any(is_not_equal):
                        broadcast_dim = onp.nonzero(onp.where(is_not_equal, 1, 0))
                        t_grad = onp.sum(t_grad, axis=tuple(broadcast_dim[0].reshape(1, -1)[0]), keepdims=True)
                val_grad[idx] += t_grad
            else:
                t_grad = out_grad[t_ind]
                if type(val_grad) is not onp.ndarray or val_grad.shape == ():
                    t_grad = onp.sum(t_grad)
                else:
                    if type(t_grad) is onp.ndarray:
                        ext_dim = t_grad.ndim() - val_grad.ndim()
                        if ext_dim:
                            t_grad = onp.sum(t_grad, axis=tuple(onp.arange(ext_dim)))
                        t_grad_shape = onp.array(t_grad.shape)
                        val_grad_shape = onp.array(val_grad.shape)
                        is_not_equal = t_grad_shape - val_grad_shape
                        if onp.any(is_not_equal):
                            broadcast_dim = onp.nonzero(onp.where(is_not_equal, 1, 0))
                            t_grad = onp.sum(t_grad, axis=tuple(broadcast_dim.reshape(1, -1)[0]), keepdims=True)
                val_grad += t_grad
        if grad_req_a == 'add':
            a_grad += init_a_grad
        if grad_req_val == 'add':
            val_grad += init_val_grad
        return a_grad, val_grad

    # a.shape, ind.shape, val.shape, ind_ndim, ind_num
    configs = [((2, ), np.array(1, dtype=onp.int32), (1, ), 1, 1)]
    shape = tuple(onp.random.randint(1, 6, size=(4))) # a.shape
    for ind_ndim in range(1, 5): # ind.shape: (ind_ndim, ind_num)
        ind_num = onp.random.randint(1, 7)
        ind = []
        for ind_dim in range(ind_ndim):
            ind.append(onp.random.randint(0, shape[ind_dim], size=(ind_num)))
        ind = onp.array(ind).astype(onp.int32)
        # case: val is scalar
        configs.append(tuple([shape, ind, (), ind_ndim, ind_num]))
        for _ in range(1, 5 - ind_ndim):
            val_shape = [1 if onp.random.randint(0, 5)==0 else ind_num]
            for val_dim in range(ind_ndim, 4):
                val_shape.append(1 if onp.random.randint(0, 5)==0 else shape[val_dim])
            # case: val is tensor
            configs.append(tuple([shape, ind, tuple(val_shape), ind_ndim, ind_num]))

    dtypes = ['float32', 'float64', 'int32', 'int64']
    grad_req = ['write', 'null', 'add']
    for hybridize, grad_req_a, grad_req_val, dtype, indtype in \
        itertools.product([True, False], grad_req, grad_req, dtypes, ['int32', 'int64']):
        for a_shape, ind, val_shape ,ind_ndim, ind_num in configs:
            eps = 1e-3
            atype = dtype
            valtype = dtype
            test_index_add = TestIndexAdd()
            if hybridize:
                test_index_add.hybridize()
            a = mx.nd.random.uniform(-10.0, 10.0, shape=a_shape).as_np_ndarray().astype(atype)
            a.attach_grad(grad_req=grad_req_a)
            val = mx.nd.random.uniform(-10.0, 10.0, shape=val_shape).as_np_ndarray().astype(valtype)
            val.attach_grad(grad_req=grad_req_val)
            expected_ret = index_add_forward(a.asnumpy(), ind.astype(indtype), val.asnumpy(), ind_ndim, ind_num)
            with mx.autograd.record():
                mx_ret = test_index_add(a, np.array(ind).astype(indtype), val)
            assert mx_ret.shape == a.shape
            assert expected_ret.shape == a.shape
            assert mx_ret.dtype == a.dtype
            assert expected_ret.dtype == a.dtype
            assert_almost_equal(mx_ret.asnumpy(), expected_ret, rtol=eps, atol=eps)

            if atype not in ['float16', 'float32', 'float64'] or valtype not in ['float16', 'float32', 'float64']:
                continue
            if grad_req_a != 'null' or grad_req_val != 'null':
                init_a_grad = mx.nd.random.uniform(-10.0, 10.0, shape=a_shape).as_np_ndarray().astype(atype)
                init_val_grad = mx.nd.random.uniform(-10.0, 10.0, shape=val_shape).as_np_ndarray().astype(valtype)
                out_grad = mx.nd.random.uniform(-10.0, 10.0, shape=a_shape).as_np_ndarray().astype(atype)
                if grad_req_a == 'add':
                    if init_a_grad.ndim == 0:
                        a.grad[()] = init_a_grad.item()
                    else:
                        a.grad[:] = init_a_grad
                if grad_req_val == 'add':
                    if init_val_grad.ndim == 0:
                        val.grad[()] = init_val_grad.item()
                    else:
                        val.grad[:] = init_val_grad
                mx_ret.backward(out_grad)
                expected_bwd_a, expected_bwd_val = index_add_bwd(out_grad.asnumpy(), init_a_grad.asnumpy(), ind,
                                                                 init_val_grad.asnumpy(), ind_ndim, ind_num,
                                                                 grad_req_a, grad_req_val)
                if grad_req_a == 'null':
                    assert a.grad is None
                else:
                    assert_almost_equal(a.grad.asnumpy(), expected_bwd_a, rtol = eps, atol=eps)
                if grad_req_val == 'null':
                    assert val.grad is None
                else:
                    assert_almost_equal(val.grad.asnumpy(), expected_bwd_val, rtol = eps, atol=eps)

            mx_out = npx.index_add(a, np.array(ind).astype(indtype), val)
            assert_almost_equal(mx_out.asnumpy(), expected_ret, rtol=eps, atol=eps)


@use_np
def test_npx_index_update():
    class TestIndexUpdate(HybridBlock):
        def __init__(self):
            super(TestIndexUpdate, self).__init__()

        def forward(self, a, ind, val):
            return npx.index_update(a, ind, val)

    def check_index_update_forward(mx_ret, a, ind, val, ind_ndim, ind_num, eps):
        if val.dtype != a.dtype:
            val = val.astype(a.dtype)
        ind_arr = ind.transpose()
        if ind_arr.ndim == 0:
            ind_arr = onp.array([ind_arr])
        for i in range(ind_arr.shape[0]):
            t_ind = ind_arr[i]
            t_ind = tuple(t_ind.tolist()) if type(t_ind) is onp.ndarray else t_ind.tolist()
            if val.ndim + ind_ndim > a.ndim:
                t_val = val[tuple([0 if val.shape[0]==1 else i])]
                if type(t_val) is onp.ndarray and t_val.shape[0] == 1:
                    expect_tmp = onp.squeeze(t_val, axis=0)
                else:
                    expect_tmp = t_val
            else:
                expect_tmp = val
            mx_tmp = mx_ret[t_ind]
            close_pos = onp.where(onp.isclose(expect_tmp, mx_tmp, rtol=eps, atol=eps))
            if a[t_ind].ndim == 0:
                if close_pos[0].size == 1:
                    mx_ret[t_ind] = 0
                    a[t_ind] = 0
            else:
                mx_ret[t_ind][close_pos] = 0
                a[t_ind][close_pos] = 0
        assert_almost_equal(mx_ret, a, rtol=eps, atol=eps)

    def index_update_bwd(out_grad, a_grad, ind, val_grad, ind_ndim, ind_num, grad_req_a, grad_req_val):
        if grad_req_a == 'add':
            init_a_grad = onp.array(a_grad)
        if grad_req_val == 'add':
            init_val_grad = onp.array(val_grad)
        a_grad = onp.zeros(a_grad.shape) + out_grad
        a_grad = a_grad.astype(a_grad.dtype)
        val_grad = onp.zeros(val_grad.shape).astype(val_grad.dtype)

        ind_arr = ind.transpose()
        if ind_arr.ndim == 0:
            ind_arr = onp.array([ind_arr])
        for i in range(ind_arr.shape[0]):
            t_ind = ind_arr[i]
            t_ind = tuple(ind_arr[i].tolist()) if type(ind_arr[i]) is onp.ndarray else ind_arr[i].tolist()
            a_grad[t_ind] = 0
            if val_grad.ndim + ind_ndim > a_grad.ndim:
                idx = 0 if val_grad.shape[0]==1 else i
                t_grad = out_grad[t_ind]
                t_grad_shape = onp.array(t_grad.shape)
                val_grad_shape = onp.array(val_grad[idx].shape)
                if type(val_grad[idx]) is not onp.ndarray:
                    t_grad = onp.sum(t_grad)
                else:
                    is_not_equal = t_grad_shape - val_grad_shape
                    if onp.any(is_not_equal):
                        broadcast_dim = onp.nonzero(onp.where(is_not_equal, 1, 0))
                        t_grad = onp.sum(t_grad, axis=tuple(broadcast_dim[0].reshape(1, -1)[0]), keepdims=True)
                val_grad[idx] += t_grad
            else:
                t_grad = out_grad[t_ind]
                if type(val_grad) is not onp.ndarray or val_grad.shape == ():
                    t_grad = onp.sum(t_grad)
                else:
                    if type(t_grad) is onp.ndarray:
                        ext_dim = t_grad.ndim() - val_grad.ndim()
                        if ext_dim:
                            t_grad = onp.sum(t_grad, axis=tuple(onp.arange(ext_dim)))
                        t_grad_shape = onp.array(t_grad.shape)
                        val_grad_shape = onp.array(val_grad.shape)
                        is_not_equal = t_grad_shape - val_grad_shape
                        if onp.any(is_not_equal):
                            broadcast_dim = onp.nonzero(onp.where(is_not_equal, 1, 0))
                            t_grad = onp.sum(t_grad, axis=tuple(broadcast_dim.reshape(1, -1)[0]), keepdims=True)
                val_grad += t_grad
        if grad_req_a == 'add':
            a_grad += init_a_grad
        if grad_req_val == 'add':
            val_grad += init_val_grad
        return a_grad, val_grad

    # a.shape, ind.shape, val.shape, ind_ndim, ind_num
    configs = [((2, ), np.array(1, dtype=onp.int32), (1, ), 1, 1)]
    shape = tuple(onp.random.randint(1, 6, size=(4))) # a.shape
    for ind_ndim in range(1, 5): # ind.shape: (ind_ndim, ind_num)
        ind_num = onp.random.randint(1, 7)
        ind = []
        for ind_dim in range(ind_ndim):
            ind.append(onp.random.randint(0, shape[ind_dim], size=(ind_num)))
        ind = onp.array(ind).astype(onp.int32)
        # case: val is scalar
        configs.append(tuple([shape, ind, (), ind_ndim, ind_num]))
        for _ in range(1, 5 - ind_ndim):
            val_shape = [1 if onp.random.randint(0, 5)==0 else ind_num]
            for val_dim in range(ind_ndim, 4):
                val_shape.append(1 if onp.random.randint(0, 5)==0 else shape[val_dim])
            # case: val is tensor
            configs.append(tuple([shape, ind, tuple(val_shape), ind_ndim, ind_num]))

    dtypes = ['float32', 'float64', 'int32', 'int64']
    grad_req = ['write', 'null', 'add']
    for hybridize, grad_req_a, grad_req_val, dtype, indtype in \
        itertools.product([True, False], grad_req, grad_req, dtypes, ['int32', 'int64']):
        for a_shape, ind, val_shape ,ind_ndim, ind_num in configs:
            eps = 1e-3
            atype = dtype
            valtype = dtype
            test_index_update = TestIndexUpdate()
            if hybridize:
                test_index_update.hybridize()
            a = mx.nd.random.uniform(-10.0, 10.0, shape=a_shape).as_np_ndarray().astype(atype)
            a.attach_grad(grad_req=grad_req_a)
            val = mx.nd.random.uniform(-10.0, 10.0, shape=val_shape).as_np_ndarray().astype(valtype)
            val.attach_grad(grad_req=grad_req_val)
            with mx.autograd.record():
                mx_ret = test_index_update(a, np.array(ind).astype(indtype), val)
            assert mx_ret.shape == a.shape
            assert mx_ret.dtype == a.dtype
            check_index_update_forward(mx_ret.asnumpy(), a.asnumpy(), ind.astype(indtype), val.asnumpy(), ind_ndim, ind_num, eps)

            if atype not in ['float16', 'float32', 'float64'] or valtype not in ['float16', 'float32', 'float64']:
                continue
            if grad_req_a != 'null' or grad_req_val != 'null':
                init_a_grad = mx.nd.random.uniform(-10.0, 10.0, shape=a_shape).as_np_ndarray().astype(atype)
                init_val_grad = mx.nd.random.uniform(-10.0, 10.0, shape=val_shape).as_np_ndarray().astype(valtype)
                out_grad = mx.nd.random.uniform(-10.0, 10.0, shape=a_shape).as_np_ndarray().astype(atype)
                if grad_req_a == 'add':
                    if init_a_grad.ndim == 0:
                        a.grad[()] = init_a_grad.item()
                    else:
                        a.grad[:] = init_a_grad
                if grad_req_val == 'add':
                    if init_val_grad.ndim == 0:
                        val.grad[()] = init_val_grad.item()
                    else:
                        val.grad[:] = init_val_grad
                mx_ret.backward(out_grad)
                expected_bwd_a, expected_bwd_val = index_update_bwd(out_grad.asnumpy(), init_a_grad.asnumpy(), ind,
                                                                    init_val_grad.asnumpy(), ind_ndim, ind_num,
                                                                    grad_req_a, grad_req_val)

                if grad_req_a == 'null':
                    assert a.grad is None
                else:
                    assert_almost_equal(a.grad.asnumpy(), expected_bwd_a, rtol = eps, atol=eps)
                if grad_req_val == 'null':
                    assert val.grad is None
                else:
                    assert_almost_equal(val.grad.asnumpy(), expected_bwd_val, rtol = eps, atol=eps)

            mx_out = npx.index_update(a, np.array(ind).astype(indtype), val)
            check_index_update_forward(mx_out.asnumpy(), a.asnumpy(), ind.astype(indtype), val.asnumpy(), ind_ndim, ind_num, eps)


@use_np
def test_npx_batch_dot():
    device = mx.device.current_device()
    dtypes = ['float32', 'float64']
    if device.device_type == 'gpu':
        dtypes += ['float16']
    eps_dict = {'float32': 1E-4, 'float64': 1E-4, 'float16': 1E-3}
    class TestBatchDot(HybridBlock):
        def __init__(self, transpose_a, transpose_b):
            super(TestBatchDot, self).__init__()
            self._transpose_a = transpose_a
            self._transpose_b = transpose_b

        def forward(self, lhs, rhs):
            return npx.batch_dot(lhs, rhs,
                                   transpose_a=self._transpose_a,
                                   transpose_b=self._transpose_b)

    def batch_dot_numpy(lhs, rhs, transpose_a, transpose_b):
        assert lhs.ndim == rhs.ndim >= 3
        if transpose_a:
            lhs = lhs.swapaxes(-1, -2)
        if transpose_b:
            rhs = rhs.swapaxes(-1, -2)
        return onp.matmul(lhs, rhs)

    def gt_grad_batch_dot_numpy(lhs, rhs, ograd, transpose_a, transpose_b, lhs_req, rhs_req,
                                init_lhs_grad, init_rhs_grad):

        if transpose_a and transpose_b:
            # Gradient of z = dot(x.T, y.T)
            # dx = dot(dz, y).T = dot(y.T, dz.T)
            # dy = dot(x, dz).T = dot(dz.T, x.T)
            lhs_grad = batch_dot_numpy(rhs, ograd, transpose_a=True, transpose_b=True)
            rhs_grad = batch_dot_numpy(ograd, lhs, transpose_a=True, transpose_b=True)
        elif not transpose_a and transpose_b:
            # Gradient of z = dot(x, y.T)
            # dx = dot(dz, y)
            # dy = dot(x.T, dz).T = dot(dz.T, x)
            lhs_grad = batch_dot_numpy(ograd, rhs, transpose_a=False, transpose_b=False)
            rhs_grad = batch_dot_numpy(ograd, lhs, transpose_a=True, transpose_b=False)
        elif transpose_a and not transpose_b:
            # Gradient of z = dot(x.T, y)
            # dx = dot(dz, y.T).T = dot(y, dz.T)
            # dy = dot(x, dz)
            lhs_grad = batch_dot_numpy(rhs, ograd, transpose_a=False, transpose_b=True)
            rhs_grad = batch_dot_numpy(lhs, ograd, transpose_a=False, transpose_b=False)
        else:
            # Gradient of z = dot(x, y)
            # dx = dot(dz, y.T)
            # dy = dot(x.T, dz)
            lhs_grad = batch_dot_numpy(ograd, rhs, transpose_a=False, transpose_b=True)
            rhs_grad = batch_dot_numpy(lhs, ograd, transpose_a=True, transpose_b=False)
        if lhs_req == 'add':
            lhs_grad += init_lhs_grad
        if rhs_req == 'add':
            rhs_grad += init_rhs_grad
        return lhs_grad, rhs_grad


    configs = [
        ((2, 3, 0), (2, 4, 0), False, True),
        ((2, 4, 3), (2, 4, 3), True, False),
        ((0, 3, 0), (0, 0, 2), False, False),
        ((3, 2, 3, 2), (3, 2, 2, 3), True, True),
        ((3, 1, 5, 2), (3, 1, 2, 1), False, False)
    ]
    bad_configs = [
        ((5, 3, 2), (5, 1, 3), False, False),
        ((2, 5, 3, 1), (2, 4, 3, 1), True, False)
    ]
    for hybridize in [True, False]:
        for lhs_shape, rhs_shape, transpose_a, transpose_b in configs:
            for dtype in dtypes:
                eps = eps_dict[dtype]
                for lhs_grad_req in ['write', 'add']:
                    for rhs_grad_req in ['write', 'add']:
                        f_batch_dot = TestBatchDot(transpose_a=transpose_a,
                                                   transpose_b=transpose_b)
                        if hybridize:
                            f_batch_dot.hybridize()
                        lhs_val = mx.np.array(onp.random.uniform(-1.0, 1.0, lhs_shape), dtype=dtype)
                        rhs_val = mx.np.array(onp.random.uniform(-1.0, 1.0, rhs_shape), dtype=dtype)
                        lhs_val.attach_grad(grad_req=lhs_grad_req)
                        rhs_val.attach_grad(grad_req=rhs_grad_req)
                        gt_out = batch_dot_numpy(lhs_val.asnumpy(), rhs_val.asnumpy(),
                                                 transpose_a, transpose_b)
                        init_lhs_grad = mx.np.random.uniform(-1.0, 1.0, lhs_shape, dtype=dtype)
                        init_rhs_grad = mx.np.random.uniform(-1.0, 1.0, rhs_shape, dtype=dtype)
                        o_grad = mx.np.random.uniform(-1.0, 1.0, gt_out.shape, dtype=dtype)
                        if lhs_grad_req == 'add':
                            lhs_val.grad[:] = init_lhs_grad
                        if rhs_grad_req == 'add':
                            rhs_val.grad[:] = init_rhs_grad
                        with mx.autograd.record():
                            out = f_batch_dot(lhs_val, rhs_val)
                        out.backward(o_grad)
                        assert_almost_equal(out.asnumpy(), gt_out, rtol=eps, atol=eps)
                        gt_lhs_grad, gt_rhs_grad = gt_grad_batch_dot_numpy(lhs_val.asnumpy(),
                                                              rhs_val.asnumpy(),
                                                              o_grad.asnumpy(),
                                                              transpose_a=transpose_a,
                                                              transpose_b=transpose_b,
                                                              lhs_req=lhs_grad_req,
                                                              rhs_req=rhs_grad_req,
                                                              init_lhs_grad=init_lhs_grad.asnumpy(),
                                                              init_rhs_grad=init_rhs_grad.asnumpy())
                        assert_almost_equal(lhs_val.grad.asnumpy(), gt_lhs_grad, rtol=eps, atol=eps)
                        assert_almost_equal(rhs_val.grad.asnumpy(), gt_rhs_grad, rtol=eps, atol=eps)
    for lhs_shape, rhs_shape, transpose_a, transpose_b in bad_configs:
        for dtype in dtypes:
            lhs_val = mx.np.array(onp.random.uniform(-1.0, 1.0, lhs_shape), dtype=dtype)
            rhs_val = mx.np.array(onp.random.uniform(-1.0, 1.0, rhs_shape), dtype=dtype)
            pytest.raises(MXNetError, lambda: mx.npx.batch_dot(lhs_val, rhs_val,
                                                               transpose_a=transpose_a,
                                                               transpose_b=transpose_b))


@use_np
@pytest.mark.parametrize('shape', [(4, 2), (4, 3, 4),
    (4, 6, 4, 5), (4, 5, 6, 4, 5)])
@pytest.mark.parametrize('fix_gamma', [False, True])
@pytest.mark.parametrize('cudnn_off', [False, True])
@pytest.mark.parametrize('output_mean_var', [False, True])
@pytest.mark.flaky
def test_npx_batch_norm(shape, fix_gamma, cudnn_off, output_mean_var):
    momentum = 0.9
    epsilon = 1e-5
    class TestBatchNorm(HybridBlock):
        def __init__(self, eps=1e-5, fix_gamma=False, momentum=0.9, **kwargs):
            super().__init__()
            self.eps = eps
            self.fix_gamma = fix_gamma
            self.momentum = momentum
            self.kwargs = kwargs
        def forward(self, data, bn_gamma, bn_beta,
                           bn_running_mean, bn_running_var):
            op = npx.batch_norm
            output = op(data, bn_gamma, bn_beta,
                        bn_running_mean, bn_running_var,
                        momentum=self.momentum, eps=self.eps,
                        fix_gamma=self.fix_gamma, **self.kwargs)
            return output

    def _test_batchnorm_impl(axis,
                             data_grad_req, gamma_grad_req, beta_grad_req):
        kwargs = dict(output_mean_var=output_mean_var)
        kwargs.update(dict(axis=axis, cudnn_off=cudnn_off))
        op = TestBatchNorm(eps=epsilon, fix_gamma=fix_gamma, momentum=momentum, **kwargs)
        nch = shape[axis]

        if not fix_gamma:
            bn_gamma = np.random.uniform(size=(nch,))
            bn_gamma.attach_grad(grad_req=gamma_grad_req)
        else:
            bn_gamma = np.ones((nch,))

        bn_beta = np.random.uniform(size=(nch,))
        bn_beta.attach_grad(grad_req=beta_grad_req)

        bn_running_mean = np.zeros(nch)
        bn_running_var = np.ones(nch)

        running_mean = np.zeros(nch)
        running_var = np.ones(nch)
        num_iters = 10
        expand_shape = [1] * len(shape)
        expand_shape[axis] = shape[axis]
        expand_shape = tuple(expand_shape)
        data = np.random.uniform(size=shape)
        data.attach_grad(grad_req=data_grad_req)
        adX, adW, adb = 0, 0, 0
        is_train = data_grad_req != 'null' or \
            (not fix_gamma and gamma_grad_req != 'null') or \
            beta_grad_req != 'null'
        for _ in range(num_iters):
            if data_grad_req != 'add':
                data = np.random.uniform(size=shape)
                data.attach_grad(grad_req=data_grad_req)
            ograd = np.random.uniform(size=shape)
            with mx.autograd.record():
                output = op(data, bn_gamma, bn_beta,
                            bn_running_mean, bn_running_var)
                if output_mean_var:
                    output, output_mean, output_std = output
                if is_train:
                    output.backward(ograd)
            mx.nd.waitall()

            assert 0 <= axis < data.ndim
            reduce_axis = tuple(i for i in range(data.ndim) if i != axis)
            assert len(reduce_axis) == data.ndim - 1
            data_mean = data.mean(
                axis=reduce_axis, keepdims=True)
            data_var = ((data - data_mean) ** 2).mean(axis=reduce_axis,
                                                        keepdims=True)

            target_output = (data - data_mean) / \
                np.sqrt(data_var + epsilon) * \
                bn_gamma.reshape(expand_shape) + \
                bn_beta.reshape(expand_shape)

            # squeeze data_mean and data_var
            data_mean_flat = data_mean.squeeze()
            data_var_flat = data_var.squeeze()

            running_mean = running_mean * momentum + \
                data_mean_flat * (1 - momentum)

            m = onp.prod(shape) / shape[axis]
            # cudnn uses m-1 in the denominator of its sample variance calculation, not m
            sample_var_adjust = 1.0 if cudnn_off or fix_gamma else m / (m-1)
            running_var = running_var * momentum + \
                data_var_flat * sample_var_adjust * (1 - momentum)

            W = bn_gamma.reshape(expand_shape)
            dnx = ograd * W
            xsm = data - data_mean
            nd = 1.0 / np.sqrt(data_var + epsilon)
            nx = xsm * nd
            dvar = (dnx * xsm).sum(axis=reduce_axis, keepdims=True,
                                  ) * (-0.5) * np.power(nd, 3)
            dmean = -nd * dnx.sum(axis=reduce_axis, keepdims=True) - \
                dvar * xsm.mean(axis=reduce_axis, keepdims=True,
                                ) * 2.0
            dX = dnx * nd + dvar * xsm * (2.0 / m) + dmean * (1.0 / m)
            dW = (ograd * nx).sum(axis=reduce_axis)
            db = ograd.sum(axis=reduce_axis)
            adX = dX if data_grad_req != 'add' else adX + dX
            adW = dW if gamma_grad_req != 'add' else adW + dW
            adb = db if beta_grad_req != 'add' else adb + db

            atol, rtol = 5e-2, 5e-2

            if output_mean_var:
                assert_almost_equal(output_mean.asnumpy(),
                                    data_mean_flat.asnumpy(),
                                    atol=atol, rtol=rtol)
                assert_almost_equal(output_std.asnumpy(),
                                    (1.0 / np.sqrt(data_var_flat +
                                            epsilon)).asnumpy(),
                                    atol=atol, rtol=rtol)
            assert_almost_equal(output.asnumpy(), target_output.asnumpy(),
                                atol=atol, rtol=rtol)
            if is_train:
                assert_almost_equal(bn_running_mean.asnumpy(
                ), running_mean.asnumpy(), atol=atol, rtol=rtol)
                assert_almost_equal(bn_running_var.asnumpy(
                ), running_var.asnumpy(), atol=atol, rtol=rtol)

            if data_grad_req != 'null':
                assert_almost_equal(data.grad.asnumpy(),
                                    adX.asnumpy(), atol=atol, rtol=rtol)
            if not fix_gamma:
                if gamma_grad_req != 'null':
                    assert_almost_equal(
                        bn_gamma.grad.asnumpy(), adW.asnumpy(),
                        atol=atol, rtol=rtol)
            else:
                assert((bn_gamma.asnumpy() == 1).all())
            if beta_grad_req != 'null':
                assert_almost_equal(
                    bn_beta.grad.asnumpy(), adb.asnumpy(), atol=atol, rtol=rtol)

    grad_reqs = ['write'] if len(shape) != 4 else ['null', 'write', 'add']
    for data_grad_req in grad_reqs:
        for gamma_grad_req in grad_reqs:
            if fix_gamma and gamma_grad_req != 'null':
                continue
            for beta_grad_req in grad_reqs:
                for axis in range(len(shape)):
                    _test_batchnorm_impl(axis,
                        data_grad_req, gamma_grad_req, beta_grad_req)


def np_softmax(x, axis=-1):
    if (x.shape[axis] == 0):
        return onp.sum(x, axis=axis, keepdims=True)
    x = x - onp.max(x, axis=axis, keepdims=True)
    x = onp.exp(x)
    x /= onp.sum(x, axis=axis, keepdims=True)
    return x

def np_log_softmax(x, axis=-1):
    return onp.log(np_softmax(x, axis))

@use_np
def test_npx_softmax():
    class TestSoftmax(HybridBlock):
        def __init__(self, axis):
            super(TestSoftmax, self).__init__()
            self._axis = axis

        def forward(self, a):
            return npx.softmax(a, axis=axis)

    class TestLogSoftmax(HybridBlock):
        def __init__(self, axis):
            super(TestLogSoftmax, self).__init__()
            self._axis = axis

        def forward(self, a):
            return npx.log_softmax(a, axis=axis)


    #(operator, function) tuples
    tested_ops = [(TestSoftmax, np_softmax),
                  (TestLogSoftmax, np_log_softmax)]

    # only testing 0-size shaped inputs here, other input cases have been tested in test_opeartor.py
    for SoftmaxOp, softmax_function in tested_ops:
        for hybridize in [True, False]:
            for shape in [(3, 0, 4), (0, 0)]:
                mx_a = np.random.uniform(size=shape)
                mx_a.attach_grad()
                for axis in range(-len(shape), len(shape)):
                    test_softmax_op = SoftmaxOp(axis)
                    if hybridize:
                        test_softmax_op.hybridize()

                    with mx.autograd.record():
                        mx_out = test_softmax_op(mx_a)

                    mx_out.wait_to_read()

                    np_out = softmax_function(mx_a.asnumpy(), axis)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, equal_nan=True)

                    mx_out.backward()
                    mx_a.grad.wait_to_read()
                    assert_almost_equal(mx_a.grad.asnumpy(), onp.zeros(shape), rtol=1e-3, atol=1e-5)


def np_masked_softmax(data, mask, axis=-1, temperature=1.0):
    neg = -1e18
    if data.dtype == onp.float16:
        neg = -1e4
    temp = onp.where(mask, data, neg)
    result = (np_softmax(temp, axis=axis) / temperature) * mask
    return result

def np_masked_log_softmax(data, mask, axis=-1, temperature=1.0):
    neg = -1e18
    if data.dtype == onp.float16:
        neg = -1e4
    data = onp.where(mask, data, neg)
    return onp.where(mask, np_log_softmax(data, axis=axis) / temperature, -onp.inf)

@use_np
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('shape', [(3, 0, 4), (0, 0)])
def test_npx_masked_softmax(hybridize, shape):
    class TestMaskedSoftmax(HybridBlock):
        def __init__(self, axis):
            super(TestMaskedSoftmax, self).__init__()
            self._axis = axis

        def forward(self, a, mask):
            return npx.masked_softmax(a, mask, axis=self._axis)

    class TestMaskedLogSoftmax(HybridBlock):
        def __init__(self, axis):
            super(TestMaskedLogSoftmax, self).__init__()
            self._axis = axis

        def forward(self, a, mask):
            return npx.masked_log_softmax(a, mask, axis=self._axis)

    #(operator, function) tuples
    tested_ops = [(TestMaskedSoftmax, np_masked_softmax),
                  (TestMaskedLogSoftmax, np_masked_log_softmax)]

    # only testing 0-size shaped inputs here, other input cases have been tested in test_opeartor.py
    for SoftmaxOp, softmax_function in tested_ops:
        mx_a = np.random.uniform(size=shape)
        mask = np.random.randint(0, 2, shape)
        mx_a.attach_grad()
        mask.attach_grad()
        for axis in range(-len(shape), len(shape)):
            test_softmax_op = SoftmaxOp(axis)
            if hybridize:
                test_softmax_op.hybridize()

            with mx.autograd.record():
                mx_out = test_softmax_op(mx_a, mask)

            mx_out.wait_to_read()

            np_out = softmax_function(mx_a.asnumpy(), mask.asnumpy(), axis)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, equal_nan=True)


@use_np
def test_npi_boolean_assign():
    class TestBooleanAssignScalar(HybridBlock):
        def __init__(self, val, start_axis):
            super(TestBooleanAssignScalar, self).__init__()
            self._val = val
            self._start_axis = start_axis

        def forward(self, a, mask):
            return _npi.boolean_mask_assign_scalar(a, mask, self._val, start_axis=self._start_axis, out=a)

    class TestBooleanAssignTensor(HybridBlock):
        def __init__(self, start_axis):
            super(TestBooleanAssignTensor, self).__init__()
            self._start_axis = start_axis

        def forward(self, a, mask, value):
            return _npi.boolean_mask_assign_tensor(a, mask, value, start_axis=self._start_axis, out=a)

    configs = [
        ((3, 4), (3, 4), 0),
        ((3, 0), (3, 0), 0),
        ((), (), 0),
        ((2, 3, 4, 5), (2, 3), 0),
        ((2, 3, 4, 5), (3, 4), 1),
        ((2, 3, 4, 5), (4, 5), 2),
    ]

    for hybridize in [False]:
        for config in configs:
            dshape, mshape, start_axis = config
            test_data = np.random.uniform(size=dshape)
            valid_num = 0
            while valid_num == 0:
                mx_mask = np.random.choice(np.array([False, True], dtype=np.bool), size=mshape)
                if test_data.size == 0:
                    break
                valid_num = int(mx_mask.asnumpy().sum())
            np_mask = mx_mask.asnumpy().astype(onp.bool)
            vshape = []
            vshape_broadcast = []
            for i in range(len(dshape)):
                if i < start_axis:
                    vshape.append(dshape[i])
                    vshape_broadcast.append(dshape[i])
                elif i == start_axis:
                    vshape.append(valid_num)
                    vshape_broadcast.append(1)
                elif i >= start_axis + len(mshape):
                    vshape.append(dshape[i])
                    vshape_broadcast.append(dshape[i])
            vshape_broadcast = tuple(vshape_broadcast)
            for val in [42.0, onp.array(42.), onp.array([42.]), onp.random.uniform(size=vshape), onp.random.uniform(size=vshape_broadcast)]:
                mx_val = val if isinstance(val, float) else np.array(val, dtype=np.float32)
                test_block = TestBooleanAssignScalar(val, start_axis) if isinstance(val, float) else TestBooleanAssignTensor(start_axis)
                if hybridize:
                    test_block.hybridize()
                np_data = test_data.asnumpy()
                mx_data1 = test_data.copy()
                mx_data2 = test_data.copy()
                trailing_axis = len(np_data.shape) - len(np_mask.shape) - start_axis
                if start_axis == 0:
                    if trailing_axis == 0:
                        np_data[np_mask] = val
                        mx_data1[mx_mask] = mx_val
                    elif trailing_axis == 1:
                        np_data[np_mask, :] = val
                        mx_data1[mx_mask, :] = mx_val
                    elif trailing_axis == 2:
                        np_data[np_mask, :, :] = val
                        mx_data1[mx_mask, :, :] = mx_val
                elif start_axis == 1:
                    if trailing_axis == 0:
                        np_data[:, np_mask] = val
                        mx_data1[:, mx_mask] = mx_val
                    elif trailing_axis == 1:
                        np_data[:, np_mask, :] = val
                        mx_data1[:, mx_mask, :] = mx_val
                elif start_axis == 2:
                    if trailing_axis == 0:
                        np_data[:, :, np_mask] = val
                        mx_data1[:, :, mx_mask] = mx_val
                mx_data1 = test_block(mx_data2, mx_mask) if isinstance(val, float) else test_block(mx_data2, mx_mask, mx_val)
                assert_almost_equal(mx_data1.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)
                assert_almost_equal(mx_data2.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)


@use_np
def test_np_reshape():
    class TestReshape(HybridBlock):
        def __init__(self, newshape):
            super(TestReshape, self).__init__()
            self._newshape = newshape

        def forward(self, a):
            return np.reshape(a, self._newshape)

    shape_pairs = [((2, 6), (6, 2)), ((2, 6), (3, 4)), ((1, 0), (0,)), ((0, 0), (0,)), ((), (1, 1, 1))]
    for hybridize in [True, False]:
        for shape_pair in shape_pairs:
            shape1, shape2 = shape_pair
            test_reshape = TestReshape(shape2)
            if hybridize:
                test_reshape.hybridize()
            x = rand_ndarray(shape1).as_np_ndarray()
            x.attach_grad()
            np_out = onp.reshape(x.asnumpy(), shape2)
            with mx.autograd.record():
                mx_out = test_reshape(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)
            mx_out.backward()
            np_backward = onp.ones(shape1)
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5, use_broadcast=False)

            mx_out = np.reshape(x, shape2)
            np_out = onp.reshape(x.asnumpy(), shape2)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@use_np
@pytest.mark.parametrize('descending', [True, False])
@pytest.mark.parametrize('shape', [
    (),
    (2, 3),
    (1, 0, 2),
])
@pytest.mark.parametrize('hybrid', [False, True])
def test_np_argsort(descending, shape, hybrid):
    class TestArgsort(HybridBlock):
        def __init__(self, axis, descending):
            super(TestArgsort, self).__init__()
            self._axis = axis
            self._descending = descending

        def forward(self, x):
            return np.argsort(x, axis=self._axis, descending=self._descending)

    data = np.random.uniform(size=shape)
    np_data = data.asnumpy()
    for axis in [None] + [i for i in range(-len(shape), len(shape))]:
        if descending:
            np_out = onp.argsort(-1 * np_data, axis)
        else:
            np_out = onp.argsort(np_data, axis)

        test_argsort = TestArgsort(axis, descending)

        if hybrid:
            test_argsort.hybridize()
        mx_out = test_argsort(data)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-5, atol=1e-6, use_broadcast=False)

        mx_out = np.argsort(data, axis, descending)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-5, atol=1e-6, use_broadcast=False)


@use_np
@pytest.mark.parametrize('descending', [True, False])
@pytest.mark.parametrize('shape', [
    (),
    (1,),
    (5,),
    (4, 3),
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
])
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_sort(shape, dtype, hybridize, descending):
    class TestSort(HybridBlock):
        def __init__(self, axis, descending):
            super(TestSort, self).__init__()
            self._axis = axis
            self._descending = descending

        def forward(self, x):
            return np.sort(x, self._axis, descending=self._descending)

    a = np.random.uniform(low=0, high=100, size=shape, dtype='float64').astype(dtype)
    axis_list = list(range(len(shape)))
    axis_list.append(None)
    axis_list.append(-1)
    for axis in axis_list:
        test = TestSort(axis, descending)
        if hybridize:
            test.hybridize()
        if axis == -1 and len(shape)==0:
            continue
        ret = test(a)
        if descending:
            expected_ret = -onp.sort(-1 * a.asnumpy(), axis)
        else:
            expected_ret = onp.sort(a.asnumpy(), axis)
        assert_almost_equal(ret.asnumpy(), expected_ret, atol=1e-5, rtol=1e-5, use_broadcast=False)

        # check imperative again
        ret = np.sort(a, axis=axis, descending=descending)
        assert_almost_equal(ret.asnumpy(), expected_ret, atol=1e-5, rtol=1e-5, use_broadcast=False)


@use_np
def test_np_squeeze():
    config = [((), None),
              ((), -1),
              ((), 0),
              ((4, 1, 2), None),
              ((1, 1, 1), None),
              ((1, 0, 1, 5), 2),
              ((1, 0, 1, 1), (-1, -4))]

    class TestSqueeze(HybridBlock):
        def __init__(self, axis):
            super(TestSqueeze, self).__init__()
            self._axis = axis

        def forward(self, x):
            return np.squeeze(x, self._axis)

    for shape, axis in config:
        data_np = onp.random.uniform(size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = onp.squeeze(data_np, axis)
        ret_mx = np.squeeze(data_mx, axis)
        assert_almost_equal(ret_mx.asnumpy(), ret_np, rtol=1e-5, atol=1e-6, use_broadcast=False)

        net = TestSqueeze(axis)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            data_mx.attach_grad()
            with mx.autograd.record():
                ret_mx = net(data_mx)
            assert_almost_equal(ret_mx.asnumpy(), ret_np, rtol=1e-5, atol=1e-6, use_broadcast=False)
            ret_mx.backward()
            assert_almost_equal(data_mx.grad.asnumpy(), onp.ones_like(data_np),
                                rtol=1e-5, atol=1e-6, use_broadcast=False)


@xfail_when_nonstandard_decimal_separator
@use_np
def test_np_tri():
    class TestTri(HybridBlock):
        def __init__(self, N, M=None, k=0, dtype=None):
            super(TestTri, self).__init__()
            self._N = N
            self._M = M
            self._k = k
            self._dtype = dtype

        def forward(self, x):
            return x + np.tri(self._N, self._M, self._k, self._dtype)

    dtypes = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', None]
    hybrids = [False, True]

    for dtype, hybrid in itertools.product(dtypes, hybrids):
        N = random.randint(2,6)
        M = random.randint(2,6)
        k = random.randint(-M*2, N*2)

        test_tri = TestTri(N, M, k, dtype)
        if hybrid:
            test_tri.hybridize()
        np_out = np.tri(N, M, k, dtype)
        x = np.zeros(shape=(), dtype=dtype)
        mx_out = test_tri(x)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-5, atol=1e-6, use_broadcast=False)

        mx_out = np.tri(N, M, k, dtype)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-5, atol=1e-6, use_broadcast=False)


@use_np
def test_np_prod():
    class TestProd(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestProd, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def forward(self, a, *args, **kwargs):
            return np.prod(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    in_data_dim = random.choice([3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float32', 'float64']:
                    for dtype in ['float32', 'float64']:
                        # test gluon
                        test_prod = TestProd(axis=axis, dtype=dtype, keepdims=keepdims)
                        if hybridize:
                            test_prod.hybridize()
                        x = np.array(onp.random.uniform(-2.0, 2.0, size=shape), dtype=itype)
                        x.attach_grad()
                        expected_ret = onp.prod(x.asnumpy(), axis=axis, keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_prod(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5, use_broadcast=False)
                        y.backward()
                        # use keepdims=True so that broadcast divide can be used to calculate
                        # grad of input
                        expected_ret = onp.prod(x.asnumpy(), axis=axis, keepdims=True)
                        assert_almost_equal(x.grad.asnumpy(), expected_ret / x.asnumpy(), rtol=1e-3, atol=1e-3,
                                            use_broadcast=False)

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.prod(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=onp.float32)

                        # test imperative
                        mx_out = np.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = onp.prod(x.asnumpy(), axis=axis, keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@use_np
def test_np_flatten():
    class TestFlatten(HybridBlock):
        def forward(self, x):
            return x.flatten()

    shapes = [(), (2, 0, 1), (3, 4, 5), 6, (0,), (0, 0, 0)]
    for shape in shapes:
        for hybridize in [True, False]:
            test_flatten = TestFlatten()
            if hybridize:
                test_flatten.hybridize()
            a_np = onp.random.uniform(size=shape).astype('float32')
            a_mx = np.array(a_np, dtype=a_np.dtype)
            a_mx.attach_grad()
            with mx.autograd.record():
                ret = test_flatten(a_mx)
            expected_ret = a_np.flatten()
            assert_almost_equal(expected_ret, ret.asnumpy(), rtol=1e-5, atol=1e-6, use_broadcast=False)
            # check gradient
            ret.backward()
            assert_almost_equal(a_mx.grad.asnumpy(), onp.ones_like(a_np), rtol=1e-5, atol=1e-6, use_broadcast=False)


@use_np
@pytest.mark.parametrize('src_shape,dst_shape', [
    ((), (1, 2, 4, 5)),
    ((1,), (4, 5, 6)),
    ((1, 0), (2, 4, 0)),
    ((1, 1), (2, 4, 0)),
    ((4, 1), (1, 2, 3, 4, 5)),
    ((4, 1), (1, 0, 3, 4, 5))
])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_broadcast_to(src_shape, dst_shape, hybridize):
    class TestBroadcastTo(HybridBlock):
        def __init__(self, dst_shape):
            super(TestBroadcastTo, self).__init__()
            self._dst_shape = dst_shape

        def forward(self, x):
            return np.broadcast_to(x, self._dst_shape)

    class TestScalarBroadcastTo(HybridBlock):
        def __init__(self, scalar, dst_shape):
            super(TestScalarBroadcastTo, self).__init__()
            self._scalar = scalar
            self._dst_shape = dst_shape

        def forward(self, x):
            return np.broadcast_to(self._scalar, self._dst_shape)

    test_broadcast_to = TestBroadcastTo(dst_shape)
    if hybridize:
        test_broadcast_to.hybridize()

    a = onp.random.uniform(size=src_shape).astype(np.float32)
    expected_ret = onp.broadcast_to(a, dst_shape)
    a_mx = np.array(a, dtype=a.dtype)
    a_mx.attach_grad()
    with mx.autograd.record():
        ret = test_broadcast_to(a_mx)
    assert_almost_equal(ret.asnumpy(), expected_ret, rtol=1e-5, atol=1e-6, use_broadcast=False)
    ret.backward()
    expected_grad = collapse_sum_like(onp.ones_like(expected_ret), src_shape)
    assert_almost_equal(a_mx.grad.asnumpy(), expected_grad, rtol=1e-5, atol=1e-6, use_broadcast=False)

    # Test scalar case
    scalar = 1.0
    test_scalar_broadcast_to = TestScalarBroadcastTo(scalar, dst_shape)
    expected_ret = onp.broadcast_to(scalar, dst_shape)
    with mx.autograd.record():
        # `np.empty(())` serves as a dummpy input
        ret = test_scalar_broadcast_to(np.empty(()))
    assert_almost_equal(ret.asnumpy(), expected_ret, rtol=1e-5, atol=1e-6, use_broadcast=False)

@use_np
@pytest.mark.parametrize('src_shape,npx_dst_shape,np_dst_shape', [
    ((5,), (3, 4, -2), (3, 4, 5)),
    ((5,), (0, -2), (0, 5)),
    ((1, 0), (2, -2, -2), (2, 1, 0)),
    ((3, 4), (1, 2, 3, -2), (1, 2, 3, 4)),
    ((3, 4), (1, 0, -2, 4), (1, 0, 3, 4))
])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_broadcast_to_npx(src_shape, npx_dst_shape, np_dst_shape, hybridize):
    class TestBroadcastTo(HybridBlock):
        def __init__(self, dst_shape):
            super(TestBroadcastTo, self).__init__()
            self._dst_shape = dst_shape

        def forward(self, x):
            return np.broadcast_to(x, self._dst_shape)

    class TestScalarBroadcastTo(HybridBlock):
        def __init__(self, scalar, dst_shape):
            super(TestScalarBroadcastTo, self).__init__()
            self._scalar = scalar
            self._dst_shape = dst_shape

        def forward(self, x):
            return np.broadcast_to(self._scalar, self._dst_shape)

    test_broadcast_to = TestBroadcastTo(npx_dst_shape)
    if hybridize:
        test_broadcast_to.hybridize()

    a = onp.random.uniform(size=src_shape).astype(np.float32)
    expected_ret = onp.broadcast_to(a, np_dst_shape)
    a_mx = np.array(a, dtype=a.dtype)
    a_mx.attach_grad()
    with mx.autograd.record():
        ret = test_broadcast_to(a_mx)
    assert_almost_equal(ret.asnumpy(), expected_ret, rtol=1e-5, atol=1e-6, use_broadcast=False)
    ret.backward()
    expected_grad = collapse_sum_like(onp.ones_like(expected_ret), src_shape)
    assert_almost_equal(a_mx.grad.asnumpy(), expected_grad, rtol=1e-5, atol=1e-6, use_broadcast=False)


@use_np
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float16, onp.int32])
@pytest.mark.parametrize('data_shape,axes_workload', [
    [(), [(), None]],
    [(2,), [(0,), None]],
    [(0, 2), [(0, 1), (1, 0)]],
    [(5, 10), [(0, 1), (1, 0), None]],
    [(8, 2, 3), [(2, 0, 1), (0, 2, 1), (0, 1, 2), (2, 1, 0), (-1, 1, 0), None]],
    [(8, 2, 16), [(0, 2, 1), (2, 0, 1), (0, 1, 2), (2, 1, 0), (-1, -2, -3)]],
    [(8, 3, 4, 8), [(0, 2, 3, 1), (1, 2, 3, 0), (0, 3, 2, 1)]],
    [(8, 3, 2, 3, 8), [(0, 1, 3, 2, 4), (0, 1, 2, 3, 4), (4, 0, 1, 2, 3)]],
    [(3, 4, 3, 4, 3, 2), [(0, 1, 3, 2, 4, 5), (2, 3, 4, 1, 0, 5), None]],
    [(3, 4, 3, 4, 3, 2, 2), [(0, 1, 3, 2, 4, 5, 6),
     (2, 3, 4, 1, 0, 5, 6), None]],
    [(3, 4, 3, 4, 3, 2, 3, 2), [(0, 1, 3, 2, 4, 5, 7, 6),
     (2, 3, 4, 1, 0, 5, 7, 6), None]],
])
@pytest.mark.parametrize('grad_req', ['write', 'add'])
def test_np_transpose(data_shape, axes_workload, hybridize, dtype, grad_req):
    def np_transpose_grad(out_shape, dtype, axes=None):
        ograd = onp.ones(out_shape, dtype=dtype)
        if axes is None or axes == ():
            return onp.transpose(ograd, axes)
        np_axes = onp.array(list(axes))
        transpose_axes = onp.zeros_like(np_axes)
        transpose_axes[np_axes] = onp.arange(len(np_axes))
        return onp.transpose(ograd, tuple(list(transpose_axes)))

    class TestTranspose(HybridBlock):
        def __init__(self, axes=None):
            super(TestTranspose, self).__init__()
            self.axes = axes

        def forward(self, a):
            return np.transpose(a, self.axes)

    for axes in axes_workload:
        test_trans = TestTranspose(axes)
        if hybridize:
            test_trans.hybridize()
        x = np.random.normal(0, 1, data_shape).astype(dtype)
        x = x.astype(dtype)
        x.attach_grad(grad_req=grad_req)
        if grad_req == 'add':
            x.grad[()] = np.random.normal(0, 1, x.grad.shape).astype(x.grad.dtype)
            x_grad_np = x.grad.asnumpy()
        np_out = onp.transpose(x.asnumpy(), axes)
        with mx.autograd.record():
            mx_out = test_trans(x)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)
        mx_out.backward()
        np_backward = np_transpose_grad(np_out.shape, dtype, axes)
        if grad_req == 'add':
            assert_almost_equal(x.grad.asnumpy(), np_backward + x_grad_np,
                                rtol=1e-3, atol=1e-5, use_broadcast=False)
        else:
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5, use_broadcast=False)

        mx_out = x.transpose(axes)
        np_out = x.asnumpy().transpose(axes)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)

        if isinstance(axes, (list, tuple)):
            mx_out = x.transpose(*axes)
            np_out = x.asnumpy().transpose(*axes)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@use_np
def test_np_transpose_error():
    # Test for error raising
    dat = np.random.normal(0, 1, (3, 4, 5), dtype=np.float32)
    pytest.raises(ValueError, lambda: dat.transpose((0, 0, 1)))
    pytest.raises(MXNetError, lambda: dat.transpose((0, 1, 3)))


@use_np
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float16, onp.int32])
@pytest.mark.parametrize('data_shape,axes_workload', [
    [(), [(), None]],
    [(2,), [(0,), None]],
    [(0, 2), [(0, 1), (1, 0)]],
    [(5, 10), [(0, 1), (1, 0), None]],
    [(8, 2, 3), [(2, 0, 1), (0, 2, 1), (0, 1, 2), (2, 1, 0), (-1, 1, 0), None]],
    [(8, 2, 16), [(0, 2, 1), (2, 0, 1), (0, 1, 2), (2, 1, 0), (-1, -2, -3)]],
    [(8, 3, 4, 8), [(0, 2, 3, 1), (1, 2, 3, 0), (0, 3, 2, 1)]],
    [(8, 3, 2, 3, 8), [(0, 1, 3, 2, 4), (0, 1, 2, 3, 4), (4, 0, 1, 2, 3)]],
    [(3, 4, 3, 4, 3, 2), [(0, 1, 3, 2, 4, 5), (2, 3, 4, 1, 0, 5), None]],
    [(3, 4, 3, 4, 3, 2, 2), [(0, 1, 3, 2, 4, 5, 6),
     (2, 3, 4, 1, 0, 5, 6), None]],
    [(3, 4, 3, 4, 3, 2, 3, 2), [(0, 1, 3, 2, 4, 5, 7, 6),
     (2, 3, 4, 1, 0, 5, 7, 6), None]],
])
@pytest.mark.parametrize('grad_req', ['write', 'add'])
def test_np_permute_dims(data_shape, axes_workload, hybridize, dtype, grad_req):
    def np_permute_dims_grad(out_shape, dtype, axes=None):
        ograd = onp.ones(out_shape, dtype=dtype)
        if axes is None or axes == ():
            return onp.transpose(ograd, axes)
        np_axes = onp.array(list(axes))
        permute_dims_axes = onp.zeros_like(np_axes)
        permute_dims_axes[np_axes] = onp.arange(len(np_axes))
        return onp.transpose(ograd, tuple(list(permute_dims_axes)))

    class TestPermuteDims(HybridBlock):
        def __init__(self, axes=None):
            super(TestPermuteDims, self).__init__()
            self.axes = axes

        def forward(self, a):
            return np.permute_dims(a, self.axes)

    for axes in axes_workload:
        test_trans = TestPermuteDims(axes)
        if hybridize:
            test_trans.hybridize()
        x = np.random.normal(0, 1, data_shape).astype(dtype)
        x = x.astype(dtype)
        x.attach_grad(grad_req=grad_req)
        if grad_req == 'add':
            x.grad[()] = np.random.normal(0, 1, x.grad.shape).astype(x.grad.dtype)
            x_grad_np = x.grad.asnumpy()
        np_out = onp.transpose(x.asnumpy(), axes)
        with mx.autograd.record():
            mx_out = test_trans(x)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)
        mx_out.backward()
        np_backward = np_permute_dims_grad(np_out.shape, dtype, axes)
        if grad_req == 'add':
            assert_almost_equal(x.grad.asnumpy(), np_backward + x_grad_np,
                                rtol=1e-3, atol=1e-5, use_broadcast=False)
        else:
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5, use_broadcast=False)


@use_np
def test_np_meshgrid():
    nx, ny = (4, 5)
    x = np.array(onp.linspace(0, 1, nx), dtype=np.float32)
    y = np.array(onp.linspace(0, 1, ny), dtype=np.float32)
    z = np.ones(())
    xv, yv, zv = np.meshgrid(x, y, z)
    xv_expected, yv_expected, zv_expected = onp.meshgrid(x.asnumpy(), y.asnumpy(), z.asnumpy())
    assert same(xv.asnumpy(), xv_expected)
    assert same(yv.asnumpy(), yv_expected)
    assert same(zv.asnumpy(), zv_expected)


@use_np
@pytest.mark.parametrize('shapes', [
    [(), (2, 1), (1, 3), (4, 1, 1), (5, 4, 2, 3)],
    [(0,), (), (2, 1), (1, 0), (3, 2, 1)]
])
def test_np_broadcast_arrays(shapes):
    arrays_np = [onp.random.randint(low=0, high=1000, size=shape, dtype=onp.int32) for shape in shapes]
    arrays_mx = [np.array(arr, dtype=arr.dtype) for arr in arrays_np]
    expected_rets = onp.broadcast_arrays(*arrays_np)
    rets = np.broadcast_arrays(*arrays_mx)
    for expected_ret, ret in zip(expected_rets, rets):
        assert same(expected_ret, ret.asnumpy())


@use_np
def test_np_tile():
    config = [
        ((), ()),
        ((), 0),
        ((), (2, 0)),
        ((), (2, 3)),
        ((4, 2), (2,)),
        ((4, 2), (2, 3)),
        ((4, 2), (2, 1, 4)),
        ((4, 2), (2, 3, 4)),
        ((4, 2), (2, 0)),
        ((4, 2), (2, 0, 3)),
        ((4, 2), (2, 0, 3)),
        ((4, 0), (2, 0, 3)),
    ]

    class TestTile(HybridBlock):
        def __init__(self, reps):
            super(TestTile, self).__init__()
            self._reps = reps

        def forward(self, x):
            return np.tile(x, reps=self._reps)

    for shape, reps in config:
        data_np = onp.random.randint(low=0, high=1000, size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = onp.tile(data_np, reps=reps)
        ret_mx = np.tile(data_mx, reps=reps)
        assert same(ret_mx.asnumpy(), ret_np)

        net = TestTile(reps)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            ret_mx = net(data_mx)
            assert same(ret_mx.asnumpy(), ret_np)


@use_np
def test_np_tril():
    # numpy tril does not support scalar array (zero-dim)
    config = [
        ((4, 2), 3),
        ((4, 2), 9),
        ((4, 2), 0),
        ((4, 2), -1),
        ((4, 5, 6), 0),
        ((4, 5, 6), 5),
        ((4, 5, 6), 2),
        ((4, 5, 6), -2),
        ((4, 5, 6), -5),
        ((4, 0), 0),
        ((4, 0), 2),
        ((4, 0), 4),
        ((4, 0), -3),
        ((4, 0, 5), 0),
        ((4, 0, 5), 1),
        ((4, 0, 5), 5),
        ((4, 0, 5), -3),
        ((3, ), 0),
        ((3, ), 2),
        ((3, ), 5)
    ]

    class TestTril(HybridBlock):
        def __init__(self, k):
            super(TestTril, self).__init__()
            self._k = k

        def forward(self, x):
            return np.tril(x, k=self._k)

    for prefix in [1, -1]:
        for shape, k in config:
            data_np = onp.random.uniform(size=shape).astype(onp.float32)
            data_mx = np.array(data_np, dtype=data_np.dtype)
            data_mx.attach_grad()
            ret_np = onp.tril(data_np, k*prefix)
            with mx.autograd.record():
                ret_mx = np.tril(data_mx, k*prefix)
            assert same(ret_mx.asnumpy(), ret_np)
            ret_mx.backward()
            if len(shape) == 2:
                grad_np = onp.tri(*shape, k=k*prefix)
                assert same(data_mx.grad.asnumpy(), grad_np)
            if len(shape) == 1:
                grad_np = onp.tri(*shape, k=k*prefix)
                grad_np = grad_np.sum(axis=0, keepdims=False)
                assert same(data_mx.grad.asnumpy(), grad_np)

            net = TestTril(k*prefix)
            for hybrid in [False, True]:
                if hybrid:
                    net.hybridize()
                ret_mx = net(data_mx)
                assert same(ret_mx.asnumpy(), ret_np)


@use_np
def test_np_triu():
    # numpy triu does not support scalar array (zero-dim)
    config = [
        ((4, 2), 3),
        ((4, 2), 9),
        ((4, 2), 0),
        ((4, 2), -1),
        ((4, 5, 6), 0),
        ((4, 5, 6), 5),
        ((4, 5, 6), 2),
        ((4, 5, 6), -2),
        ((4, 5, 6), -5),
        ((4, 0), 0),
        ((4, 0), 2),
        ((4, 0), 4),
        ((4, 0), -3),
        ((4, 0, 5), 0),
        ((4, 0, 5), 1),
        ((4, 0, 5), 5),
        ((4, 0, 5), -3),
        ((3, ), 0),
        ((3, ), 2),
        ((3, ), 5)
    ]

    class TestTriu(HybridBlock):
        def __init__(self, k):
            super(TestTriu, self).__init__()
            self._k = k

        def forward(self, x):
            return np.triu(x, k=self._k)

    for prefix in [1, -1]:
        for shape, k in config:
            data_np = onp.random.uniform(size=shape).astype(onp.float32)
            data_mx = np.array(data_np, dtype=data_np.dtype)
            data_mx.attach_grad()
            ret_np = onp.triu(data_np, k*prefix)
            with mx.autograd.record():
                ret_mx = np.triu(data_mx, k*prefix)
            assert same(ret_mx.asnumpy(), ret_np)
            ret_mx.backward()
            if len(shape) == 2:
                grad_np = onp.triu(onp.ones_like(data_np), k*prefix)
                assert same(data_mx.grad.asnumpy(), grad_np)
            if len(shape) == 1:
                grad_np = onp.triu(onp.ones(shape), k*prefix)
                grad_np = grad_np.sum(axis=0, keepdims=False)
                assert same(data_mx.grad.asnumpy(), grad_np)

            net = TestTriu(k*prefix)
            for hybrid in [False, True]:
                if hybrid:
                    net.hybridize()
                ret_mx = net(data_mx)
                assert same(ret_mx.asnumpy(), ret_np)


@use_np
def test_np_unary_funcs():
    def check_unary_func(func, ref_grad, shape, low, high):
        class TestUnary(HybridBlock):
            def __init__(self, func):
                super(TestUnary, self).__init__()
                self._func = func

            def forward(self, a, *args, **kwargs):
                return getattr(np, self._func)(a)

        np_func = getattr(onp, func)
        np_test_data = onp.random.uniform(low, high, shape).astype(onp.float32)
        mx_test_data = mx.numpy.array(np_test_data)
        for hybridize in [True, False]:
            mx_func = TestUnary(func)
            if hybridize:
                mx_func.hybridize()
            if ref_grad:
                mx_test_data.attach_grad()
            np_out = np_func(np_test_data)
            with mx.autograd.record():
                y = mx_func(mx_test_data)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            if np_out.dtype == np.bool_:
                assert y.dtype == np.bool_

            if ref_grad:
                y.backward()
                assert_almost_equal(mx_test_data.grad.asnumpy(), ref_grad(np_test_data), rtol=1e-1, atol=1e-2, equal_nan=True)

        np_out = getattr(onp, func)(np_test_data)
        mx_out = getattr(mx.np, func)(mx_test_data)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, where=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  subok=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  dtype=onp.int8)
        assertRaises(TypeError, getattr(np, func), mx_test_data,  dtype="abcdefg")
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  casting='safe')
        assertRaises(TypeError, getattr(np, func), mx_test_data,  casting='mxnet')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  order='C')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  order='mxnet')

    funcs = {
        'absolute' : (lambda x: -1. * (x < 0) + (x > 0), -1.0, 1.0),
        'logical_not' : (None, -1.0, 1.0),
        'negative' : (lambda x: -1. * onp.ones(x.shape), -1.0, 1.0),
        'positive' : (lambda x: onp.ones(x.shape), -1.0, 1.0),
        'reciprocal' : (lambda x: -1. / (x ** 2), 0.01, 1.0),
        'sign' : (None, -1.0, 1.0),
        'square' : (lambda x: 2.0 * x, -1.0, 1.0),
    }
    if has_tvm_ops():
        funcs['rad2deg'] = (lambda x: 180. / onp.pi * onp.ones(x.shape), -1.0, 1.0)
        funcs['deg2rad'] = (lambda x: onp.pi / 180. * onp.ones(x.shape), -1.0, 1.0)
    ndim = random.choice([2, 3, 4])
    for shape in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]:
        for func, func_data in funcs.items():
            ref_grad, low, high = func_data
            check_unary_func(func, ref_grad, shape, low, high)


@use_np
def test_negation():
    class TestNegation(HybridBlock):
        def forward(self, a):
            return -a
    mx_func = TestNegation()
    for dtype in [onp.int8, onp.int32, onp.float16, onp.float32, onp.float64]:
        np_test_data = onp.random.uniform(-1, 1, (5, 5)).astype(dtype)
        for hybridize in [True, False]:
            mx_test_data = mx.numpy.array(np_test_data, dtype=dtype)
            if hybridize:
                mx_func.hybridize()
            y = mx_func(mx_test_data)
            assert y.shape == (5, 5)
            assert y.dtype == dtype
            assert_almost_equal(y.asnumpy(), -np_test_data)


@use_np
@retry(3)
@pytest.mark.parametrize('func,ref_grad,low,high', [
    ('cbrt', lambda x: 1. / (3. * onp.cbrt(x) ** 2), -1.0, 1.0),
    ('ceil', None, -10.0, 10.0),
    ('exp', lambda x: onp.exp(x), -1.0, 1.0),
    ('expm1', lambda x: onp.exp(x), -1.0, 1.0),
    ('fix', None, -10.0, 10.0),
    ('floor', None, -10.0, 10.0),
    ('log', lambda x: 1.0 / x, 0.1, 5.0),
    ('log10', lambda x: 1.0 / (x * onp.log(10)), 0.1, 10.0),
    ('log1p', lambda x: 1.0 / (1.0 + x), -0.9, 5.0),
    ('log2', lambda x: 1.0 / (x * onp.log(2)), 0.1, 2.0),
    ('rint', None, -5.0, 5.0),
    ('sqrt', lambda x: 0.5 / onp.sqrt(x), 0.001, 10.0),
    ('trunc', None, -5.0, 5.0),
    ('sin', lambda x: onp.cos(x), -1.0, 1.0),
    ('cos', lambda x: -onp.sin(x), -1.0, 1.0),
    ('tan', lambda x: onp.tan(x) ** 2 + 1.0, -1.0, 1.0),
    ('arcsin', lambda x: 1. / (1. - x ** 2) ** (1. / 2.), -1.0, 1.0),
    ('arccos', lambda x: -1. / (1. - x ** 2.) ** (1. / 2.), -1.0, 1.0),
    ('arctan', lambda x: 1. / (x ** 2. + 1.), -1.0, 1.0),
    ('degrees', lambda x: 180. / onp.pi * onp.ones(x.shape), -1.0, 1.0),
    ('radians', lambda x: onp.pi / 180. * onp.ones(x.shape), -1.0, 1.0),
    ('sinh', lambda x: onp.cosh(x), -1.0, 1.0),
    ('cosh', lambda x: onp.sinh(x), -1.0, 1.0),
    ('tanh', lambda x: 1. - onp.tanh(x) ** 2, -1.0, 1.0),
    ('arcsinh', lambda x: 1./(x**2 + 1.)**(1./2.), -1.0, 1.0),
    ('arccosh', lambda x: 1./(x**2 - 1.)**(1./2.), 2.0, 5.0),
    ('arctanh', lambda x: -1./(x**2 - 1.), -0.99, 0.99)
])
@pytest.mark.parametrize('ndim', [2, 3, 4])
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int8', 'uint8', 'int32', 'int64', 'bool'])
def test_np_mixedType_unary_funcs(func, ref_grad, low, high, ndim, dtype):
    class TestMixedUnary(HybridBlock):
        def __init__(self, func):
            super(TestMixedUnary, self).__init__()
            self._func = func

        def forward(self, a, *args, **kwargs):
            return getattr(np, self._func)(a)

    import math

    shapes = [i for i in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]];
    for shape in shapes:
        print(func, dtype, shape)
        rtol = 1e-2 if dtype == np.float16 else 1e-3
        atol = 1e-4 if dtype == np.float16 else 1e-5
        # get rid of warning: divide by zero
        if((func=='log' or func=='log10' or func=='log2') and
            (dtype=='int8' or dtype=='uint8' or dtype=='int32' or
            dtype=='int64')):
            low = 1
        if (func=='arctanh' and dtype=='bool'):
            continue
        np_func = getattr(onp, func)
        mx_func = TestMixedUnary(func)
        np_test_data = onp.random.uniform(low, high, shape).astype(dtype)
        mx_test_data = np.array(np_test_data)
        for hybridize in [True, False]:
            if hybridize:
                mx_func.hybridize()
            if ref_grad:
                mx_test_data.attach_grad()
            np_out = np_func(np_test_data)
            with mx.autograd.record():
                y = mx_func(mx_test_data)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            if np_out.dtype == np.bool_:
                assert y.dtype == np.bool_

            if ref_grad and (dtype == 'float16' or dtype == 'float32' or dtype == 'float64'):
                y.backward()
                assert_almost_equal(mx_test_data.grad.asnumpy(), ref_grad(np_test_data), rtol=1e-1, atol=1e-2, equal_nan=True)

        np_out = getattr(onp, func)(np_test_data)
        mx_out = getattr(mx.np, func)(mx_test_data)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, where=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, subok=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, dtype=onp.int8)
        assertRaises(TypeError, getattr(np, func), mx_test_data, dtype="abcdefg")
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, casting='safe')
        assertRaises(TypeError, getattr(np, func), mx_test_data, casting='mxnet')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, order='C')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, order='mxnet')


@use_np
@pytest.mark.parametrize('ndim', [2, 3, 4])
@pytest.mark.parametrize('func,low,high', [
    ('bitwise_not', -5, 5),
    ('invert', -5, 5),
])
def test_np_bitwise_not(func, low, high, ndim):
    def check_unary_func(func, shape, low, high):
        class TestUnary(HybridBlock):
            def __init__(self, func):
                super(TestUnary, self).__init__()
                self._func = func

            def forward(self, a, *args, **kwargs):
                return getattr(np, self._func)(a)

        np_func = getattr(onp, func)
        mx_func = TestUnary(func)
        np_test_data = onp.random.uniform(low, high, shape).astype(onp.int32)
        mx_test_data = mx.numpy.array(np_test_data).astype(onp.int32)
        for hybridize in [True, False]:
            if hybridize:
                mx_func.hybridize()
            np_out = np_func(np_test_data)
            with mx.autograd.record():
                y = mx_func(mx_test_data)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            if np_out.dtype == np.bool_:
                assert y.dtype == np.bool_

        np_out = getattr(onp, func)(np_test_data)
        mx_out = getattr(mx.np, func)(mx_test_data)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, where=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  subok=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  dtype=onp.int8)
        assertRaises(TypeError, getattr(np, func), mx_test_data,  dtype="abcdefg")
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  casting='safe')
        assertRaises(TypeError, getattr(np, func), mx_test_data,  casting='mxnet')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  order='C')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  order='mxnet')

    shape = random.choice([rand_shape_nd(ndim, dim=3), (1, 0, 2)])
    for shape in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]:
        check_unary_func(func, shape, low, high)


@use_np
@pytest.mark.parametrize('ndim', [2, 3, 4])
@pytest.mark.parametrize('func,low,high', [
    ('left_shift', -5, 5),
    ('right_shift', -5, 5),
])
def test_np_bitwise_shift(func, low, high, ndim):
    def check_unary_func(func, shape, low, high):
        class TestUnary(HybridBlock):
            def __init__(self, func):
                super(TestUnary, self).__init__()
                self._func = func

            def forward(self, a, b, *args, **kwargs):
                return getattr(np, self._func)(a, b)

        np_func = getattr(onp, func)
        mx_func = TestUnary("bitwise_" + func)
        np_test_data1 = onp.random.randint(low, high, shape).astype(onp.int64)
        np_test_data2 = onp.random.randint(low + 5, high + 5, shape).astype(onp.int64)
        mx_test_data1 = mx.numpy.array(np_test_data1).astype(onp.int64)
        mx_test_data2 = mx.numpy.array(np_test_data2).astype(onp.int64)
        for hybridize in [True, False]:
            if hybridize:
                mx_func.hybridize()
            np_out = np_func(np_test_data1, np_test_data2)
            with mx.autograd.record():
                y = mx_func(mx_test_data1, mx_test_data2)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            if np_out.dtype == np.bool_:
                assert y.dtype == np.bool_

        np_out = getattr(onp, func)(np_test_data1, np_test_data2)
        mx_out = getattr(mx.np, "bitwise_" + func)(mx_test_data1, mx_test_data2)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, where=False)
        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, subok=False)
        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, dtype=onp.int8)
        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, dtype="abcdefg")
        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, casting='safe')
        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, casting='mxnet')
        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, order='C')
        assertRaises(TypeError, getattr(np, "bitwise_" + func), mx_test_data1, mx_test_data2, order='mxnet')

    shape = random.choice([rand_shape_nd(ndim, dim=3), (1, 0, 2)])
    for shape in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]:
        check_unary_func(func, shape, low, high)


@use_np
def test_np_binary_funcs():
    def check_binary_func(func, lshape, rshape, low, high, lgrads, rgrads=None, alltypes=None):
        class TestBinary(HybridBlock):
            def __init__(self, func):
                super(TestBinary, self).__init__()
                self._func = func

            def forward(self, a, b, *args, **kwargs):
                return getattr(np, self._func)(a, b)

        np_func = getattr(onp, func)
        mx_func = TestBinary(func)
        alltypes = alltypes if alltypes else [[onp.float16, onp.float32, onp.float64]]
        for dtypes, lgrad, rgrad in zip(alltypes, lgrads, rgrads if rgrads else lgrads):
            for dtype in dtypes:
                ldtype = rdtype = dtype
                if isinstance(dtype, tuple):
                    assert len(dtype) == 2
                    ldtype, rdtype = dtype
                npldtype = ldtype if dtype != onp.float16 else onp.float32
                nprdtype = rdtype if dtype != onp.float16 else onp.float32
                np_test_x1 = onp.random.uniform(low, high, lshape).astype(ldtype).astype(npldtype)
                np_test_x2 = onp.random.uniform(low, high, rshape).astype(rdtype).astype(nprdtype)
                mx_test_x1 = mx.numpy.array(np_test_x1, dtype=ldtype)
                mx_test_x2 = mx.numpy.array(np_test_x2, dtype=rdtype)
                for hybridize in [True, False]:
                    if hybridize:
                        mx_func.hybridize()
                    if lgrad:
                        mx_test_x1.attach_grad()
                        mx_test_x2.attach_grad()
                    np_out = np_func(np_test_x1, np_test_x2)
                    with mx.autograd.record():
                        y = mx_func(mx_test_x1, mx_test_x2)
                    assert y.shape == np_out.shape
                    assert_almost_equal(y.asnumpy(), np_out.astype(y.dtype), rtol=1e-3, atol=1e-5,
                                        use_broadcast=False, equal_nan=True)

                    if lgrad:
                        y.backward()
                        assert_almost_equal(mx_test_x1.grad.asnumpy(),
                                            collapse_sum_like(lgrad(y.asnumpy(), np_test_x1, np_test_x2), mx_test_x1.shape),
                                            rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)
                        if rgrads is None:
                            assert_almost_equal(mx_test_x2.grad.asnumpy(),
                                                collapse_sum_like(rgrad(y.asnumpy(), np_test_x2, np_test_x1), mx_test_x2.shape),
                                                rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)
                        else:
                            assert_almost_equal(mx_test_x2.grad.asnumpy(),
                                                collapse_sum_like(rgrad(y.asnumpy(), np_test_x1, np_test_x2), mx_test_x2.shape),
                                                rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)

                np_out = getattr(onp, func)(np_test_x1, np_test_x2)
                mx_out = getattr(mx.np, func)(mx_test_x1, mx_test_x2)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out.astype(mx_out.dtype), rtol=1e-3, atol=1e-5,
                                    use_broadcast=False, equal_nan=True)

                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2, where=False)
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  subok=False)
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  dtype=onp.int8)
                assertRaises(TypeError, getattr(np, func), mx_test_x1, mx_test_x2,  dtype="abcdefg")
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  casting='safe')
                assertRaises(TypeError, getattr(np, func), mx_test_x1, mx_test_x2,  casting='mxnet')
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  order='C')
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  order='mxnet')

    funcs = {
        'add': (-1.0, 1.0, [lambda y, x1, x2: onp.ones(y.shape)], None),
        'subtract':
        (-1.0, 1.0, [lambda y, x1, x2: onp.ones(y.shape)],
                    [lambda y, x1, x2: -onp.ones(y.shape)]),
        'multiply': (-1.0, 1.0, [lambda y, x1, x2: onp.broadcast_to(x2, y.shape)],
                                [lambda y, x1, x2: onp.broadcast_to(x1, y.shape)]),
        'divide': (0.1, 1.0, [lambda y, x1, x2: onp.ones(y.shape) / x2],
                   [lambda y, x1, x2: -x1 / (x2 * x2)]),
        'floor_divide': (0.1, 1.0, [lambda y, x1, x2: onp.zeros(y.shape)],
                 [lambda y, x1, x2: onp.zeros(y.shape)]),
        'mod': (1.0, 10.0,
                [lambda y, x1, x2: onp.ones(y.shape),
                 lambda y, x1, x2: onp.zeros(y.shape)],
                [lambda y, x1, x2: -onp.floor(x1 / x2),
                 lambda y, x1, x2: onp.zeros(y.shape)],
                [[onp.float16, onp.float32, onp.float64], [onp.int32]]),
        'fmod': (1.0, 10.0,
                [lambda y, x1, x2: onp.ones(y.shape),
                 lambda y, x1, x2: onp.zeros(y.shape)],
                [lambda y, x1, x2: -onp.floor(x1 / x2),
                 lambda y, x1, x2: onp.zeros(y.shape)],
                [[onp.float16, onp.float32, onp.float64], [onp.int32]]),
        'remainder': (1.0, 10.0,
                      [lambda y, x1, x2: onp.ones(y.shape),
                       lambda y, x1, x2: onp.zeros(y.shape)],
                      [lambda y, x1, x2: -onp.floor(x1 / x2),
                       lambda y, x1, x2: onp.zeros(y.shape)],
                      [[onp.float16, onp.float32, onp.float64], [onp.int32]]),
        'power': (1.0, 3.0, [lambda y, x1, x2: onp.power(x1, x2 - 1.0) * x2],
                             [lambda y, x1, x2: onp.power(x1, x2) * onp.log(x1)]),
        'gcd': (-100, 100, [None], None, [[onp.int32]]),
        'lcm': (-100, 100, [None], None, [[onp.int32]]),
        'bitwise_and': (-100, 100, [None], None, [[onp.int32]]),
        'bitwise_xor': (-100, 100, [None], None, [[onp.int32]]),
        'bitwise_or': (-100, 100, [None], None, [[onp.int32]]),
        'maximum': (-10, 10, [lambda y, x1, x2: onp.ones(y.shape) * (x1 >= x2)],
                             [lambda y, x1, x2: onp.ones(y.shape) * (x1 < x2)],
                             [[onp.int32, onp.float16, onp.float32, onp.float64]]),
        'fmax': (-1, 1, [lambda y, x1, x2: onp.ones(y.shape) * (x1 >= x2)],
                        [lambda y, x1, x2: onp.ones(y.shape) * (x1 < x2)]),
        'minimum': (-10, 10, [lambda y, x1, x2: onp.ones(y.shape) * (x1 <= x2)],
                             [lambda y, x1, x2: onp.ones(y.shape) * (x1 > x2)],
                             [[onp.int32, onp.float16, onp.float32, onp.float64]]),
        'fmin': (-1, 1, [lambda y, x1, x2: onp.ones(y.shape) * (x1 <= x2)],
                        [lambda y, x1, x2: onp.ones(y.shape) * (x1 > x2)]),
        'copysign': (-1, 1,
                     [lambda y, x1, x2: onp.ones(y.shape) * (((x1 * x2) >= 0).astype(onp.float32) - ((x1 * x2) < 0).astype(onp.float32))],
                     [lambda y, x1, x2: onp.zeros(y.shape)]),
        'arctan2': (-1, 1, [lambda y, x1, x2: x2 / (onp.square(x1) + onp.square(x2))],
                           [lambda y, x1, x2: -x1 / (onp.square(x1) + onp.square(x2))]),
        'hypot': (-1, 1, [lambda y, x1, x2: x1 / y],
                         [lambda y, x1, x2: x2 / y]),
        'ldexp': (-3, 3, [None], None, [[onp.int32]]),
        'logaddexp': (-10, 10, [lambda y, x1, x2: onp.exp(x1) / (onp.exp(x1) + onp.exp(x2))],
                               [lambda y, x1, x2: onp.exp(x2) / (onp.exp(x1) + onp.exp(x2))])
    }
    if is_op_runnable():
        funcs['logical_and'] = (-100, 100, [None], None, [[onp.float32, onp.float64]])
        funcs['logical_or'] = (-100, 100, [None], None, [[onp.float32, onp.float64]])
        funcs['logical_xor'] = (-100, 100, [None], None, [[onp.float32, onp.float64]])
    shape_pairs = [((3, 2), (3, 2)),
                   ((3, 2), (3, 1)),
                   ((3, 1), (3, 0)),
                   ((0, 2), (1, 2)),
                   ((2, 3, 4), (3, 1)),
                   ((2, 3), ()),
                   ((), (2, 3))]
    for lshape, rshape in shape_pairs:
        for func, func_data in funcs.items():
            dtypes = None
            assert (len(func_data) == 4 or len(func_data) == 5)
            if len(func_data) is 4:
                low, high, lgrads, rgrads = func_data
            else:
                low, high, lgrads, rgrads, dtypes = func_data
            check_binary_func(func, lshape, rshape, low, high, lgrads, rgrads, dtypes)


@use_np
def test_np_mixed_precision_binary_funcs():
    itypes = [np.bool, np.int8, np.int32, np.int64]
    ftypes = [np.float16, np.float32, np.float64]
    def check_mixed_precision_binary_func(func, low, high, lshape, rshape, lgrad, rgrad, ltype, rtype):
        class TestMixedBinary(HybridBlock):
            def __init__(self, func):
                super(TestMixedBinary, self).__init__()
                self._func = func

            def forward(self, a, b, *args, **kwargs):
                return getattr(np, self._func)(a, b)

        if (func in ['multiply', 'mod', 'equal', 'not_equal', 'greater',
                    'greater_equal', 'less', 'less_equal']) and \
            (lshape == () or rshape == ()) :
        # the behaviors of infer type in dealing with the input shape of '()' are different between np and onp
        # for example,
        # mx_test_x1 = np.random.uniform(-2, 2, (2,3)).astype(np.float32)
        # mx_test_x2 = np.random.uniform(-2, 2, ()).astype(np.float16)
        # np_out = onp.mod(mx_test_x1.asnumpy(), mx_test_x2.asnumpy()) # float16
        # mx_out = np.mod(mx_test_x1, mx_test_x2) # float32

        # logcial ops: when two numbers are only different in precision, NumPy also has a weird behavior
        # for example,
        # a = np.array([[1.441]], dtype = np.float16)
        # b = np.array(1.4413278, dtype = np.float32)
        # c = np.array([1.4413278], dtype = np.float32)
        # np.greater(a,b), np.greater(a,c) # True True
        # onp.greater(a.asnumpy(),b.asnumpy()), onp.greater(a.asnumpy(),c.asnumpy()) # False True

        # thus, skip the tests
            return

        np_func = getattr(onp, func)
        mx_func = TestMixedBinary(func)
        np_test_x1 = onp.random.uniform(low, high, lshape).astype(ltype)
        np_test_x2 = onp.random.uniform(low, high, rshape).astype(rtype)
        mx_test_x1 = mx.numpy.array(np_test_x1, dtype=ltype)
        mx_test_x2 = mx.numpy.array(np_test_x2, dtype=rtype)
        rtol = 1e-2 if ltype is np.float16 or rtype is np.float16 else 1e-3
        atol = 1e-3 if ltype is np.float16 or rtype is np.float16 else 1e-5
        for hybridize in [True, False]:
            if hybridize:
                mx_func.hybridize()
            if lgrad:
                mx_test_x1.attach_grad()
                mx_test_x2.attach_grad()
            np_out = np_func(np_test_x1, np_test_x2)
            with mx.autograd.record():
                y = mx_func(mx_test_x1, mx_test_x2)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out.astype(y.dtype), rtol=rtol, atol=atol,
                                use_broadcast=False, equal_nan=True)

            if lgrad:
                if (ltype in itypes) and (rtype in itypes):
                    continue
                y.backward()
                if ltype not in itypes:
                    assert_almost_equal(mx_test_x1.grad.asnumpy(),
                                        collapse_sum_like(lgrad(y.asnumpy(), np_test_x1, np_test_x2), mx_test_x1.shape),
                                        rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)
                if rtype not in itypes:
                    if rgrad is None:
                        assert_almost_equal(mx_test_x2.grad.asnumpy(),
                                            collapse_sum_like(rgrad(y.asnumpy(), np_test_x2, np_test_x1), mx_test_x2.shape),
                                            rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)
                    else:
                        assert_almost_equal(mx_test_x2.grad.asnumpy(),
                                            collapse_sum_like(rgrad(y.asnumpy(), np_test_x1, np_test_x2), mx_test_x2.shape),
                                            rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)


        np_out = getattr(onp, func)(np_test_x1, np_test_x2)
        mx_out = getattr(mx.np, func)(mx_test_x1, mx_test_x2)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out.astype(mx_out.dtype), rtol=rtol, atol=atol,
                            use_broadcast=False, equal_nan=True)

    funcs = {
        'add': (-1.0, 1.0, lambda y, x1, x2: onp.ones(y.shape),
                           lambda y, x1, x2: onp.ones(y.shape)),
        'subtract': (-1.0, 1.0, lambda y, x1, x2: onp.ones(y.shape),
                                lambda y, x1, x2: onp.ones(y.shape) * -1),
        'multiply': (-1.0, 1.0, lambda y, x1, x2: onp.broadcast_to(x2, y.shape),
                                lambda y, x1, x2: onp.broadcast_to(x1, y.shape)),
        'mod': (1.0, 5.0, None, None),
        'power': (1.0, 3.0, lambda y, x1, x2: onp.power(x1, x2 - 1.0) * x2,
                            lambda y, x1, x2: onp.power(x1, x2) * onp.log(x1)),
        'equal': (0.0, 2.0, None, None),
        'not_equal': (0.0, 2.0, None, None),
        'greater': (0.0, 2.0, None, None),
        'less': (0.0, 2.0, None, None),
        'greater_equal': (0.0, 2.0, None, None),
        'less_equal': (0.0, 2.0, None, None),
        'logical_and': (0.0, 2.0, None, None),
        'logical_or': (0.0, 2.0, None, None),
        'logical_xor': (0.0, 2.0, None, None),
    }

    shape_pairs = [((3, 2), (3, 2)),
                   ((3, 2), (3, 1)),
                   ((3, 0), (3, 0)),
                   ((3, 1), (3, 0)),
                   ((0, 2), (1, 2)),
                   ((2, 3, 4), (3, 1)),
                   ((2, 3), ()),
                   ((), (2, 3))]

    itypes = [np.bool, np.int8, np.int32, np.int64]
    ftypes = [np.float16, np.float32, np.float64]
    for func, func_data in funcs.items():
        low, high, lgrad, rgrad = func_data
        for lshape, rshape in shape_pairs:
            for type1, type2 in itertools.product(itypes, ftypes):
                check_mixed_precision_binary_func(func, low, high, lshape, rshape, lgrad, rgrad, type1, type2)
                check_mixed_precision_binary_func(func, low, high, lshape, rshape, lgrad, rgrad, type2, type1)

            for type1, type2 in itertools.product(ftypes, ftypes):
                if type1 == type2:
                    continue
                check_mixed_precision_binary_func(func, low, high, lshape, rshape, lgrad, rgrad, type1, type2)

            if func == 'subtract' or func == 'mod':
                continue
            for type1, type2 in itertools.product(itypes, itypes):
                if type1 == type2:
                    continue
                check_mixed_precision_binary_func(func, low, high, lshape, rshape, lgrad, rgrad, type1, type2)

@use_np
def test_np_mixed_mxnp_op_funcs():
    # generate onp & mx_np in same type
    _np = onp.array([1,2,3,4,5]).astype("int64")
    mx_np = mx.np.array([1,2,3,4,5]).astype("int64")
    # inplace onp mx_np
    _np += mx_np
    assert isinstance(_np, onp.ndarray)
    _np -= mx_np
    assert isinstance(_np, onp.ndarray)
    _np *= mx_np
    assert isinstance(_np, onp.ndarray)
    # inplace mx_np onp
    mx_np ^= _np
    assert isinstance(mx_np, mx.np.ndarray)
    mx_np |= _np
    assert isinstance(mx_np, mx.np.ndarray)
    mx_np &= _np
    assert isinstance(mx_np, mx.np.ndarray)
    # mxnp onp
    out = mx_np << _np
    assert isinstance(out, mx.np.ndarray)
    out = mx_np >> _np
    assert isinstance(out, mx.np.ndarray)
    out = mx_np != _np
    assert isinstance(out, mx.np.ndarray)
    # onp mxnp
    out = _np == mx_np
    assert isinstance(out, mx.np.ndarray)
    out = _np >= mx_np
    assert isinstance(out, mx.np.ndarray)
    out = _np < mx_np
    assert isinstance(out, mx.np.ndarray)
    _np = onp.array([1,2,3,4,5]).astype("float32")
    mx_np = mx.np.array([1,2,3,4,5]).astype("float32")
    out = _np @ mx_np
    assert isinstance(out, mx.np.ndarray)
    out = _np / mx_np
    assert isinstance(out, mx.np.ndarray)

@use_np
def test_np_binary_scalar_funcs():
    itypes = [np.int8, np.int32, np.int64]
    def check_binary_scalar_func(func, low, high, lshape, lgrad, ltype, scalar_is_int, hybridize):
        class TestBinaryScalar(HybridBlock):
            def __init__(self, func, scalar):
                super(TestBinaryScalar, self).__init__()
                self._func = func
                self._scalar = scalar

            def forward(self, a, *args, **kwargs):
                return getattr(np, self._func)(a, self._scalar)

        np_test_x1 = onp.random.uniform(low, high, lshape).astype(ltype)
        np_test_x2 = int(onp.random.uniform(low, high)) if scalar_is_int else onp.random.uniform(low, high)
        mx_test_x1 = np.array(np_test_x1, dtype=ltype)
        mx_test_x2 = np_test_x2
        np_func = getattr(onp, func)
        mx_func = TestBinaryScalar(func, mx_test_x2)
        if hybridize:
            mx_func.hybridize()
        rtol = 1e-2 if ltype is np.float16 else 1e-3
        atol = 1e-3 if ltype is np.float16 else 1e-5
        if ltype not in itypes:
            if lgrad:
                mx_test_x1.attach_grad()
            np_out = np_func(np_test_x1, np_test_x2)
            with mx.autograd.record():
                y = mx_func(mx_test_x1)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out.astype(y.dtype), rtol=rtol, atol=atol)
            if lgrad:
                y.backward()
                assert_almost_equal(mx_test_x1.grad.asnumpy(),
                                    collapse_sum_like(lgrad(y.asnumpy(), np_test_x1, np_test_x2), mx_test_x1.shape),
                                    rtol=rtol, atol=atol, equal_nan=True, use_broadcast=False)

        # Test imperative
        np_out = getattr(onp, func)(np_test_x1, np_test_x2)
        mx_out = getattr(mx.np, func)(mx_test_x1, mx_test_x2)
        assert mx_out.shape == np_out.shape
        assert mx_out.asnumpy().dtype == np_out.dtype
        assert_almost_equal(mx_out.asnumpy(), np_out.astype(mx_out.dtype), rtol=rtol, atol=atol)

    funcs = {
        'add': (-1.0, 1.0, None),
        'subtract': (-1.0, 1.0, None),
        'multiply': (-1.0, 1.0, lambda y, x1, x2: onp.broadcast_to(x2, y.shape)),
        'power': (1.0, 5.0, lambda y, x1, x2: onp.power(x1, x2 - 1.0) * x2),
    }

    shapes = [(3, 2), (3, 0), (3, 1), (0, 2), (2, 3, 4)]
    ltypes = [np.int32, np.int64, np.float16, np.float32, np.float64]
    flags = [True, False]
    for func, func_data in funcs.items():
        low, high, lgrad = func_data
        for shape, ltype, is_int, hybridize in itertools.product(shapes, ltypes, flags, flags):
                check_binary_scalar_func(func, low, high, shape, lgrad, ltype, is_int, hybridize)


@use_np
def test_np_boolean_binary_funcs():
    def check_boolean_binary_func(func, mx_x1, mx_x2):
        class TestBooleanBinary(HybridBlock):
            def __init__(self, func):
                super(TestBooleanBinary, self).__init__()
                self._func = func

            def forward(self, a, b, *args, **kwargs):
                return getattr(np, self._func)(a, b)

        np_x1 = mx_x1.asnumpy()
        np_x2 = mx_x2.asnumpy()
        np_func = getattr(onp, func)
        mx_func = TestBooleanBinary(func)
        for hybridize in [True, False]:
            if hybridize:
                mx_func.hybridize()
            np_out = np_func(np_x1, np_x2)
            with mx.autograd.record():
                y = mx_func(mx_x1, mx_x2)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out.astype(y.dtype), rtol=1e-3, atol=1e-20,
                                use_broadcast=False, equal_nan=True)

        np_out = getattr(onp, func)(np_x1, np_x2)
        mx_out = getattr(mx.np, func)(mx_x1, mx_x2)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out.astype(mx_out.dtype), rtol=1e-3, atol=1e-20,
                            use_broadcast=False, equal_nan=True)


    funcs = [
        'add',
        'multiply',
        'true_divide',
    ]

    shape_pairs = [((3, 2), (3, 2)),
                   ((3, 2), (3, 1)),
                   ((3, 1), (3, 0)),
                   ((0, 2), (1, 2)),
                   ((2, 3, 4), (3, 1)),
                   ((2, 3), ()),
                   ((), (2, 3))]

    for lshape, rshape in shape_pairs:
        for func in funcs:
            x1 = np.array(onp.random.uniform(size=lshape) > 0.5)
            x2 = np.array(onp.random.uniform(size=rshape) > 0.5)
            check_boolean_binary_func(func, x1, x2)


@use_np
def test_npx_relu():
    def np_relu(x):
        return onp.maximum(x, 0.0)
    def np_relu_grad(x):
        return 1.0 * (x > 0.0)

    class TestReLU(HybridBlock):
        def __init__(self):
            super(TestReLU, self).__init__()

        def forward(self, a):
            return npx.relu(a)

    shapes = [(), (2, 3, 4), (2, 0, 3), (1, 0, 0)]
    for hybridize in [True, False]:
        for shape in shapes:
            test_relu = TestReLU()
            if hybridize:
                test_relu.hybridize()
            x = rand_ndarray(shape).as_np_ndarray()
            x.attach_grad()
            np_out = np_relu(x.asnumpy())
            with mx.autograd.record():
                mx_out = test_relu(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            mx_out.backward()
            np_backward = np_relu_grad(x.asnumpy())
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

            mx_out = npx.relu(x)
            np_out = np_relu(x.asnumpy())
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_npx_activation_log_sigmoid():
    def np_log_sigmoid(x):
        return onp.log(onp.divide(1.0, (1.0 + onp.exp(-x))))
    def np_log_sigmoid_grad(x):
        return onp.divide(1.0, onp.add(1.0, onp.exp(x)))

    class TestLogSigmoid(HybridBlock):
        def __init__(self):
            super(TestLogSigmoid, self).__init__()

        def forward(self, a):
            return npx.activation(a, act_type='log_sigmoid')

    shapes = [(), (2, 3, 4)]
    for hybridize in [True, False]:
        for shape in shapes:
            test_log_sigmoid = TestLogSigmoid()
            if hybridize:
                test_log_sigmoid.hybridize()
            x = rand_ndarray(shape).as_np_ndarray()
            x.attach_grad()
            np_out = np_log_sigmoid(x.asnumpy())
            with mx.autograd.record():
                mx_out = test_log_sigmoid(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            mx_out.backward()
            np_backward = np_log_sigmoid_grad(x.asnumpy())
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

            mx_out = npx.activation(x, act_type='log_sigmoid')
            np_out = np_log_sigmoid(x.asnumpy())
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_npx_activation_mish():
    def np_mish(a):
        return a * onp.tanh(onp.log1p(onp.exp(a)))
    def np_mish_grad(a):
        softrelu = onp.log1p(onp.exp(a))
        tanh = onp.tanh(softrelu)
        sigmoid = onp.divide(1.0, (1.0 + onp.exp(-a)))
        return tanh + a * sigmoid * (1.0 - tanh * tanh)

    class TestMish(HybridBlock):
        def __init__(self):
            super(TestMish, self).__init__()

        def forward(self, a):
            return npx.activation(a, act_type='mish')

    shapes = [(), (2, 3, 4)]
    for hybridize in [True, False]:
        for shape in shapes:
            test_mish = TestMish()
            if hybridize:
                test_mish.hybridize()
            x = rand_ndarray(shape).as_np_ndarray()
            x.attach_grad()
            np_out = np_mish(x.asnumpy())
            with mx.autograd.record():
                mx_out = test_mish(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            mx_out.backward()
            np_backward = np_mish_grad(x.asnumpy())
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

            mx_out = npx.activation(x, act_type='mish')
            np_out = np_mish(x.asnumpy())
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_npx_sigmoid():
    def np_sigmoid(x):
        return onp.divide(1.0, (1.0 + onp.exp(-x)))
    def np_sigmoid_grad(ya):
        return ya * (1 - ya)

    class TestSigmoid(HybridBlock):
        def __init__(self):
            super(TestSigmoid, self).__init__()

        def forward(self, a):
            return npx.sigmoid(a)

    shapes = [(), (2, 3, 4), (2, 0, 3), (1, 0, 0)]
    for hybridize in [True, False]:
        for shape in shapes:
            test_sigmoid = TestSigmoid()
            if hybridize:
                test_sigmoid.hybridize()
            x = rand_ndarray(shape).as_np_ndarray()
            x.attach_grad()
            np_out = np_sigmoid(x.asnumpy())
            with mx.autograd.record():
                mx_out = test_sigmoid(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            mx_out.backward()
            np_backward = np_sigmoid_grad(np_out)
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

            mx_out = npx.sigmoid(x)
            np_out = np_sigmoid(x.asnumpy())
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_atleast_nd():
    class TestAtleastND(HybridBlock):
        def __init__(self, n):
            super(TestAtleastND, self).__init__()
            self._n = n

        def forward(self, *arys):
            if self._n == 1:
                return np.atleast_1d(*arys)
            elif self._n == 2:
                return np.atleast_2d(*arys)
            elif self._n == 3:
                return np.atleast_3d(*arys)

    tensor_shapes = [
        ((), (2,), (3, 4, 5)),
        ((2, 3, 4, 5), (), (2, 3))
    ]
    flags = [True, False]
    ns = [1, 2, 3]
    dtypes = ['int32', 'int64', 'float16', 'float32', 'float64']
    funcs = {
        "numpy": {1: lambda *ts: onp.atleast_1d(*ts),
                  2: lambda *ts: onp.atleast_2d(*ts),
                  3: lambda *ts: onp.atleast_3d(*ts)},
        "mxnet": {1: lambda *ts: np.atleast_1d(*ts),
                  2: lambda *ts: np.atleast_2d(*ts),
                  3: lambda *ts: np.atleast_3d(*ts)}
    }
    for hybridize, n, tensor_shape, dtype in \
        itertools.product(flags, ns, tensor_shapes, dtypes):
        test_atleast_nd = TestAtleastND(n)
        if hybridize:
            test_atleast_nd.hybridize()
        if dtype in ['int32', 'int64']:
            tensors = list(map(lambda s: np.random.randint(-1, 1, size=s, dtype=dtype), tensor_shape))
        else:
            tensors = list(map(lambda s: np.random.uniform(-1.0, 1.0, size=s, dtype=dtype), tensor_shape))
        tensors_np = [t.asnumpy() for t in tensors]
        mx_out = test_atleast_nd(*tensors)
        np_out = funcs["numpy"][n](*tensors_np)
        for i in range(len(tensors)):
            assert mx_out[i].shape == np_out[i].shape
            assert same(mx_out[i].asnumpy(), np_out[i])

        mx_out = funcs["mxnet"][n](*tensors)
        np_out = funcs["numpy"][n](*tensors_np)
        for i in range(len(tensors)):
            assert mx_out[i].shape == np_out[i].shape
            assert same(mx_out[i].asnumpy(), np_out[i])


@use_np
def test_np_arange():
    configs = [
        (1, 10, 2),
        (1, 10, 4),
        (1, -10, 4),
        (1, -10, -2),
        (1, -10, -4),
        (2, 3),
        (2, -3),
        (-2, -3),
        (-2, 3),
        (4, 0, 5),
        (-4, 0, 5),
        (-4, 0, -5),
        (0, 0),
        (11, 11),
        (0, 0, 2),
        (0, 0, -2),
        (0, 5, None),
        (0, -5, None),
        0,
        6,
    ]
    dtypes = ['int32', 'float16', 'float32', 'float64', None]
    for config in configs:
        for dtype in dtypes:
            if isinstance(config, tuple):
                mx_ret = np.arange(*config, dtype=dtype)
                np_ret = onp.arange(*config, dtype=dtype)
            else:
                mx_ret = np.arange(config, dtype=dtype)
                np_ret = onp.arange(config, dtype=dtype)
            assert same(mx_ret.asnumpy(), np_ret)

    class TestRange(HybridBlock):
        def __init__(self, start, stop=None, step=None, dtype=None):
            super(TestRange, self).__init__()
            self._start = start
            self._stop = stop
            self._step = step
            self._dtype = dtype

        def forward(self, x):
            return x + np.arange(self._start, self._stop, self._step, dtype=self._dtype)

    for dtype in dtypes:
        x = np.zeros(shape=(), dtype=dtype)
        for config in configs:
            for hybridize in [False, True]:
                if isinstance(config, tuple):
                    net = TestRange(*config, dtype=dtype)
                    np_out = onp.arange(*config, dtype=dtype)
                else:
                    net = TestRange(config, dtype=dtype)
                    np_out = onp.arange(config, dtype=dtype)
                if hybridize:
                    net.hybridize()
                mx_out = net(x)
                assert same(mx_out.asnumpy(), np_out)


@use_np
def test_np_insert():
    class TestInsert(HybridBlock):
        def __init__(self, obj, axis=None):
            super(TestInsert, self).__init__()
            self._obj = obj
            self._axis = axis

        def forward(self, a, b):
            return np.insert(a, self._obj, b, axis=self._axis)

    def GetSize(tp):
        res = 1
        for x in tp:
            res = res * x
        return res

    def GetNdim(tp):
        return len(tp)

    A = (3, 2)
    B = (2)
    C = (2, 2)
    D = (2, 3)
    E = (1)
    F = (3, 1)
    G = (3, 2)
    H = (2, 2, 3, 8)
    config = []
    # test scale index
    for idx in range(-1 * GetSize(A), GetSize(A) + 1):
        config.append(tuple([A, idx, B, None]))
        config.append(tuple([A, idx, E, None]))
        config.append(tuple([A, idx, 1, None]))
    for idx in range(-1 * A[0], A[0] + 1):
        config.append(tuple([A, idx, C, 0]))
        config.append(tuple([A, idx, E, 0]))
        config.append(tuple([A, idx, F, 0]))
        config.append(tuple([A, idx, 1, 0]))
    for idx in range(-1 * A[1], A[1] + 1):
        config.append(tuple([A, idx, D, 1]))
        config.append(tuple([A, idx, E, 1]))
        config.append(tuple([A, idx, F, 1]))
        config.append(tuple([A, idx, 1, 1]))
    # test tuple of indices with size = 1
    for idx in range(-1 * GetSize(A), GetSize(A) + 1):
        config.append(tuple([A, [idx], B, None]))
        config.append(tuple([A, [idx], E, None]))
        config.append(tuple([A, [idx], 1, None]))
    for idx in range(-1 * A[0], A[0] + 1):
        config.append(tuple([A, [idx], C, 0]))
        config.append(tuple([A, [idx], E, 0]))
        config.append(tuple([A, [idx], F, 0]))
        config.append(tuple([A, [idx], 1, 0]))
    for idx in range(-1 * A[1], A[1] + 1):
        config.append(tuple([A, [idx], G, 1]))
        config.append(tuple([A, [idx], E, 1]))
        config.append(tuple([A, [idx], F, 1]))
        config.append(tuple([A, [idx], 1, 1]))
    # test tuple of indices with size > 1
    for ax in range(-1 * GetNdim(A), GetNdim(A)):
        idx = onp.random.randint(-1 * A[ax], A[ax] + 1, size = (3)).tolist()
        config.append(tuple([A, idx, F, ax]))
        config.append(tuple([A, idx, 1, ax]))
        config.append(tuple([A, slice(0, 3), F, ax]))
        config.append(tuple([A, slice(0, 3), 1, ax]))
    # test multidimensional array and unequal dimensions case
    config.append(tuple([H, 0, D, 3]))
    config.append(tuple([H, 0, 1, 3]))
    config.append(tuple([H, [1], E, 2]))
    config.append(tuple([H, [1], 1, 2]))
    idx = onp.random.randint(-1 * H[3], H[3] + 1, size = (5)).tolist()
    config.append(tuple([H, idx, E, 3]))
    config.append(tuple([H, idx, 1, 3]))
    # test slice
    for st in [-5, -3, -1, 0, 1, 3, 5, None]:
        for ed in [-5, -3, -1, 0, 1, 3, 5, None]:
            for stp in [-1, 1, 2, None]:
                config.append(tuple([A, slice(st, ed, stp), F, 1]))
    dtypes = ['int32', 'float16', 'float32', 'float64', None]

    for arr_shape, obj, val_shape, axis in config:
        for atype, btype in itertools.product(dtypes, dtypes):
            if type(obj) == list:
                obj_mxnp = np.array(obj, dtype='int64')
                obj_onp = onp.array(obj)
            elif type(obj) == slice:
                obj_mxnp = obj
                obj_onp = obj
            else:  # integer
                obj_mxnp = obj
                obj_onp = obj
            test_insert = TestInsert(obj=obj_mxnp, axis=axis)

            a = mx.nd.random.uniform(-10.0, 10.0, shape=arr_shape).as_np_ndarray().astype(atype)
            a.attach_grad()
            b = mx.nd.random.uniform(-10.0, 10.0, shape=val_shape).as_np_ndarray().astype(btype)
            b.attach_grad()
            expected_ret = onp.insert(a.asnumpy(), obj_onp, b.asnumpy(), axis=axis)
            with mx.autograd.record():
                y = test_insert(a, b)

            assert y.shape == expected_ret.shape
            assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)

            #test imperative
            mx_out = np.insert(a, obj_mxnp, b, axis=axis)
            np_out = onp.insert(a.asnumpy(), obj_onp, b.asnumpy(), axis=axis)

            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_split():
    class TestSplit(HybridBlock):
        def __init__(self, indices_or_sections, axis=None):
            super(TestSplit, self).__init__()
            self._axis = axis
            self._indices_or_sections = indices_or_sections

        def forward(self, a, *args, **kwargs):
            return np.split(a, indices_or_sections=self._indices_or_sections,
                              axis=self._axis)

    def get_indices(axis_size):
        if axis_size is 0:
            axis_size = random.randint(3, 6)
        samples = random.randint(1, axis_size - 1)
        indices = sorted(random.sample([i for i in range(1, axis_size)], samples))
        indices = tuple(indices)
        return indices

    dim = random.randint(0, 3)
    shape = [0] + [random.randint(2, 4) for i in range(dim)]
    for hybridize in [True, False]:
        for axis in range(-len(shape)+1, len(shape)):
            indices = get_indices(shape[axis])
            sections = 7 if shape[axis] is 0 else shape[axis]
            for indices_or_sections in [indices, sections]:
                # test gluon
                test_split = TestSplit(axis=axis, indices_or_sections=indices_or_sections)
                if hybridize:
                    test_split.hybridize()

                a = mx.nd.random.uniform(-1.0, 1.0, shape=shape).as_np_ndarray()
                a.attach_grad()
                expected_ret = onp.split(a.asnumpy(), indices_or_sections=indices_or_sections, axis=axis)
                with mx.autograd.record():
                    y = test_split(a)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                mx.autograd.backward(y)

                assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_outs = np.split(a, indices_or_sections=indices_or_sections, axis=axis)
                np_outs = onp.split(a.asnumpy(), indices_or_sections=indices_or_sections, axis=axis)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_array_split():
    class TestArray_split(HybridBlock):
        def __init__(self, indices_or_sections, axis=None):
            super(TestArray_split, self).__init__()
            self._axis = axis
            self._indices_or_sections = indices_or_sections

        def forward(self, a, *args, **kwargs):
            return np.array_split(a, indices_or_sections=self._indices_or_sections,
                              axis=self._axis)

    def get_indices(axis_size):
        if axis_size is 0:
            axis_size = random.randint(3, 6)
        samples = random.randint(1, axis_size - 1)
        indices = sorted(random.sample([i for i in range(0, axis_size + 1)], samples))
        indices = tuple(indices)
        return indices

    shapes = [(), (5, ), (10, ),
              (2, 5), (5, 5), (10, 10),
              (4, 4, 4), (4, 6, 9), (6, 6, 6),
              (7, 8, 9, 10)]
    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64]

    combinations = itertools.product([False, True], shapes, dtypes)
    for hybridize, shape, dtype in combinations:
        rtol = 1e-2 if dtype == np.float16 else 1e-3
        atol = 1e-4 if dtype == np.float16 else 1e-5
        for axis in range(len(shape)):
            x = np.random.uniform(-5.0, 5.0, size=shape).astype(dtype)
            indices = get_indices(shape[axis])
            sections = 7 if x.shape[axis] is 0 else random.randint(1,x.shape[axis])
            for indices_or_sections in [indices, sections]:
                # test gluon
                test_array_split = TestArray_split(axis=axis, indices_or_sections=indices_or_sections)
                if hybridize:
                    test_array_split.hybridize()
                x.attach_grad()
                expected_ret = onp.array_split(x.asnumpy(), indices_or_sections=indices_or_sections, axis=axis)
                with mx.autograd.record():
                    y = test_array_split(x)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                mx.autograd.backward(y)
                assert_almost_equal(x.grad.asnumpy(), onp.ones(x.shape), rtol=rtol, atol=atol)

                # test imperative
                mx_outs = np.array_split(x, indices_or_sections=indices_or_sections, axis=axis)
                np_outs = onp.array_split(x.asnumpy(), indices_or_sections=indices_or_sections, axis=axis)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_vsplit():
    class TestVsplit(HybridBlock):
        def __init__(self, indices_or_sections):
            super(TestVsplit, self).__init__()
            self._indices_or_sections = indices_or_sections

        def forward(self, a, *args, **kwargs):
            return np.vsplit(a, indices_or_sections=self._indices_or_sections)

    def get_indices(axis_size):
        if axis_size is 0:
            axis_size = random.randint(3, 6)
        samples = random.randint(1, axis_size - 1)
        indices = sorted(random.sample([i for i in range(1, axis_size)], samples))
        indices = tuple(indices)
        return indices

    shapes = [
        (2, 1, 2, 9),
        (4, 3, 3),
        (4, 0, 2),  # zero-size shape
        (0, 3), # first dim being zero
    ]
    for hybridize in [True, False]:
        for shape in shapes:
            axis_size = shape[0]
            indices = get_indices(axis_size)
            sections = 7 if axis_size is 0 else axis_size
            for indices_or_sections in [indices, sections]:
                # test gluon
                test_vsplit = TestVsplit(indices_or_sections=indices_or_sections)
                if hybridize:
                    test_vsplit.hybridize()
                a = rand_ndarray(shape).as_np_ndarray() # TODO: check type
                a.attach_grad()
                expected_ret = onp.vsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                with mx.autograd.record():
                    y = test_vsplit(a)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                mx.autograd.backward(y)

                assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_outs = np.vsplit(a, indices_or_sections=indices_or_sections)
                np_outs = onp.vsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_concat():
    class TestConcat(HybridBlock):
        def __init__(self, axis=None):
            super(TestConcat, self).__init__()
            self._axis = axis

        def forward(self, a, *args):
            return np.concatenate([a] + list(args), axis=self._axis)

    def get_new_shape(shape, axis):
        shape_lst = list(shape)
        if axis is not None:
            shape_lst[axis] = random.randint(0, 3)
        return tuple(shape_lst)

    shapes = [(), (0, 0), (2, 3), (2, 1, 3)]
    hybridizes = [True, False]
    axes = [0, 1, -1, None]
    grad_reqs = ['write', 'add', 'null']
    dtypes = [np.float32, np.float64, np.bool]
    combinations = itertools.product(shapes, hybridizes, axes, grad_reqs, dtypes)

    for shape, hybridize, axis, grad_req, dtype in combinations:
        # test gluon
        if shape == () and axis != None:
            continue
        test_concat = TestConcat(axis=axis)
        if hybridize:
            test_concat.hybridize()

        grad_req_c = grad_req
        grad_req_d = grad_req
        if grad_req == 'null':
            ide = random.randint(0, 2)
            grad_req_c = 'write' if ide == 0 else 'add'
            grad_req_c = 'write' if ide == 1 else 'add'

        a = np.random.uniform(-1.0, 1.0, size=get_new_shape(shape, axis)).astype(dtype)
        a.attach_grad(grad_req)
        b = np.random.uniform(-1.0, 1.0, size=get_new_shape(shape, axis)).astype(dtype)
        b.attach_grad(grad_req)
        c = np.random.uniform(-1.0, 1.0, size=get_new_shape(shape, axis)).astype(dtype)
        c.attach_grad(grad_req_c)
        d = np.random.uniform(-1.0, 1.0, size=get_new_shape(shape, axis)).astype(dtype)
        d.attach_grad(grad_req_d)
        expected_ret = onp.concatenate([a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()], axis=axis)

        with mx.autograd.record():
            y = test_concat(a, b, c, d)

        assert y.shape == expected_ret.shape
        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)

        y.backward()
        if grad_req != 'null':
            assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)
        if grad_req != 'null':
            assert_almost_equal(b.grad.asnumpy(), onp.ones(b.shape), rtol=1e-3, atol=1e-5)
        if grad_req_c != 'null':
            assert_almost_equal(c.grad.asnumpy(), onp.ones(c.shape), rtol=1e-3, atol=1e-5)
        if grad_req_d != 'null':
            assert_almost_equal(d.grad.asnumpy(), onp.ones(d.shape), rtol=1e-3, atol=1e-5)

        # test imperative
        mx_out = np.concatenate([a, b, c, d], axis=axis)
        np_out = onp.concatenate([a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()], axis=axis)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_append():
    class TestAppend(HybridBlock):
        def __init__(self, axis=None):
            super(TestAppend, self).__init__()
            self._axis = axis

        def forward(self, a, b):
            return np.append(a, b, axis=self._axis)

    def get_new_shape(shape, axis):
        shape_lst = list(shape)
        if axis is not None:
            shape_lst[axis] = random.randint(0, 3)
        return tuple(shape_lst)

    for shape in [(0, 0), (2, 3), (2, 1, 3)]:
        for hybridize in [True, False]:
            for axis in [0, 1, None]:
                for grad_req_a in ['write', 'add', 'null']:
                    if grad_req_a == 'null':
                        continue
                    #set grad_req
                    grad_req_b = grad_req_a
                    if grad_req_a == 'null':
                        ide = random.randint(0, 2)
                        grad_req_b = 'write' if ide == 0 else 'add'

                    #test gluon
                    test_append = TestAppend(axis=axis)
                    if hybridize:
                        test_append.hybridize()

                    a = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                    a.attach_grad(grad_req=grad_req_a)
                    b = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                    b.attach_grad(grad_req=grad_req_b)
                    expected_ret = onp.append(a.asnumpy(), b.asnumpy(), axis=axis)

                    with mx.autograd.record():
                        y = test_append(a, b)

                    assert y.shape == expected_ret.shape
                    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)
                    y.backward()

                    if grad_req_a != 'null':
                        assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)
                    assert_almost_equal(b.grad.asnumpy(), onp.ones(b.shape), rtol=1e-3, atol=1e-5)
                    #test imperative
                    mx_out = np.append(a, b, axis=axis)
                    np_out = onp.append(a.asnumpy(), b.asnumpy(), axis=axis)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_stack():
    class TestStack(HybridBlock):
        def __init__(self, axis=None):
            super(TestStack, self).__init__()
            self._axis = axis

        def forward(self, a, *args):
            return np.stack([a] + list(args), axis=self._axis)

    a, b, c, d = mx.sym.Variable("a"), mx.sym.Variable("b"), mx.sym.Variable("c"), mx.sym.Variable("d")
    ret = mx.sym.np.stack([a.as_np_ndarray(), b.as_np_ndarray(), c.as_np_ndarray(), d.as_np_ndarray()])
    assert type(ret) == mx.sym.np._Symbol

    for shape in [(0, 0), (2, 3)]:
        for hybridize in [True, False]:
            for axis in range(2):
                test_stack = TestStack(axis=axis)
                if hybridize:
                    test_stack.hybridize()
                np_a = onp.random.uniform(-1.0, 1.0, shape).astype(onp.float32)
                np_b = onp.random.uniform(-1.0, 1.0, shape).astype(onp.float32)
                np_c = onp.random.uniform(-1.0, 1.0, shape).astype(onp.float32)
                np_d = onp.random.uniform(-1.0, 1.0, shape).astype(onp.float32)

                mx_a = np.array(np_a)
                mx_a.attach_grad()
                mx_b = np.array(np_b)
                mx_b.attach_grad()
                mx_c = np.array(np_c)
                mx_c.attach_grad()
                mx_d = np.array(np_d)
                mx_d.attach_grad()
                expected_ret = onp.stack([np_a, np_b, np_c, np_d], axis=axis)
                with mx.autograd.record():
                    y = test_stack(mx_a, mx_b, mx_c, mx_d)

                y.backward()

                assert_almost_equal(mx_a.grad.asnumpy(), onp.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_b.grad.asnumpy(), onp.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_c.grad.asnumpy(), onp.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_d.grad.asnumpy(), onp.ones(shape), rtol=1e-3, atol=1e-5)

                np_out = onp.stack([np_a, np_b, np_c, np_d], axis=axis)
                mx_out = np.stack([mx_a, mx_b, mx_c, mx_d], axis=axis)
                assert same(mx_out.asnumpy(), np_out)


@use_np
def test_np_hstack():
    class TestHStack(HybridBlock):
        def __init__(self):
            super(TestHStack, self).__init__()

        def forward(self, a, *args):
            return np.hstack([a] + list(args))

    def get_new_shape(shape):
        if len(shape) == 0:
            l = random.randint(0,3)
            if l == 0:
                return shape
            else:
                return (l,)
        shape_lst = list(shape)
        axis = 1 if len(shape) > 1 else 0
        shape_lst[axis] = random.randint(0, 5)
        return tuple(shape_lst)

    shapes = [
        (),
        (1,),
        (2,1),
        (2,2,4),
        (2,0,0),
        (0,1,3),
        (2,0,3),
        (2,3,4,5)
    ]
    for hybridize in [True, False]:
        for shape in shapes:
            test_hstack = TestHStack()
            if hybridize:
                test_hstack.hybridize()
            # test symbolic forward
            a = np.random.uniform(size=get_new_shape(shape))
            a.attach_grad()
            b = np.random.uniform(size=get_new_shape(shape))
            b.attach_grad()
            c = np.random.uniform(size=get_new_shape(shape))
            c.attach_grad()
            d = np.random.uniform(size=get_new_shape(shape))
            d.attach_grad()
            with mx.autograd.record():
                mx_out = test_hstack(a, b, c, d)
            np_out = onp.hstack((a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

            # test symbolic backward
            mx_out.backward()
            assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(b.grad.asnumpy(), onp.ones(b.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(c.grad.asnumpy(), onp.ones(c.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(d.grad.asnumpy(), onp.ones(d.shape), rtol=1e-3, atol=1e-5)

            mx_out = np.hstack((a, b, c, d))
            np_out = onp.hstack((a.asnumpy(),b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_dstack():
    class TestDStack(HybridBlock):
        def __init__(self):
            super(TestDStack, self).__init__()

        def forward(self, a, *args):
            return np.dstack([a] + list(args))

    def get_new_shape(shape):
        if len(shape) < 3:
            return shape
        axis = 2
        shape_lst = list(shape)
        shape_lst[axis] = random.randint(0, 5)
        return tuple(shape_lst)

    shapes = [
        (),
        (1,),
        (2,1),
        (2,2,4),
        (2,0,0),
        (0,1,3),
        (2,0,3),
        (2,3,4,5)
    ]
    for hybridize in [True, False]:
        for shape in shapes:
            test_dstack = TestDStack()
            if hybridize:
                test_dstack.hybridize()
            # test symbolic forward
            a = mx.nd.random.uniform(shape=get_new_shape(shape)).as_np_ndarray()
            a.attach_grad()
            b = mx.nd.random.uniform(shape=get_new_shape(shape)).as_np_ndarray()
            b.attach_grad()
            c = mx.nd.random.uniform(shape=get_new_shape(shape)).as_np_ndarray()
            c.attach_grad()
            d = mx.nd.random.uniform(shape=get_new_shape(shape)).as_np_ndarray()
            d.attach_grad()
            with mx.autograd.record():
                mx_out = test_dstack(a, b, c, d)
            np_out = onp.dstack((a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

            # test symbolic backward
            mx_out.backward()
            assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(b.grad.asnumpy(), onp.ones(b.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(c.grad.asnumpy(), onp.ones(c.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(d.grad.asnumpy(), onp.ones(d.shape), rtol=1e-3, atol=1e-5)

            # test imperative
            mx_out = np.dstack((a, b, c, d))
            np_out = onp.dstack((a.asnumpy(),b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_ravel():
    class TestRavel(HybridBlock):
        def __init__(self):
            super(TestRavel, self).__init__()

        def forward(self, a):
            return np.ravel(a)

    types = ['float64', 'float32', 'float16', 'int64', 'int32', 'int8']
    for oneType in types:
        for hybridize in [True, False]:
            for shape in [(), (2,), (2, 2), (1, 2, 3), (3, 0), (1, 0, 2)]:
                test_ravel = TestRavel()
                if hybridize:
                    test_ravel.hybridize()
                x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                x.attach_grad()
                np_out = onp.ravel(x.asnumpy())
                with mx.autograd.record():
                    mx_out = test_ravel(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                mx_out.backward()
                np_backward = onp.ones(shape)
                assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

                mx_out = np.ravel(x)
                np_out = onp.ravel(x.asnumpy())
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_randint():
    device = mx.device.current_device()
    # test shapes
    params = [
        (0, 10),
        (5, None)
    ]
    shapes = [
        None,
        (),
        (3, 3),
        (3, 4),
        (0, 0),
        (3, 3, 3),
        (0, 0, 0),
        (2, 2, 4, 3),
        (2, 2, 4, 3),
        (2, 0, 3, 0),
        (2, 0, 2, 3)
    ]
    for shape in shapes:
        for (low, high) in params:
            data_mx = np.random.randint(low, high, size=shape)
            assert data_mx.shape == (shape if shape is not None else ())

    # test generator
    for dtype in ['int32', 'int64']:
        for low, high in [(50000000, 50001000),(-50000100,-50000000),(-500,199)]:
            scale = high - low
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.uniform.ppf(x, loc=low, scale=scale), 5)
            # Quantize bucket boundaries to reflect the actual dtype and adjust probs accordingly
            buckets = onp.array(buckets, dtype=dtype).tolist()
            probs = [(buckets[i][1] - buckets[i][0]) / float(scale) for i in range(5)]
            generator_mx = lambda x: np.random.randint(low, high, size=x, dtype=dtype, device=device).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs, nrepeat=100)
            # Scipy uses alpha = 0.01 for testing discrete distribution generator but we are using default alpha=0.05 (higher threshold ensures robustness)
            # Refer - https://github.com/scipy/scipy/blob/9f12af697763fb5f9767d5cb1280ce62456a3974/scipy/stats/tests/test_discrete_basic.py#L45
            generator_mx_same_seed = \
                lambda x: onp.concatenate(
                    [np.random.randint(low, high, size=x // 10, dtype=dtype, device=device).asnumpy()
                        for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs, nrepeat=100)


@use_np
def test_np_swapaxes():
    config = [((0, 1, 2), 0, 0),
              ((0, 1, 2), 1, 2),
              ((0, 1, 2), 1, -2),
              ((4, 5, 6, 7), 1, 1),
              ((4, 5, 6, 7), 2, -2),
              ((4, 5, 6, 7), -2, -3)]

    class TestSwapaxes(HybridBlock):
        def __init__(self, axis1, axis2):
            super(TestSwapaxes, self).__init__()
            self._axis1 = axis1
            self._axis2 = axis2

        def forward(self, x):
            return np.swapaxes(x, self._axis1, self._axis2)

    for shape, axis1, axis2 in config:
        data_np = onp.random.uniform(size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = onp.swapaxes(data_np, axis1=axis1, axis2=axis2)
        ret_mx = np.swapaxes(data_mx, axis1=axis1, axis2=axis2)
        assert same(ret_mx.asnumpy(), ret_np)

        net = TestSwapaxes(axis1, axis2)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            ret_mx = net(data_mx)
            assert same(ret_mx.asnumpy(), ret_np)


@use_np
def test_np_delete():
    class TestDelete(HybridBlock):
        def __init__(self, obj, axis=None):
            super(TestDelete, self).__init__()
            self._obj = obj
            self._axis = axis

        def forward(self, a):
            return np.delete(a, self._obj, axis=self._axis)

    def GetSize(shp):
        if len(shp) == 0:
            return 0
        else:
            res = 1
            shp_list = list(shp)
            for x in shp:
                res *= x
            return res

    def GetDimSize(shp, axis):
        if axis is None:
            return GetSize(shp)
        shp_list = list(shp)
        return shp_list[axis]

    shape = [(), (0, ), (1, ), (2, 3), (2, 1, 4, 5)]
    config = []
    for shp in shape:
        for ax in range(-1 * len(shp), len(shp), 2):
            #test slice
            for st in [-5, -2, 0, 2, 5, None]:
                for ed in [-5, -2, 0, 2, 5, None]:
                    for stp in [-5, -2, 2, 5, None]:
                        config.append(tuple([shp, slice(st, ed, stp), None]))
                        config.append(tuple([shp, slice(st, ed, stp), ax]))
            #test iteger
            for idx in range(-1 * GetDimSize(shp, ax), GetDimSize(shp, ax)):
                config.append(tuple([shp, idx, ax]))
            #test ndarray indices
            idx =  onp.random.randint(-1 * shp[ax], shp[ax] + 1, size = (4)).tolist()
            config.append(tuple([shp, idx, ax]))

    for arr_shape, obj, axis in config:
        for objtype in ['int32', 'int64']:
            if type(obj) == list:
                obj_mxnp = np.array(obj, dtype=objtype)
                obj_onp = onp.array(obj, dtype=objtype)
                # To match mxnet.numpy's behavior of ignoring out-of-bounds indices,
                # we may need to filter out indices that this numpy would not ignore.
                onp_ignores_oob_indices = parse(onp.version.version) < parse('1.19')
                if not onp_ignores_oob_indices:
                    dim_size = GetDimSize(arr_shape,axis)
                    obj_onp = obj_onp[((obj_onp>=0) & (obj_onp<dim_size))]
            elif type(obj) == slice:
                obj_mxnp = obj
                obj_onp = obj
            else:
                obj_mxnp = (onp.int32(obj) if objtype == 'int32' else onp.int64(obj))
                obj_onp = (onp.int32(obj) if objtype == 'int32' else onp.int64(obj))
            test_delete = TestDelete(obj=obj_mxnp, axis=axis)

            a = mx.nd.random.uniform(-1.0, 1.0, shape=arr_shape).as_np_ndarray()
            a.attach_grad()
            expected_ret = onp.delete(a.asnumpy(), obj_onp, axis=axis)

            with mx.autograd.record():
                y = test_delete(a)

            assert y.shape == expected_ret.shape
            assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)

            #test imperative
            mx_out = np.delete(a, obj_mxnp, axis=axis)
            np_out = onp.delete(a.asnumpy(), obj_onp, axis=axis)

            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
@pytest.mark.parametrize('shape,axis,throw_exception', [
    ((), 0, False),
    ((), -1, False),
    ((), 1, True),
    ((5, 3), None, False),
    ((5, 3), -1, False),
    ((5, 3), 1, False),
    ((5, 3), 3, True),
    ((5, 0, 3), 0, False),
    ((5, 0, 3), -1, False),
    ((5, 0, 3), None, True),
    ((5, 0, 3), 1, True),
    ((3, 5, 7), None, False),
    ((3, 5, 7), 0, False),
    ((3, 5, 7), 1, False),
    ((3, 5, 7), 2, False),
    ((3, 5, 7, 9, 11), -3, False),
])
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'bool', 'int32'])
@pytest.mark.parametrize('op_name', ['argmin', 'argmax'])
@pytest.mark.parametrize('keepdims', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_argmin_argmax(shape, axis, throw_exception, dtype, op_name, keepdims, hybridize):
    class TestArgExtreme(HybridBlock):
        def __init__(self, op_name, axis=None, keepdims=False):
            super(TestArgExtreme, self).__init__()
            self._op_name = op_name
            self._axis = axis
            self.keepdims = keepdims

        def forward(self, x):
            return getattr(x, self._op_name)(self._axis, keepdims=self.keepdims)

    a = np.random.uniform(low=0, high=100, size=shape).astype(dtype)
    if throw_exception:
        with pytest.raises(MXNetError):
            getattr(np, op_name)(a, axis)
            mx.npx.waitall()
    else:
        mx_ret = getattr(np, op_name)(a, axis=axis, keepdims=keepdims)
        np_ret = getattr(onp, op_name)(a.asnumpy(), axis=axis)
        assert mx_ret.dtype == np_ret.dtype
        if keepdims:
            assert same(np.squeeze(mx_ret, axis=axis).asnumpy(), np_ret)
        else:
            assert same(mx_ret.asnumpy(), np_ret)

    net = TestArgExtreme(op_name, axis, keepdims)
    if hybridize:
        net.hybridize()
    if throw_exception:
        with pytest.raises(MXNetError):
            getattr(np, op_name)(a, axis)
            mx.npx.waitall()
    else:
        mx_ret = net(a)
        assert mx_ret.dtype == np_ret.dtype
        if keepdims:
            assert same(np.squeeze(mx_ret, axis=axis).asnumpy(), np_ret)
        else:
            assert same(mx_ret.asnumpy(), np_ret)


@use_np
def test_np_argmin_argmax_large_tensor():
    # compare inp[arg] with ext directly because along one axis there might 
    # be multiple extrema
    def single_run(op, dtype):
        inp = np.random.normal(0, 10, size=(200, 30000), dtype=dtype)
        arg = op[0](inp, 1)
        ref = op[1](inp, 1)
        for i, idx in enumerate(arg):
            assert inp[i, idx] == ref[i]

    dtypes = ['float16', 'float32', 'float64']
    ops = [(np.argmin, np.amin), (np.argmax, np.amax)]
    for o, d in zip(ops, dtypes):
        single_run(o, d)


@use_np
def test_np_clip():
    workloads = [
        ((), None, None, True),
        ((), None, 1, False),
        ((), -1, 1, False),
        ((), -1, None, False),
        ((5, 3), None, 0.1, False),
        ((5, 3), -0.1, None, False),
        ((5, 3), -0.1, 0.1, False),
        ((5, 3), 0, 0, False),
        ((5, 0, 3), 0, None, False),
        ((5, 0, 3), None, -1, False),
        ((5, 0, 3), -1, 0, False),
    ]
    dtypes = ['float32', 'float64']

    class TestClip(HybridBlock):
        def __init__(self, a_min=None, a_max=None):
            super(TestClip, self).__init__()
            self._a_min = a_min
            self._a_max = a_max

        def forward(self, x):
            return x.clip(self._a_min, self._a_max)

    # Test scalar case
    for _, a_min, a_max, throw_exception in workloads:
        a = onp.random.uniform() # A scalar
        if throw_exception:
            # No need to test the exception case here.
            continue
        mx_ret = np.clip(a, a_min, a_max)
        np_ret = onp.clip(a, a_min, a_max)
        assert_almost_equal(mx_ret, np_ret, atol=1e-4, rtol=1e-3, use_broadcast=False)

    for shape, a_min, a_max, throw_exception in workloads:
        for dtype in dtypes:
            a = np.random.uniform(size=shape, dtype=dtype)
            if throw_exception:
                # Cannot use assert_exception because sometimes the main thread
                # proceeds to `assert False` before the exception is thrown
                # in the worker thread. Have to use mx.nd.waitall() here
                # to block the main thread.
                try:
                    a.clip(min=a_min, max=a_max)
                    mx.nd.waitall()
                    assert False
                except:
                    pass
            else:
                mx_ret = a.clip(min=a_min, max=a_max)
                np_ret = a.asnumpy().clip(min=a_min, max=a_max)
                assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-4, rtol=1e-3, use_broadcast=False)

            for hybridize in [False, True]:
                net = TestClip(a_min, a_max)
                if hybridize:
                    net.hybridize()
                if throw_exception:
                    try:
                        net(a)
                        mx.nd.waitall()
                        assert False
                    except:
                        pass
                else:
                    mx_ret = net(a)
                    assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-4, rtol=1e-3, use_broadcast=False)


@use_np
def test_npx_random_bernoulli():
    def _test_bernoulli_exception(prob, logit):
        output = npx.random.bernoulli(prob=prob, logit=logit).asnumpy()

    shapes = [(), (1,), (2, 3), (4, 0, 5), 6, (7, 8), None]
    dtypes = ['float16', 'float32', 'float64', 'int32', 'bool']
    for shape, dtype in itertools.product(shapes, dtypes):
        prob = np.random.uniform(size=shape)
        logit = np.log(prob) - np.log(1 - prob)
        expected_shape = shape
        if not isinstance(shape, tuple):
            expected_shape = () if shape is None else (shape,)
        out_prob = npx.random.bernoulli(prob=prob, size=shape, dtype=dtype)
        assert out_prob.shape == expected_shape
        assert int((out_prob.asnumpy() == 0).sum() + (out_prob.asnumpy() == 1).sum()) == out_prob.size
        out_logit = npx.random.bernoulli(logit=logit, size=shape, dtype=dtype)
        assert out_logit.shape == expected_shape
        assert int((out_logit.asnumpy() == 0).sum() + (out_logit.asnumpy() == 1).sum()) == out_logit.size
        # Test Exception.
        assertRaises(ValueError, _test_bernoulli_exception, prob, logit)
        if prob.size > 0:
            # larger than 1
            assertRaises(ValueError, _test_bernoulli_exception, prob + 2.0, None)
            # smaller than 0
            assertRaises(ValueError, _test_bernoulli_exception, prob - 2.0, None)
            # mixed case
            low, high = (-1.0, 2.0)
            # uniform(-1, 2)
            scaled_prob = low + (high - low) * prob
            if not ((scaled_prob.asnumpy() >= 0).all() and (scaled_prob.asnumpy() <= 1).all()):
                assertRaises(ValueError, _test_bernoulli_exception, scaled_prob, None)


@use_np
def test_npx_constraint_check():
    msg = "condition violated"
    class TestConstraintViolatedCheck(HybridBlock):
        def __init__(self):
            super(TestConstraintViolatedCheck, self).__init__()

        def forward(self, boolean_tensor):
            return npx.constraint_check(boolean_tensor, msg)

    class TestConstraintNotViolatedCheck(HybridBlock):
        def __init__(self):
            super(TestConstraintNotViolatedCheck, self).__init__()

        def forward(self, input, boolean_tensor):
            return input * npx.constraint_check(boolean_tensor, msg)

    def raiseFunc(block):
        def executor(boolean_tensor):
            out = block(boolean_tensor).asnumpy()
        return executor

    shapes = [(1,), (2, 3), 6, (7, 8)]

    expect_success_output = np.array(True)
    for shape, hybridize in itertools.product(shapes, [True, False]):
        test_constraint = TestConstraintViolatedCheck()
        if hybridize:
            test_constraint.hybridize()
        assertRaises(ValueError, raiseFunc(test_constraint), np.zeros(shape, dtype='bool'))

    for shape, hybridize in itertools.product(shapes, [True, False]):
        test_constraint = TestConstraintNotViolatedCheck()
        if hybridize:
            test_constraint.hybridize()
        input_tensor = np.random.normal(size=shape)
        out = test_constraint(input_tensor, np.ones(shape, dtype='bool'))
        assert (input_tensor.asnumpy() == out.asnumpy()).all()


@use_np
def test_npx_special_unary_func():
    def check_unary_func(func, ref_grad, shape, low, high):
        class TestUnary(HybridBlock):
            def __init__(self, func):
                super(TestUnary, self).__init__()
                self._func = func

            def forward(self, a, *args, **kwargs):
                return getattr(npx, self._func)(a)

        np_func = getattr(scipy_special, func)
        mx_func = TestUnary(func)
        np_test_data = onp.random.uniform(low, high, shape).astype(onp.float32)
        mx_test_data = mx.numpy.array(np_test_data)
        for hybridize in [True, False]:
            if hybridize:
                mx_func.hybridize()
            if ref_grad:
                mx_test_data.attach_grad()
            np_out = np_func(np_test_data)
            with mx.autograd.record():
                y = mx_func(mx_test_data)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
            if np_out.dtype == np.bool_:
                assert y.dtype == np.bool_

            if ref_grad:
                y.backward()
                assert_almost_equal(mx_test_data.grad.asnumpy(), ref_grad(np_test_data), rtol=1e-1, atol=1e-2, equal_nan=True)

        np_out = getattr(scipy_special, func)(np_test_data)
        mx_out = getattr(mx.npx, func)(mx_test_data)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    import math
    funcs = {
        'erf' : (lambda x: 2.0 / math.sqrt(math.pi) * onp.exp(-(x ** 2)), 0.5, 0.5),
        'erfinv' : (lambda x: 0.5 * math.sqrt(math.pi) * onp.exp(scipy_special.erfinv(x) ** 2), 0.5, 0.5),
        'gamma' : (lambda x: scipy_special.gamma(x) * scipy_special.psi(x), 0.5, 0.5),
        'gammaln' : (lambda x: scipy_special.psi(x), 0.5, 0.5),
        'digamma' : (lambda x: scipy_special.polygamma(1, x), 0.5, 0.5)
    }
    ndim = random.choice([2, 3, 4])
    shape = random.choice([rand_shape_nd(ndim, dim=3), (1, 0, 2)])
    for shape in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]:
        for func, func_data in funcs.items():
            ref_grad, low, high = func_data
            check_unary_func(func, ref_grad, shape, low, high)


@xfail_when_nonstandard_decimal_separator
@use_np
def test_np_random_grad():
    class TestRandomGrad(HybridBlock):
        def __init__(self, shape, op_name):
            super(TestRandomGrad, self).__init__()
            self._shape = shape
            self._dist_name = op_name
        def forward(self, loc, scale):
            op = getattr(np.random, self._dist_name, None)
            assert op is not None
            return op(loc=loc, scale=scale, size=self._shape)

    param_shape = [
        [(3, 2), (3, 2)],
        [(3, 2, 2), (3, 2, 2)],
        [(3, 4, 5), (4, 1)],
    ]
    output_shapes = [
        (3, 2),
        (4, 3, 2, 2),
        (3, 4, 5)
    ]
    op_names = ["normal", "logistic", "gumbel"]
    for op_name in op_names:
        for hybridize in [False, True]:
            for ((shape1, shape2), out_shape) in zip(param_shape, output_shapes):
                test_random_grad = TestRandomGrad(out_shape, op_name)
                if hybridize:
                    test_random_grad.hybridize()
                loc = np.zeros(shape1)
                loc.attach_grad()
                scale = np.ones(shape2)
                scale.attach_grad()
                with mx.autograd.record():
                    samples = test_random_grad(loc, scale)
                samples.backward()
                assert loc.grad.shape == shape1
                assert scale.grad.shape == shape2
                assert_almost_equal(loc.grad.asnumpy().sum(), onp.ones(out_shape).sum(), rtol=1e-3, atol=1e-5)

        for (loc, scale) in [(2, (2,3)), ((2,3), 2), ((2,3), (2,3))]:
            if isinstance(loc, tuple):
                loc = np.ones(loc)
            if isinstance(scale, tuple):
                scale = np.ones(scale)
            mx_out = getattr(np.random, op_name)(loc, scale)
            np_out = getattr(onp.random, op_name)(loc, scale)
            assert mx_out.asnumpy().shape == np_out.shape


@use_np
def test_np_lognormal_grad():
    class TestLognormalGrad(HybridBlock):
        def __init__(self, shape):
            super(TestLognormalGrad, self).__init__()
            self._shape = shape

        def forward(self, mean, sigma):
            return np.random.lognormal(mean, sigma, self._shape)

    param_shape = [
        [(3, 2), (3, 2)],
        [(3, 2, 2), (3, 2, 2)],
        [(3, 4, 5), (4, 1)],
    ]
    output_shapes = [
        (3, 2),
        (4, 3, 2, 2),
        (3, 4, 5)
    ]
    for hybridize in [False, True]:
        for ((shape1, shape2), out_shape) in zip(param_shape, output_shapes):
            test_lognormal_grad = TestLognormalGrad(out_shape)
            if hybridize:
                test_lognormal_grad.hybridize()
            mean = np.zeros(shape1)
            mean.attach_grad()
            sigma = np.ones(shape2)
            sigma.attach_grad()
            with mx.autograd.record():
                mx_out = test_lognormal_grad(mean, sigma)
            np_out = onp.random.lognormal(mean = mean.asnumpy(),
                                            sigma = sigma.asnumpy(), size = out_shape)
            assert np_out.shape == mx_out.shape
            mx_out.backward()
            assert mean.grad.shape == shape1
            assert sigma.grad.shape == shape2
            assert_almost_equal(mean.grad.asnumpy().sum(), mx_out.asnumpy().sum(), rtol=1e-3, atol=1e-5)

    for ((shape1, shape2), out_shape) in zip(param_shape, output_shapes):
        mx_out = np.random.lognormal(np.zeros(shape1), np.ones(shape2), out_shape)
        np_out = onp.random.lognormal(np.zeros(shape1).asnumpy(), np.ones(shape2).asnumpy(), out_shape)
        assert mx_out.asnumpy().shape == np_out.shape

    def _test_lognormal_exception(sigma):
        output = np.random.lognormal(sigma=sigma).asnumpy()
    assertRaises(ValueError, _test_lognormal_exception, -1)


@use_np
def test_npx_sample_n():
    def shape_formatter(s):
        if s is None:
            return ()
        if isinstance(s, tuple):
            return s
        # scalar case
        return (s,)

    class TestSampleN(HybridBlock):
        def __init__(self, shape, op_name, dtype):
            super(TestSampleN, self).__init__()
            self._shape = shape
            self._op_name = op_name
            self._dtype = dtype

        def forward(self, param1, param2):
            op = getattr(npx.random, self._op_name, None)
            assert op is not None
            return op(param1, param2, batch_shape=self._shape, dtype=self._dtype)

    batch_shapes = [(10,), (2, 3), 6, ()]
    event_shapes = [(), (2,), (2,2)]
    dtypes = ['float16', 'float32', 'float64']
    op_names = ['uniform_n', 'normal_n']

    for bshape, eshape, dtype, op in itertools.product(batch_shapes, event_shapes, dtypes, op_names):
        for hybridize in [True, False]:
            net = TestSampleN(bshape, op, dtype)
            if hybridize:
                net.hybridize()
            expected_shape = (shape_formatter(bshape) +
                              shape_formatter(eshape))
            out = net(np.ones(shape=eshape), np.ones(shape=eshape))
            assert out.shape == expected_shape


@use_np
def test_np_random():
    shapes = [(), (1,), (2, 3), (4, 0, 5), 6, (7, 8), None]
    dtypes = ['float16', 'float32', 'float64']
    op_names = ['uniform', 'normal', 'gamma', 'laplace']
    for shape in shapes:
        for dtype in dtypes:
            for op_name in op_names:
                op = getattr(np.random, op_name, None)
                assert op is not None
                if op_name == 'gamma':
                    out = op(1, size=shape, dtype=dtype)
                else:
                    out = op(size=shape, dtype=dtype)
                expected_shape = shape
                if not isinstance(shape, tuple):
                    expected_shape = () if shape is None else (shape,)
                assert out.shape == expected_shape

    class TestRandom(HybridBlock):
        def __init__(self, shape, op_name, param=None):
            super(TestRandom, self).__init__()
            self._shape = shape
            self._op_name = op_name
            # In case parameters are not optional
            self._param = param

        def forward(self, x):
            op = getattr(np.random, self._op_name, None)
            assert op is not None
            if self._param is not None:
                return x + op(self._param, size=self._shape)
            return x + op(size=self._shape)

    x = np.ones(())
    for op_name in op_names:
        for shape in shapes:
            for hybridize in [False, True]:
                if op_name == "gamma":
                    net = TestRandom(shape, op_name, 1)
                else:
                    net = TestRandom(shape, op_name)
                if hybridize:
                    net.hybridize()
                out = net(x)
                expected_shape = shape
                if not isinstance(shape, tuple):
                    expected_shape = () if shape is None else (shape,)
                assert out.shape == expected_shape


@use_np
def test_gamma_exception():
    def _test_gamma_exception(shape, scale):
        return np.random.gamma(shape, scale).asnumpy()

    shape_list = [
        1,
        np.array(1),
        np.array(1),
        0,
        0,
        np.array(0)
    ]
    scale_list = [
        0,
        0,
        np.array(-1.0),
        1,
        np.array(1),
        np.array(1)
    ]
    for (shape, scale) in zip(shape_list, scale_list):
        assertRaises(ValueError, _test_gamma_exception, shape, scale)


@use_np
@pytest.mark.parametrize("shape", [(1,), (2, 2), (4, 2, 2)])
@pytest.mark.parametrize("a", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("b", [0.5, 1.0, 1.5])
def test_gamma_grad(shape, a, b):
    class TestGammaGrad(HybridBlock):
        def __init__(self, size, beta):
            super(TestGammaGrad, self).__init__()
            self._size = size
            self._beta = beta

        def forward(self, a):
            return np.random.gamma(a, self._beta, size=self._size)

    for hybridize in [True, False]:
        param = np.ones(shape) * a
        param.attach_grad()
        net = TestGammaGrad(shape, b)
        if hybridize:
            net.hybridize()
        with mx.autograd.record():
            samples = net(param)
        samples.backward()
        # Check shape
        assert param.grad.shape == param.shape
        # Check correctness
        cdf = ss.gamma.cdf
        log_pdf = ss.gamma.logpdf
        eps = (0.01 * param / (1.0 + param ** 0.5)).asnumpy()
        x = samples.asnumpy().astype('float64') / b
        # d(cdf(x;alpha,beta))/d(alpha)
        cdf_alpha = (cdf(x, param.asnumpy() + eps) -
                        cdf(x, param.asnumpy() - eps)) / (2 * eps)
        # d(cdf(x;alpha,beta))/d(x)
        log_cdf_x = log_pdf(x, param.asnumpy())
        expected_grad = -b * cdf_alpha / onp.exp(log_cdf_x)
        assert_almost_equal(expected_grad, param.grad.asnumpy(), rtol=1e-2, atol=1e-3)


@use_np
def test_np_random_beta():
    class TestRandomBeta(HybridBlock):
        def __init__(self, size=None, dtype=None, device=None):
            super(TestRandomBeta, self).__init__()
            self._size = size
            self._dtype = dtype
            self._device = device

        def forward(self, a, b):
            return np.random.beta(a, b, size=self._size, dtype=self._dtype, device=self._device)

    def _test_random_beta_range(output):
        bigger_than_zero = onp.all(output > 0)
        smaller_than_one = onp.all(output < 1)
        return bigger_than_zero and smaller_than_one

    # Starting with numpy 1.19.0, output shape of () is no longer supported
    shape_list = [(0,), (1,), (2, 3), (4, 0, 5), 6, (7, 8), None]
    # since fp16 might incur precision issue, the corresponding test is skipped
    dtype_list = [np.float32, np.float64]
    hybridize_list = [False, True]
    data = np.array([1])
    for [param_shape, in_dtype, out_dtype, hybridize] in itertools.product(shape_list,
            dtype_list, dtype_list, hybridize_list):
        mx_data = data.astype(in_dtype)
        np_data = mx_data.asnumpy()
        test_random_beta = TestRandomBeta(size=param_shape, dtype=out_dtype)
        if hybridize:
            test_random_beta.hybridize()
        np_out = onp.random.beta(np_data, np_data, size=param_shape)
        mx_out = test_random_beta(mx_data, mx_data)
        mx_out_imperative = mx.np.random.beta(mx_data, mx_data, size=param_shape, dtype=out_dtype)

        assert np_out.shape == mx_out.shape
        assert np_out.shape == mx_out_imperative.shape
        assert _test_random_beta_range(mx_out.asnumpy()) == True
        assert _test_random_beta_range(mx_out_imperative.asnumpy()) == True

        # test scalar
        mx_out_imperative = mx.np.random.beta(1, 1, size=param_shape, dtype=out_dtype)
        assert _test_random_beta_range(mx_out_imperative.asnumpy()) == True


@use_np
def test_np_random_f():
    class TestRandomF(HybridBlock):
        def __init__(self, size=None):
            super(TestRandomF, self).__init__()
            self._size = size

        def forward(self, dfnum, dfden):
            return np.random.f(dfnum, dfden, size=self._size)

    # Starting with numpy 1.19.0, output shape of () is no longer supported
    shape_list = [(0,), (1,), (2, 3), (4, 0, 5), 6, (7, 8), None]
    hybridize_list = [False, True]
    df = np.array([1])
    for [param_shape, hybridize] in itertools.product(shape_list,
         hybridize_list):
        if sys.version_info.major < 3 and param_shape == ():
            continue
        mx_df = df
        np_df = mx_df.asnumpy()
        test_random_f = TestRandomF(size=param_shape)
        if hybridize:
            test_random_f.hybridize()
        np_out = onp.random.f(np_df, np_df, size=param_shape)
        mx_out = test_random_f(mx_df, mx_df)
        mx_out_imperative = mx.np.random.f(mx_df, mx_df, size=param_shape)

        assert np_out.shape == mx_out.shape
        assert np_out.shape == mx_out_imperative.shape


@use_np
def test_np_random_chisquare():
    class TestRandomChisquare(HybridBlock):
        def __init__(self, size=None, dtype=None, device=None):
            super(TestRandomChisquare, self).__init__()
            self._size = size
            self._dtype = dtype
            self._device = device

        def forward(self, df):
            return np.random.chisquare(df, size=self._size, dtype=self._dtype, device=self._device)

    # Starting with numpy 1.19.0, output shape of () is no longer supported
    shape_list = [(0,), (1,), (2, 3), (4, 0, 5), 6, (7, 8), None]

    dtype_list = [np.float16, np.float32, np.float64]
    hybridize_list = [False, True]
    df = np.array([1])
    for [param_shape, in_dtype, out_dtype, hybridize] in itertools.product(shape_list,
            dtype_list, dtype_list, hybridize_list):
        if sys.version_info.major < 3 and param_shape == ():
            continue
        mx_df = df.astype(in_dtype)
        np_df = mx_df.asnumpy()
        test_random_chisquare = TestRandomChisquare(size=param_shape, dtype=out_dtype)
        if hybridize:
            test_random_chisquare.hybridize()
        np_out = onp.random.chisquare(np_df, size=param_shape)
        mx_out = test_random_chisquare(mx_df)
        mx_out_imperative = mx.np.random.chisquare(mx_df, size=param_shape, dtype=out_dtype)

        assert np_out.shape == mx_out.shape
        assert np_out.shape == mx_out_imperative.shape


@use_np
def test_np_random_rayleigh():
    class TestRayleigh(HybridBlock):
        def __init__(self, shape):
            super(TestRayleigh, self).__init__()
            self._shape = shape

        def forward(self, scale):
            return np.random.rayleigh(scale, self._shape)

    shapes = [(2, 3), (4, 0, 5), (7, 8)]
    for hybridize in [False, True]:
        for shape in shapes:
            test_rayleigh = TestRayleigh(shape)
            if hybridize:
                test_rayleigh.hybridize()

            scale = np.ones(shape)
            scale.attach_grad()
            with mx.autograd.record():
                mx_out = test_rayleigh(scale)
            np_out = onp.random.rayleigh(scale = scale.asnumpy(), size = shape)
            assert np_out.shape == mx_out.shape
            mx_out.backward()
            assert scale.grad.shape == shape
            assert_almost_equal(scale.grad.asnumpy().sum(), mx_out.asnumpy().sum(), rtol=1e-3, atol=1e-5)

    for shape in shapes:
        mx_out = np.random.rayleigh(np.array([1]), shape)
        np_out = onp.random.rayleigh(np.array([1]).asnumpy(), shape)
        assert mx_out.asnumpy().shape == np_out.shape

    def _test_rayleigh_exception(scale):
        output = np.random.rayleigh(scale=scale).asnumpy()
    assertRaises(ValueError, _test_rayleigh_exception, -1)


@use_np
def test_np_exponential():
    class TestRandomExp(HybridBlock):
        def __init__(self, shape):
            super(TestRandomExp, self).__init__()
            self._shape = shape

        def forward(self, scale):
            return np.random.exponential(scale, self._shape)

    output_shapes = [
        (3, 2),
        (4, 3, 2, 2),
        (3, 4, 5)
    ]
    for hybridize in [False, True]:
        for out_shape in output_shapes:
            test_exponential_grad = TestRandomExp(out_shape)
            if hybridize:
                test_exponential_grad.hybridize()
            scale = np.ones(out_shape)
            scale.attach_grad()
            with mx.autograd.record():
                mx_out = test_exponential_grad(scale)
            np_out = onp.random.exponential(scale = scale.asnumpy(), size = out_shape)
            assert np_out.shape == mx_out.shape
            mx_out.backward()
            assert scale.grad.shape == out_shape
            assert_almost_equal(scale.grad.asnumpy().sum(), mx_out.asnumpy().sum(), rtol=1e-3, atol=1e-5)

    def _test_exponential_exception(scale):
        output = np.random.exponential(scale=scale).asnumpy()
    assertRaises(ValueError, _test_exponential_exception, -1)


@use_np
def test_np_random_a():
    op_names = ['pareto', 'power', 'weibull']
    # these distributions have one required parameter a
    shapes = [(1,), (2, 3), (4, 0, 5), 6, (7, 8), (), None]

    def _test_random_x_range(output):
        ge_zero = onp.all(output >= 0)
        smaller_equal_one = onp.all(output <= 1)
        return ge_zero and smaller_equal_one

    # test imperative size shapes
    for [shape, op_name] in itertools.product(shapes, op_names):
        op = getattr(np.random, op_name, None)
        assert op is not None
        out = op(1.0, size=shape)
        expected_shape = shape
        if not isinstance(shape, tuple):
            expected_shape = () if shape is None else (shape,)
        assert out.shape == expected_shape
        # test range of generated values for power distribution
        if op_name == 'power':
            assert _test_random_x_range(out.asnumpy()) == True

    # test symbolic/hybridized size shapes
    class TestRandomA(HybridBlock):
        def __init__(self, shape, op_name):
            super(TestRandomA, self).__init__()
            self._shape = shape
            self._op_name = op_name

        def forward(self, a):
            op = getattr(np.random, self._op_name, None)
            assert op is not None
            return op(a, size=self._shape)

    hybridize = [False, True]
    for [op_name, shape, hybridize] in itertools.product(op_names, shapes, hybridize):
        test_op = TestRandomA(shape, op_name)
        if hybridize:
            test_op.hybridize()
        mx_out = test_op(np.array(1.0))
        expected_shape = shape
        if not isinstance(shape, tuple):
            expected_shape = () if shape is None else (shape,)
        assert mx_out.shape == expected_shape

    # test broadcasting of required parameter a shape when a is array-like
    ashapes = [(1,), (2, 3), (4, 0, 5), 6, (7, 8)]
    for shape in ashapes:
        a = np.ones(shape)
        for op_name in op_names:
            op = getattr(np.random, op_name, None)
            assert op is not None
            mx_out = op(a, size=None)
            expected_shape = a.shape
            assert mx_out.shape == expected_shape

    # test illegal parameter values
    def _test_exception(a):
        output = op(a=a).asnumpy()
    for op in op_names:
        op = getattr(np.random, op_name, None)
        if op is not None:
            assertRaises(ValueError, _test_exception, -1)
            assertRaises(ValueError, _test_exception, 0)


@use_np
def test_np_weibull_grad():
    class TestRandomW(HybridBlock):
        def __init__(self, shape):
            super(TestRandomW, self).__init__()
            self._shape = shape

        def forward(self, a):
            return np.random.weibull(a, self._shape)

    output_shapes = [
        (3, 2),
        (4, 3, 2, 2),
        (3, 4, 5)
    ]
    for hybridize in [False, True]:
        for out_shape in output_shapes:
            test_w_grad = TestRandomW(out_shape)
            if hybridize:
                test_w_grad.hybridize()
            a = np.ones(out_shape)
            a.attach_grad()
            with mx.autograd.record():
                mx_out = test_w_grad(a)
            mx_out.backward()

            # gradient formula calculus (a=1)
            formula_grad = - mx_out * np.log(mx_out)
            assert a.grad.shape == out_shape
            assert_almost_equal(a.grad.asnumpy().sum(), formula_grad.asnumpy().sum(), rtol=1e-3, atol=1e-5)


@use_np
def test_np_pareto_grad():
    class TestRandomP(HybridBlock):
        def __init__(self, shape):
            super(TestRandomP, self).__init__()
            self._shape = shape

        def forward(self, a):
            return np.random.pareto(a, self._shape)

    output_shapes = [
        (3, 2),
        (4, 3, 2, 2),
        (3, 4, 5)
    ]
    for hybridize in [False, True]:
        for out_shape in output_shapes:
            test_w_grad = TestRandomP(out_shape)
            if hybridize:
                test_w_grad.hybridize()
            a = np.ones(out_shape)
            a.attach_grad()
            with mx.autograd.record():
                mx_out = test_w_grad(a)
            mx_out.backward()

            # gradient formula from calculus (a=1)
            noise = np.log(mx_out + np.ones(mx_out.shape))
            formula_grad = - (mx_out + np.ones(mx_out.shape)) * noise
            assert a.grad.shape == out_shape
            assert_almost_equal(a.grad.asnumpy().sum(), formula_grad.asnumpy().sum(), rtol=1e-3, atol=1e-5)


@use_np
def test_np_randn():
    # Test shapes.
    shapes = [
        (3, 3),
        (3, 4),
        (0, 0),
        (3, 3, 3),
        (0, 0, 0),
        (2, 2, 4, 3),
        (2, 2, 4, 3),
        (2, 0, 3, 0),
        (2, 0, 2, 3)
    ]
    dtypes = ['float16', 'float32', 'float64']
    for dtype in dtypes:
        for shape in shapes:
            data_mx = np.random.randn(*shape, dtype=dtype)
            assert data_mx.shape == shape


@use_np
@pytest.mark.skip(reason='Test hangs. Tracked in #18144')
def test_np_multivariate_normal():
    class TestMultivariateNormal(HybridBlock):
        def __init__(self, size=None):
            super(TestMultivariateNormal, self).__init__()
            self.size = size

        def forward(self, mean, cov):
            return np.random.multivariate_normal(mean, cov, self.size)

    hybridize_list = [True, False]
    dtypes = ['float16', 'float32', 'float64']
    size_list = [None, 1, (), (2, 3), (2, 0)]
    # [mean_shape, cov_shape]: onp.broadcast(mean_shape, cov_shape[:-1]) should not raise error
    batch_shape_list = [[(2,), (2, 2)], [(3, 2), (2, 2)], [(2,), (3, 2, 2)], [(3, 2), (4, 3, 2, 2)]]
    # most basic case for mean and cov
    mean = np.array([0.123456789, 10])
    cov = np.array([[1, 0], [0, 10]])

    for [hybridize, dtype, size, batch_shape] in itertools.product(hybridize_list,\
                dtypes, size_list, batch_shape_list):
        # simplest case: 1-d, 0 batch
        # compared with official numpy
        mean_shape = batch_shape[0]
        cov_shape = batch_shape[1]
        new_mean = np.broadcast_to(mean, mean_shape).astype(dtype)
        new_cov = np.broadcast_to(cov, cov_shape).astype(dtype)

        test_multivariate_normal = TestMultivariateNormal(size)
        if hybridize:
            test_multivariate_normal.hybridize()

        test_shape = test_multivariate_normal(new_mean, new_cov).shape
        actual_shape = np.random.multivariate_normal(new_mean, new_cov, size).shape

        desired_shape = np.broadcast_arrays(np.empty(mean_shape), np.empty(cov_shape[:-1]))[0].shape

        if size is not None:
            size = [size] if isinstance(size, int) else list(size)
            desired_shape = size + list(desired_shape)

        assert list(desired_shape) == list(test_shape)
        assert list(desired_shape) == list(actual_shape)


@use_np
def test_npx_categorical():
    class TestNumpyCategorical(HybridBlock):
        def __init__(self, size=None):
            super(TestNumpyCategorical, self).__init__()
            self.size = size

        def forward(self, prob):
            if self.size is None:
                return npx.random.categorical(prob)
            return npx.random.categorical(prob, shape=self.size)

    batch_sizes = [(2,), (2, 3)]
    event_shapes = [None, (10,), (10, 12)]
    num_event = [2, 4, 10]
    for batch_size, num_event, event_shape in itertools.product(batch_sizes, num_event, event_shapes):
        for hybridize in [True, False]:
            prob = np.ones(batch_size + (num_event,)) / num_event
            net = TestNumpyCategorical(event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(prob)
            desired_shape = batch_size + event_shape if event_shape is not None else batch_size
            assert mx_out.shape == desired_shape


@use_np
def test_npx_multinomial():
    class TestNumpyMultinomial(HybridBlock):
        def __init__(self, size=None):
            super(TestNumpyMultinomial, self).__init__()
            self.size = size

        def forward(self, n, prob):
            if self.size is None:
                return npx.random.multinomial(n, prob)
            return npx.random.multinomial(n, prob, shape=self.size)

    batch_sizes = [(2,), (2, 3)]
    event_shapes = [None, (10,), (10, 12)]
    num_event = [2, 4, 10]
    for batch_size, num_event, event_shape in itertools.product(batch_sizes, num_event, event_shapes):
        for hybridize in [True, False]:
            n = np.ones(batch_size)
            prob = np.ones(batch_size + (num_event,)) / num_event
            net = TestNumpyMultinomial(event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(n, prob)
            desired_shape = batch_size + event_shape + (num_event,) if event_shape is not None else batch_size + (num_event,)
            assert mx_out.shape == desired_shape


@use_np
def test_random_seed():
    for seed in [234, 594, 7240, 20394]:
        ret = []
        for _ in range(2):
            npx.random.seed(seed=seed)
            ret.append(np.random.uniform(size=(2, 3)))
        assert_almost_equal(ret[0].asnumpy(), ret[1].asnumpy(), rtol=1e-4, atol=1e-5, use_broadcast=False)


@use_np
def test_np_cumsum():
    def np_cumsum_backward(ograd, axis=None, dtype=None):
        return onp.flip(onp.cumsum(onp.flip(ograd, axis=axis), axis=axis, dtype=dtype), axis=axis)

    class TestCumsum(HybridBlock):
        def __init__(self, axis=None, dtype=None):
            super(TestCumsum, self).__init__()
            self._axis = axis
            self._dtype = dtype

        def forward(self, a):
            return a.cumsum(axis=self._axis, dtype=self._dtype)

    shapes = [(2, 3, 4), (2, 0, 3), ()]
    for hybridize in [True, False]:
        for shape in shapes:
            for axis in [None] + [i for i in range(0, len(shape))]:
                for otype in [None, onp.float32, onp.float64]:
                    test_cumsum = TestCumsum(axis=axis, dtype=otype)
                    if hybridize:
                        test_cumsum.hybridize()
                    for itype in [onp.float16, onp.float32, onp.float64]:
                        x = rand_ndarray(shape).astype(itype).as_np_ndarray()
                        x.attach_grad()
                        np_out = onp.cumsum(x.asnumpy(), axis=axis, dtype=otype)
                        with mx.autograd.record():
                            mx_out = test_cumsum(x)
                        assert mx_out.shape == np_out.shape
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                        mx_out.backward()
                        np_backward = np_cumsum_backward(onp.ones(np_out.shape, dtype=otype),
                                                         axis=axis, dtype=otype).reshape(x.shape)
                        assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

                        mx_out = np.cumsum(x, axis=axis, dtype=otype)
                        np_out = onp.cumsum(x.asnumpy(), axis=axis, dtype=otype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    for shape in shapes:
        for axis in [None] + [i for i in range(0, len(shape))]:
            for otype in [None, onp.int32, onp.int64]:
                for itype in [onp.bool, onp.int8, onp.int32, onp.int64]:
                    x = rand_ndarray(shape).astype(itype).as_np_ndarray()
                    np_out = onp.cumsum(x.asnumpy(), axis=axis, dtype=otype)
                    mx_out = np.cumsum(x, axis=axis, dtype=otype)
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
@pytest.mark.skip(reason='Skipped as the test is flaky and the feature causes curand error. Tracked in #18100')
def test_np_histogram():
    shapes = [(), (3, 4), (3, 0)]

    for shape in shapes:
        mx_a = np.random.uniform(0.0, 10.0, size=shape)
        np_a = mx_a.asnumpy()
        mx_bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5., 6., 7., 8., 9., 10.])
        np_bins = mx_bins.asnumpy()
        for bins, _range in [(20, (0.0, 10.0)), (mx_bins, None)]:
            mx_cnts, mx_bins = np.histogram(mx_a, bins=bins, range=_range)
            np_cnts, np_bins = onp.histogram(np_a, bins=bins if isinstance(bins, mx.base.numeric_types) else bins.asnumpy(), range=_range)
            assert_almost_equal(mx_cnts.asnumpy(), np_cnts, rtol=1e-3, atol=1e-5)
            assert_almost_equal(mx_bins.asnumpy(), np_bins, rtol=1e-3, atol=1e-5)


@use_np
@pytest.mark.skip(reason='Skipped as the test is flaky and the feature causes curand error. Tracked in #18100')
def test_np_choice():
    class TestUniformChoice(HybridBlock):
        def __init__(self, sample_size, replace):
            super(TestUniformChoice, self).__init__()
            self.sample_size = sample_size
            self.replace = replace

        def forward(self, a):
            return np.random.choice(a=a, size=self.sample_size, replace=self.replace, p=None)

    class TestWeightedChoice(HybridBlock):
        def __init__(self, sample_size, replace):
            super(TestWeightedChoice, self).__init__()
            self.sample_size = sample_size
            self.replace = replace

        def forward(self, a, p):
            op = getattr(np.random, "choice", None)
            return np.random.choice(a, self.sample_size, self.replace, p)

    def test_sample_with_replacement(sampler, num_classes, shape, weight=None):
        samples = sampler(num_classes, shape, replace=True, p=weight).asnumpy()
        generated_density = onp.histogram(samples, onp.arange(num_classes + 1), density=True)[0]
        expected_density = (weight.asnumpy() if weight is not None else
                            onp.array([1 / num_classes] * num_classes))
        # test almost equal
        assert_almost_equal(generated_density, expected_density, rtol=1e-1, atol=1e-1)
        # test shape
        assert (samples.shape == shape)

    def test_sample_without_replacement(sampler, num_classes, shape, num_trials, weight=None):
        samples = sampler(num_classes, shape, replace=False, p=weight).asnumpy()
        # Check shape and uniqueness
        assert samples.shape == shape
        assert len(onp.unique(samples)) == samples.size
        # Check distribution
        bins = onp.zeros((num_classes))
        expected_freq = (weight.asnumpy() if weight is not None else
                         onp.array([1 / num_classes] * num_classes))
        for _ in range(num_trials):
            out = sampler(num_classes, 1, replace=False, p=weight).item()
            bins[out] += 1
        bins /= num_trials
        assert_almost_equal(bins, expected_freq, rtol=1e-1, atol=1e-1)

    def test_indexing_mode(sampler, set_size, samples_size, replace, weight=None):
        a = np.arange(set_size)
        if weight is not None:
            samples = sampler(a, weight)
        else:
            samples = sampler(a)
        assert len(samples) == samples_size
        if not replace:
            assert len(onp.unique(samples.asnumpy())) == samples_size

    num_classes = 10
    num_samples = 10 ** 8
    # Density tests are commented out due to their huge time comsumption.
    # Tests passed locally.
    # shape_list1 = [
    #     (10 ** 8, 1),
    #     (10 ** 5, 10 ** 3),
    #     (10 ** 2, 10 ** 3, 10 ** 3)
    # ]
    # for shape in shape_list1:
    #     test_sample_with_replacement(np.random.choice, num_classes, shape)
    #     weight = np.array(onp.random.dirichlet([1.0] * num_classes))
    #     test_sample_with_replacement(np.random.choice, num_classes, shape, weight)

    # Tests passed locally,
    # commented out for the same reason as above.
    # shape_list2 = [
    #     (6, 1),
    #     (2, 3),
    #     (1, 2, 3),
    #     (2, 2),
    # ]
    # for shape in shape_list2:
    #     test_sample_without_replacement(np.random.choice, num_classes, shape, 10 ** 5)
    #     weight = np.array(onp.random.dirichlet([1.0] * num_classes))
    #     test_sample_without_replacement(np.random.choice, num_classes, shape, 10 ** 5, weight)

    # Test hypridize mode:
    for wtype in ['float16', 'float32', 'float64']:
        for hybridize in [True, False]:
            for replace in [True, False]:
                test_choice = TestUniformChoice(num_classes // 2, replace)
                test_choice_weighted = TestWeightedChoice(num_classes // 2, replace)
                if hybridize:
                    test_choice.hybridize()
                    test_choice_weighted.hybridize()
                weight = np.array(onp.random.dirichlet([1.0] * num_classes)).astype(wtype)
                test_indexing_mode(test_choice, num_classes, num_classes // 2, replace, None)
                test_indexing_mode(test_choice_weighted, num_classes, num_classes // 2, replace, weight)


@use_np
def test_np_eye():
    configs = [
        4,
        1000,
        (4, 3),
        (5, None),
        (4, None, 1),
        (2, 2, 1),
        (4, 6, 1),
        (7, 3, -3),
        (3, 2, -2),
        (4, 0),
        (0, 0),
        (0, 3),
        (0, 0, -2)
    ]
    exception_configs = [
        -1,
        -1000,
        (-2, None),
        (1, -1)
    ]
    dtypes = ['int32', 'float16', 'float32', 'float64', None]
    for config in configs:
        for dtype in dtypes:
            if isinstance(config, tuple):
                mx_ret = np.eye(*config, dtype=dtype)
                np_ret = onp.eye(*config, dtype=dtype)
            else:
                mx_ret = np.eye(config, dtype=dtype)
                np_ret = onp.eye(config, dtype=dtype)
            assert same(mx_ret.asnumpy(), np_ret)
    # check for exception input
    for config in exception_configs:
        if isinstance(config, tuple):
            assertRaises(MXNetError, np.eye, *config)
        else:
            assertRaises(MXNetError, np.eye, config)

    class TestEye(HybridBlock):
        def __init__(self, N, M=None, k=0, dtype=None):
            super(TestEye, self).__init__()
            self._N = N
            self._M = M
            self._k = k
            self._dtype = dtype

        def forward(self, x):
            return x + np.eye(self._N, self._M, self._k, dtype=self._dtype)

    for dtype in dtypes:
        x = np.zeros(shape=(), dtype=dtype)
        for config in configs:
            for hybridize in [False, True]:
                if isinstance(config, tuple):
                    net = TestEye(*config, dtype=dtype)
                    np_out = onp.eye(*config, dtype=dtype)
                else:
                    net = TestEye(config, dtype=dtype)
                    np_out = onp.eye(config, dtype=dtype)
                if hybridize:
                    net.hybridize()
                mx_out = net(x)
                assert same(mx_out.asnumpy(), np_out)


@use_np
def test_np_indices():
    dtypes = ['int32', 'int64', 'float16', 'float32', 'float64']
    shapes = [
        (0,),
        (3,),
        (2, 3, 4),
        (2, 0, 4),
        (1, 1, 1, 1),
        (1, 0, 0, 1),
        (2, 3, 4, 5, 6, 7)
    ]
    if platform.system() == 'Windows':
        shapes = shapes[1:]  # beacuse in numpy windows version, indces not support dimensions is empty tuple.
    for dtype in dtypes:
        for shape in shapes:
            np_out = onp.indices(dimensions=shape, dtype=dtype)
            mx_out = np.indices(dimensions=shape, dtype=dtype)
            assert same(mx_out.asnumpy(), np_out)
            assert mx_out.shape == np_out.shape

    @use_np
    class TestIndices(HybridBlock):
        def __init__(self, dimensions=None, dtype=None):
            super(TestIndices, self).__init__()
            self._dimensions = dimensions
            self._dtype = dtype

        def forward(self, x):
            return x + np.indices(dimensions=self._dimensions, dtype=self._dtype)

    for dtype in dtypes:
        for shape in shapes:
            x = np.zeros(shape=(), dtype=dtype)
            for hybridize in [False, True]:
                net = TestIndices(dimensions=shape, dtype=dtype)
                np_out = onp.indices(dimensions=shape, dtype=dtype)
                if hybridize:
                    net.hybridize()
                mx_out = net(x)
                assert same(mx_out.asnumpy(), np_out)
                assert mx_out.shape == np_out.shape


@use_np
def test_np_repeat():
    config = [
        ((), 2, None),
        ((), 0, None),
        ((4, 2), 2, None),
        ((4, 2), 2, 0),
        ((4, 2), 2, 1),
        ((4, 2), 2, -1),
        ((4, 2), [2,3] * 4, None),
        ((4, 2), [1,2], 1),
    ]

    class TestRepeat(HybridBlock):
        def __init__(self, repeats, axis=None):
            super(TestRepeat, self).__init__()
            self._repeats = repeats
            self._axis = axis

        def forward(self, x):
            return x.repeat(self._repeats, self._axis)

    for shape, repeats, axis in config:
        data_np = onp.random.randint(low=0, high=1000, size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = data_np.repeat(repeats, axis)
        ret_mx = data_mx.repeat(repeats, axis)
        assert same(ret_mx.asnumpy(), ret_np)

        net = TestRepeat(repeats, axis)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            ret_mx = net(data_mx)
            assert same(ret_mx.asnumpy(), ret_np)


@use_np
def test_np_linalg_norm():
    class TestLinalgNorm(HybridBlock):
        def __init__(self, ord=None, axis=None, keepdims=False):
            super(TestLinalgNorm, self).__init__()
            self._ord = ord
            self._axis = axis
            self._keepdims = keepdims

        def forward(self, x):
            return np.linalg.norm(x, ord=self._ord, axis=self._axis, keepdims=self._keepdims)

    configs = [
        ((2, 3, 4), 1, (2, 1)),
        ((2, 3, 4), 2, (1, 2)),
        ((2, 3, 4), None, None),
        ((3,), None, None),
        ((2, 3), 2, 1),
        ((2, 3, 4), 1, 1),
        ((2, 3, 4), -1, 2),
        ((2, 3, 4), 2, 1),
        ((2, 3, 4), 4, 1),
        ((2, 3, 0, 4), -2, 1),
        ((2, 3, 4, 5), 2, (2, 3)),
        ((2, 3), -1, None),
        ((2, 3, 4), 'inf', 1),
        ((2, 3, 4), '-inf', (1, 0)),
        ((2, 3), None, (0, 1)),
        ((3, 2, 3), None, (1, 2)),
        ((2, 3), None, None),
        ((2, 3, 4), 'fro', (0, 2)),
        ((2, 0, 4), 'fro', (0, 2)),
        ((2, 3, 4), None, (0, 2)),
        ((2, 3, 4), -3.2, 2),
        ((2, 3, 4), -1, (0, 1)),
        ((2, 3, 4), 'inf', (0, 2)),
        ((2, 3, 4), '-inf', (0, 2)),
        ((4, 4, 4, 4), -2, (0, 2)),
        ((2, 3, 4), 'nuc', (0, 2)),
        ((2, 2), 'nuc', None),
    ]

    def spectral_norm_grad(data):
        with mx.autograd.record():
            UT, S, V = np.linalg.svd(data)
            norm = np.max(np.abs(S), axis=-1)
        norm.backward()
        return data.grad.asnumpy()

    # numpy is flaky under float16, also gesvd does not support fp16
    dtypes = [np.float32, np.float64]
    for hybridize, itype, (shape, ord, axis), keepdims in \
        itertools.product([True, False], dtypes, configs, [True, False]):
        net = TestLinalgNorm(ord, axis, keepdims)
        rtol = 1e-2
        atol = 1e-2
        if hybridize:
            net.hybridize()
        a = mx.nd.random.uniform(-10.0, 10.0, shape=shape, dtype=itype).as_np_ndarray()
        a.attach_grad()
        with mx.autograd.record():
            mx_ret = net(a)
        if ord == 'inf':
            np_ret = onp.linalg.norm(a.asnumpy(), ord=onp.inf, axis=axis, keepdims=keepdims)
        elif ord == '-inf':
            np_ret = onp.linalg.norm(a.asnumpy(), ord=-onp.inf, axis=axis, keepdims=keepdims)
        else:
            np_ret = onp.linalg.norm(a.asnumpy(), ord=ord, axis=axis, keepdims=keepdims)

        assert np_ret.shape == mx_ret.shape
        assert_almost_equal(mx_ret.asnumpy(), np_ret, rtol=rtol, atol=atol)

        mx_ret.backward()

        grad_axis = axis
        if axis is None and len(shape) >= 2 and ord is not None:
            grad_axis = (len(shape) - 2, len(shape) - 1)
        elif axis is None and ord is None:
            grad_axis = tuple([i for i in range(len(shape))])
        elif axis is None:
            grad_axis = len(shape) - 1

        if not keepdims and isinstance(grad_axis, tuple):
            if len(grad_axis) == 2 and grad_axis[0] > grad_axis[1] and grad_axis[0] > len(np_ret.shape):
                grad_axis = (grad_axis[1], grad_axis[0])
            for i in grad_axis:
                np_ret = onp.expand_dims(np_ret, axis=i)
        elif not keepdims:
            np_ret = onp.expand_dims(np_ret, axis=grad_axis)

        if ord == 4:
            backward_expected = onp.sign(a.asnumpy()) * onp.power(onp.abs(a.asnumpy()) / np_ret, ord - 1)
            assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

        if ord == 2 and not isinstance(grad_axis, tuple):
            backward_expected = onp.divide(a.asnumpy(), np_ret)
            assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)
        elif ord == 2 and isinstance(grad_axis, tuple):
            backward_expected = spectral_norm_grad(a)
            assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

        if ord == 'fro':
            backward_expected = onp.divide(a.asnumpy(), np_ret)
            assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

        assert a.grad.shape == a.shape

        # Test imperative once again
        if ord == 'inf':
            np_ret = onp.linalg.norm(a.asnumpy(), ord=onp.inf, axis=axis, keepdims=keepdims)
        elif ord == '-inf':
            np_ret = onp.linalg.norm(a.asnumpy(), ord=-onp.inf, axis=axis, keepdims=keepdims)
        else:
            np_ret = onp.linalg.norm(a.asnumpy(), ord=ord, axis=axis, keepdims=keepdims)
        mx_ret = np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
        assert_almost_equal(mx_ret.asnumpy(), np_ret, rtol=rtol, atol=atol)


@use_np
@pytest.mark.parametrize('shape,ord,axis', [
    ((2, 3, 4), 2, (1, 2)),
    ((2, 3, 4), None, None),
    ((3,), None, None),
    ((2, 3), 2, 1),
    ((2, 3, 4), 1, 1),
    ((2, 3, 4), -1, 2),
    ((2, 3, 4), 2, 1),
    ((2, 3, 4), 4, 1),
    ((2, 3, 0, 4), -2, 1),
    ((2, 3, 4, 5), 2, (2, 3)),
    ((2, 3, 4), 'inf', 1),
    ((2, 3, 4), '-inf', (1, 0)),
    ((2, 3), None, (0, 1)),
    ((3, 2, 3), None, (1, 2)),
    ((2, 3), None, None),
    ((2, 3, 4), None, (0, 2)),
    ((2, 3, 4), -3.2, 2),
    ((2, 3, 4), 'inf', (0, 2)),
    ((2, 3, 4), '-inf', (0, 2)),
    ((2, 3, 4, 5, 7), 2, (2, 3, 1)),
])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('itype', [np.float32, np.float64])
@pytest.mark.parametrize('keepdims', [True, False])
def test_np_linalg_vector_norm(shape, ord, axis, hybridize, itype, keepdims):
    class TestLinalgVectNorm(HybridBlock):
        def __init__(self, ord=None, axis=None, keepdims=False):
            super(TestLinalgVectNorm, self).__init__()
            self._ord = ord
            self._axis = axis
            self._keepdims = keepdims

        def forward(self, x):
            return np.linalg.vector_norm(x, ord=self._ord, axis=self._axis, keepdims=self._keepdims)

    def spectral_norm_grad(data):
        with mx.autograd.record():
            UT, S, V = np.linalg.svd(data)
            norm = np.max(np.abs(S), axis=-1)
        norm.backward()
        return data.grad.asnumpy()
    
    def onp_vector_norm(a, axis=None, keepdims=False, ord=2):
        if axis is None:
            a = a.flatten()
            axis = 0
        elif isinstance(axis, tuple):
            # Note: The axis argument supports any number of axes, whereas norm()
            # only supports a single axis for vector norm.
            rest = tuple(i for i in range(a.ndim) if i not in axis)
            newshape = axis + rest
            a = onp.transpose(a, newshape).reshape((reduce(lambda x, y: x * y, [a.shape[x] for x in axis]), *[a.shape[i] for i in rest]))
            axis = 0
        return onp.linalg.norm(a, axis=axis, keepdims=keepdims, ord=ord)

    # numpy is flaky under float16, also gesvd does not support fp16
    net = TestLinalgVectNorm(ord, axis, keepdims)
    rtol = 1e-2
    atol = 1e-2
    if hybridize:
        net.hybridize()
    a = mx.np.random.uniform(-10.0, 10.0, size=shape, dtype=itype)
    a.attach_grad()
    with mx.autograd.record():
        mx_ret = net(a)
    if ord == 'inf':
        np_ret = onp_vector_norm(a.asnumpy(), ord=onp.inf, axis=axis, keepdims=keepdims)
    elif ord == '-inf':
        np_ret = onp_vector_norm(a.asnumpy(), ord=-onp.inf, axis=axis, keepdims=keepdims)
    else:
        np_ret = onp_vector_norm(a.asnumpy(), ord=ord, axis=axis, keepdims=keepdims)

    assert np_ret.shape == mx_ret.shape
    assert_almost_equal(mx_ret.asnumpy(), np_ret, rtol=rtol, atol=atol)

    mx_ret.backward()

    grad_axis = axis
    if axis is None and len(shape) >= 2 and ord is not None:
        grad_axis = (len(shape) - 2, len(shape) - 1)
    elif axis is None and ord is None:
        grad_axis = tuple([i for i in range(len(shape))])
    elif axis is None:
        grad_axis = len(shape) - 1

    if not keepdims and isinstance(grad_axis, tuple):
        if len(grad_axis) == 2 and grad_axis[0] > grad_axis[1] and grad_axis[0] > len(np_ret.shape):
            grad_axis = (grad_axis[1], grad_axis[0])
        for i in grad_axis:
            np_ret = onp.expand_dims(np_ret, axis=i)
    elif not keepdims:
        np_ret = onp.expand_dims(np_ret, axis=grad_axis)

    if ord == 4:
        backward_expected = onp.sign(a.asnumpy()) * onp.power(onp.abs(a.asnumpy()) / np_ret, ord - 1)
        assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

    if ord == 2 and not isinstance(grad_axis, tuple):
        backward_expected = onp.divide(a.asnumpy(), np_ret)
        assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)
    elif ord == 2 and isinstance(grad_axis, tuple):
        backward_expected = spectral_norm_grad(a)
        assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

    assert a.grad.shape == a.shape

    # Test imperative once again
    if ord == 'inf':
        np_ret = onp_vector_norm(a.asnumpy(), ord=onp.inf, axis=axis, keepdims=keepdims)
    elif ord == '-inf':
        np_ret = onp_vector_norm(a.asnumpy(), ord=-onp.inf, axis=axis, keepdims=keepdims)
    else:
        np_ret = onp_vector_norm(a.asnumpy(), ord=ord, axis=axis, keepdims=keepdims)
    mx_ret = np.linalg.vector_norm(a, ord=ord, axis=axis, keepdims=keepdims)
    assert_almost_equal(mx_ret.asnumpy(), np_ret, rtol=rtol, atol=atol)


@use_np
@pytest.mark.parametrize('shape,ord,axis', [
    ((2, 3, 4), 1, (2, 1)),
    ((2, 3, 4), 2, (1, 2)),
    ((2, 3, 4), None, None),
    ((3,), None, None),
    ((2, 3), 2, 1),
    ((2, 3, 4), 1, 1),
    ((2, 3, 4), -1, 2),
    ((2, 3, 4), 2, 1),
    ((2, 3, 4), 4, 1),
    ((2, 3, 0, 4), -2, 1),
    ((2, 3, 4, 5), 2, (2, 3)),
    ((2, 3), -1, None),
    ((2, 3, 4), 'inf', 1),
    ((2, 3, 4), '-inf', (1, 0)),
    ((2, 3), None, (0, 1)),
    ((3, 2, 3), None, (1, 2)),
    ((2, 3), None, None),
    ((2, 3, 4), 'fro', (0, 2)),
    ((2, 0, 4), 'fro', (0, 2)),
    ((2, 3, 4), None, (0, 2)),
    ((2, 3, 4), -3.2, 2),
    ((2, 3, 4), -1, (0, 1)),
    ((2, 3, 4), 'inf', (0, 2)),
    ((2, 3, 4), '-inf', (0, 2)),
    ((4, 4, 4, 4), -2, (0, 2)),
    ((2, 3, 4), 'nuc', (0, 2)),
    ((2, 2), 'nuc', None),
])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('itype', [np.float32, np.float64])
@pytest.mark.parametrize('keepdims', [True, False])
def test_np_linalg_matrix_norm(shape, ord, axis, hybridize, itype, keepdims):
    class TestLinalgMatNorm(HybridBlock):
        def __init__(self, ord=None, axis=None, keepdims=False):
            super(TestLinalgMatNorm, self).__init__()
            self._ord = ord
            self._axis = axis
            self._keepdims = keepdims

        def forward(self, x):
            return np.linalg.matrix_norm(x, ord=self._ord, axis=self._axis, keepdims=self._keepdims)

    def spectral_norm_grad(data):
        with mx.autograd.record():
            UT, S, V = np.linalg.svd(data)
            norm = np.max(np.abs(S), axis=-1)
        norm.backward()
        return data.grad.asnumpy()

    # numpy is flaky under float16, also gesvd does not support fp16
    net = TestLinalgMatNorm(ord, axis, keepdims)
    rtol = 1e-2
    atol = 1e-2
    if hybridize:
        net.hybridize()
    a = mx.np.random.uniform(-10.0, 10.0, size=shape, dtype=itype)
    if not isinstance(axis, tuple) or not len(axis) == 2:
        assertRaises(ValueError, np.linalg.matrix_norm, a, ord, axis, keepdims)
        return
    a.attach_grad()
    with mx.autograd.record():
        mx_ret = net(a)
    if ord == 'inf':
        np_ret = onp.linalg.norm(a.asnumpy(), ord=onp.inf, axis=axis, keepdims=keepdims)
    elif ord == '-inf':
        np_ret = onp.linalg.norm(a.asnumpy(), ord=-onp.inf, axis=axis, keepdims=keepdims)
    else:
        np_ret = onp.linalg.norm(a.asnumpy(), ord=ord, axis=axis, keepdims=keepdims)

    assert np_ret.shape == mx_ret.shape
    assert_almost_equal(mx_ret.asnumpy(), np_ret, rtol=rtol, atol=atol)

    mx_ret.backward()

    grad_axis = axis
    if axis is None and len(shape) >= 2 and ord is not None:
        grad_axis = (len(shape) - 2, len(shape) - 1)
    elif axis is None and ord is None:
        grad_axis = tuple([i for i in range(len(shape))])
    elif axis is None:
        grad_axis = len(shape) - 1

    if not keepdims and isinstance(grad_axis, tuple):
        if len(grad_axis) == 2 and grad_axis[0] > grad_axis[1] and grad_axis[0] > len(np_ret.shape):
            grad_axis = (grad_axis[1], grad_axis[0])
        for i in grad_axis:
            np_ret = onp.expand_dims(np_ret, axis=i)
    elif not keepdims:
        np_ret = onp.expand_dims(np_ret, axis=grad_axis)

    if ord == 4:
        backward_expected = onp.sign(a.asnumpy()) * onp.power(onp.abs(a.asnumpy()) / np_ret, ord - 1)
        assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

    if ord == 2 and not isinstance(grad_axis, tuple):
        backward_expected = onp.divide(a.asnumpy(), np_ret)
        assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)
    elif ord == 2 and isinstance(grad_axis, tuple):
        backward_expected = spectral_norm_grad(a)
        assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

    if ord == 'fro':
        backward_expected = onp.divide(a.asnumpy(), np_ret)
        assert_almost_equal(a.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

    assert a.grad.shape == a.shape

    # Test imperative once again
    if ord == 'inf':
        np_ret = onp.linalg.norm(a.asnumpy(), ord=onp.inf, axis=axis, keepdims=keepdims)
    elif ord == '-inf':
        np_ret = onp.linalg.norm(a.asnumpy(), ord=-onp.inf, axis=axis, keepdims=keepdims)
    else:
        np_ret = onp.linalg.norm(a.asnumpy(), ord=ord, axis=axis, keepdims=keepdims)
    mx_ret = np.linalg.matrix_norm(a, ord=ord, axis=axis, keepdims=keepdims)
    assert_almost_equal(mx_ret.asnumpy(), np_ret, rtol=rtol, atol=atol)


@use_np
@pytest.mark.parametrize('shape', [
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
])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('hybridize', [False, True])
def test_np_linalg_svd(shape, dtype, hybridize):
    class TestSVD(HybridBlock):
        def __init__(self):
            super(TestSVD, self).__init__()

        def forward(self, data):
            return np.linalg.svd(data)

    def get_grad(UT, L, V):
        m = V.shape[-2]
        n = V.shape[-1]
        E = onp.zeros_like(UT)
        dUT = onp.ones_like(UT)
        dV = onp.ones_like(V)
        for i in range(m):
            for j in range(i + 1, m):
                denom1 = onp.maximum(L[..., i] - L[..., j], 1e-20)
                denom2 = onp.maximum(L[..., i] + L[..., j], 1e-20)
                E[..., i, j] = 1.0 / denom1 / denom2
                E[..., j, i] = -E[..., i, j]
            E[..., i, i] = 0
        G1 = onp.matmul(1.0 / L[..., None] * dV, onp.swapaxes(V, -2, -1)) * L[..., None, :]
        G1 = G1 + onp.matmul(onp.swapaxes(dUT, -2, -1), UT)
        X = G1 * E
        G2 = onp.eye(m) + (X + onp.swapaxes(X, -2, -1)) * L[..., None, :] - 1.0 / L[..., None] * onp.matmul(dV, onp.swapaxes(V, -2, -1)) * onp.eye(m)
        dA = onp.matmul(UT, onp.matmul(G2, V) + 1.0 / L[..., None] * dV)
        return dA

    def check_svd(UT, L, V, data_np):
        shape = data_np.shape
        # check UT @ L @ V == A
        t = onp.matmul(UT * L[..., None, :], V)
        assert t.shape == data_np.shape
        assert_almost_equal(t, data_np, rtol=rtol, atol=atol)
        # check UT @ U == I
        I = onp.matmul(UT, onp.swapaxes(UT, -2, -1))
        I_np = onp.ones_like(UT) * onp.eye(shape[-2])
        assert I.shape == I_np.shape
        assert_almost_equal(I, I_np, rtol=rtol, atol=atol)
        # check U @ UT == I
        I = onp.matmul(onp.swapaxes(UT, -2, -1), UT)
        I_np = onp.ones_like(UT) * onp.eye(shape[-2])
        assert I.shape == I_np.shape
        assert_almost_equal(I, I_np, rtol=rtol, atol=atol)
        # check V @ VT == I
        I = onp.matmul(V, onp.swapaxes(V, -2, -1))
        I_np = onp.ones_like(UT) * onp.eye(shape[-2])
        assert I.shape == I_np.shape
        assert_almost_equal(I, I_np, rtol=rtol, atol=atol)

    rtol = atol = 0.01
    test_svd = TestSVD()
    if hybridize:
        test_svd.hybridize()
    data_np = onp.random.uniform(-10.0, 10.0, shape)
    data_np = onp.array(data_np, dtype=dtype)
    data = np.array(data_np, dtype=dtype)
    if effective_dtype(data) == onp.dtype(np.float16):
        pytest.skip()
    data.attach_grad()
    with mx.autograd.record():
        ret = test_svd(data)
    UT = ret[0].asnumpy()
    L = ret[1].asnumpy()
    V = ret[2].asnumpy()
    # check svd validity
    check_svd(UT, L, V, data_np)
    # check descending singular values
    s = [L[..., i] - L[..., i + 1] for i in range(L.shape[-1] - 1)]
    s = onp.array(s)
    assert (s >= -1e-5).all()
    if L.size > 0:
        assert (L[..., -1] >= -1e-5).all()
    # check backward
    mx.autograd.backward(ret)
    if ((s > 1e-5).all() and (L.size == 0 or (L > 1e-5).all())):
        backward_expected = get_grad(ret[0].asnumpy(), ret[1].asnumpy(), ret[2].asnumpy())
        assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)
    # Test imperative once again
    ret = np.linalg.svd(data)
    UT = ret[0].asnumpy()
    L = ret[1].asnumpy()
    V = ret[2].asnumpy()
    check_svd(UT, L, V, data_np)


@use_np
@pytest.mark.parametrize('shape', [
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
])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('hybridize', [False, True])
def test_np_linalg_svdvals(shape, dtype, hybridize):
    class TestSVD(HybridBlock):
        def __init__(self):
            super(TestSVD, self).__init__()

        def forward(self, data):
            return np.linalg.svdvals(data)

    rtol = atol = 0.01
    test_svd = TestSVD()
    if hybridize:
        test_svd.hybridize()
    data_np = onp.random.uniform(-10.0, 10.0, shape)
    data_np = onp.array(data_np, dtype=dtype)
    data = np.array(data_np, dtype=dtype)
    if effective_dtype(data) == onp.dtype(np.float16):
        pytest.skip()
    mx_out = test_svd(data)
    np_out = onp.linalg.svd(data, compute_uv=False)
    # check svdvals validity
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
    # Test imperative once again
    mx_out = np.linalg.svdvals(data)
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_linalg_qr():
    class TestQR(HybridBlock):
        def __init__(self):
            super(TestQR, self).__init__()

        def forward(self, data):
            return np.linalg.qr(data)

    def get_expected_grad(a, q, r, dq, dr):
        # for all input shapes (..., m, n)
        if 0 in r.shape:
            return r
        def _copyltu(M):
            eye = onp.array([onp.eye(M.shape[-1]) for i in range(M.shape[0])])
            lower = onp.tril(M) - eye * M
            lower_mask = onp.tril(onp.ones_like(M))
            ret = lower_mask * M + lower.swapaxes(-1, -2)
            return ret
        def _case_m_ge_n(a, q, r, dq, dr):
                dq_t = dq.swapaxes(-1, -2)
                dr_t = dr.swapaxes(-1, -2)
                r_inv = onp.linalg.inv(r)
                r_inv_t = r_inv.swapaxes(-1, -2)
                r_t = r.swapaxes(-1, -2)
                # Get M
                M = onp.matmul(r, dr_t) - onp.matmul(dq_t, q)
                da = onp.matmul(dq + onp.matmul(q, _copyltu(M)), r_inv_t)
                return da
        m, n = a.shape[-2], a.shape[-1]
        x = a[..., :, :m]
        x_shape = x.shape
        y = a[..., :, m:]
        y_shape = y.shape
        u = r[..., :, :m]
        v = r[..., :, m:]
        dv = dr[..., :, m:]
        du = dr[..., :, :m]
        q = q.reshape(-1, q.shape[-2], q.shape[-1])
        u = u.reshape(-1, u.shape[-2], u.shape[-1])
        dq = dq.reshape(-1, q.shape[-2], q.shape[-1])
        du = du.reshape(-1, du.shape[-2], du.shape[-1])
        if m >= n:
            dx = _case_m_ge_n(x, q, u, dq, du).reshape(x_shape)
            return dx
        else:
            dv = dv.reshape(-1, dv.shape[-2], dv.shape[-1])
            y = y.reshape(-1, y.shape[-2], y.shape[-1])
            dy = onp.matmul(q, dv).reshape(y_shape)
            dq_prime = dq + onp.matmul(y, dv.swapaxes(-1, -2))
            dx = _case_m_ge_n(x, q, u, dq_prime, du).reshape(x_shape)
            da = onp.concatenate([dx, dy], axis=-1)
            return da

    def well_conditioned_rectang_matrix_2D(shape, ran=(-1., 1.), max_cond=4):
        m, n = shape[-2], shape[-1]
        while 1:
            Q1, R1 = onp.linalg.qr(onp.random.uniform(ran[0], ran[1], (m, m)))
            D = onp.eye(m, n)
            Q2, R2 = onp.linalg.qr(onp.random.uniform(ran[0], ran[1], (n, n)))
            a = onp.matmul(onp.matmul(Q1, D), onp.swapaxes(Q2, -1, -2))
            if (onp.linalg.cond(a, 2) < max_cond):
                return a

    def well_conditioned_rectang_matrix_nD(shape, ran=(-1., 1.), max_cond=4):
        p = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
        return onp.array([well_conditioned_rectang_matrix_2D(shape, ran, max_cond) for i in range(p)]).reshape(shape)

    def check_qr(q, r, a_np):
        # check Q@R = A
        t = onp.matmul(q, r)
        assert t.shape == a_np.shape
        assert_almost_equal(t, a_np, rtol=rtol, atol=atol)
        # check QT@Q = I
        qT = onp.swapaxes(q, -2, -1)
        I = onp.matmul(qT, q)
        Ip = onp.eye(I.shape[-2])
        assert_almost_equal(I, Ip, atol=atol, rtol=rtol)
        # check original numpy
        try:
            q_expected, r_expected = onp.linalg.qr(a_np)
        except Exception as e:
            print("a_np", a_np)
            print("a shape:", a_np.shape)
            print(e)
        else:
            assert q.shape == q_expected.shape
            assert r.shape == r_expected.shape
            assert_almost_equal(q.asnumpy(), q_expected, rtol=rtol, atol=atol)
            assert_almost_equal(r.asnumpy(), r_expected, rtol=rtol, atol=atol)
    shapes = [
        (3, 5),
        (5, 3),
        (10, 10),
        (0, 1),
        (6, 5, 6),
        (6, 6, 5),
        (2, 3, 2, 3),
        (2, 3, 3, 2),
        (5, 0, 3, 3),
        (3, 3, 0, 0),
    ]
    dtypes = ['float64', 'float32']
    for hybridize, shape, dtype in itertools.product([False, True], shapes, dtypes):
        rtol = atol = 1e-2
        if dtype == 'float32':
            rtol = atol = 3e-2

        test_qr = TestQR()
        if hybridize:
            test_qr.hybridize()
        if 0 in shape:
            data_np = onp.ones(shape)
        else:
            data_np = well_conditioned_rectang_matrix_nD(shape, max_cond=4)

        data_np = onp.array(data_np, dtype=dtype)
        data = np.array(data_np, dtype=dtype)
        if effective_dtype(data) == onp.dtype(np.float16):
            print('Skipping test on this platform: {} has a float16 effective dtype'.format(dtype))
            pytest.skip()

        data.attach_grad()
        with mx.autograd.record():
            ret = test_qr(data)
        Q, R = ret[0], ret[1]
        check_qr(Q, R, data_np)

        if 0 not in R.shape:
            assert data.grad.shape == data_np.shape
            backward_expected = get_expected_grad(data_np, Q.asnumpy(), R.asnumpy(),
                                                  onp.ones(Q.shape), onp.ones(R.shape))
            mx.autograd.backward(ret)
            assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

        # check imperative once more; mode='reduced' is default
        # behavior and optional parameter in original numpy
        ret = np.linalg.qr(data, mode='reduced')
        Q, R = ret[0], ret[1]
        check_qr(Q, R, data_np)


@use_np
@pytest.mark.parametrize('shape', [
    (0, 0),
    (1, 1),
    (5, 5),
    (6, 6),
    (10, 10),
    (6, 6, 6),
    (1, 0, 0),
    (0, 1, 1),
    (2, 3, 4, 4),
])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('upper', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_linalg_cholesky(shape, dtype, upper, hybridize):
    class TestCholesky(HybridBlock):
        def __init__(self, upper=False):
            super(TestCholesky, self).__init__()
            self._upper = upper

        def forward(self, data):
            return np.linalg.cholesky(data, upper=self._upper)

    def get_grad(L, upper):
        # shape of m is [batch, n, n]
        if 0 in L.shape:
            return L
        
        if upper:
            L = onp.swapaxes(L, -1, -2)

        def copyltu(m):
            eye = onp.array([onp.eye(m.shape[-1]) for i in range(m.shape[0])])
            lower = onp.tril(m) - eye * m
            lower_mask = onp.tril(onp.ones_like(m))
            ret = lower_mask * m + lower.swapaxes(-1, -2)
            return ret

        shape = L.shape
        L = L.reshape(-1, shape[-2], shape[-1])
        dL = onp.ones_like(L)
        L_inv = onp.linalg.inv(L)
        L_inv_T = L_inv.swapaxes(-1, -2)
        L_T = L.swapaxes(-1, -2)
        sym_L_inv = 0.5 * (L_inv + L_inv_T)
        dA = 0.5 * onp.matmul(onp.matmul(L_inv_T, copyltu(onp.matmul(L_T, dL))), L_inv)
        return dA.reshape(shape)

    def check_cholesky(L, data_np, upper):
        assert L.shape == data_np.shape
        # catch error if numpy throws rank < 2
        try:
            if upper:
                L_expected = onp.swapaxes(onp.linalg.cholesky(data_np), -1, -2)
            else:
                L_expected = onp.linalg.cholesky(data_np)
        except Exception as e:
            print(data_np)
            print(data_np.shape)
            print(e)
        else:
            assert L.shape == L_expected.shape
            assert_almost_equal(L.asnumpy(), L_expected, rtol=rtol, atol=atol)

    def newSymmetricPositiveDefineMatrix_2D(shape, ran=(0., 10.), max_cond=4):
        while 1:
            D = onp.diag(onp.random.uniform(ran[0], ran[1], shape[-1]))
            I = onp.eye(shape[-1]).reshape(shape)
            v = onp.random.uniform(-1., 1., shape[-1]).reshape(shape[:-1] + (1,))
            v = v / onp.linalg.norm(v, axis=-2, keepdims=True)
            v_T = onp.swapaxes(v, -1, -2)
            U = I - 2 * onp.matmul(v, v_T)
            a = onp.matmul(onp.matmul(U, D), onp.swapaxes(U, -1, -2))
            if (onp.linalg.cond(a, 2) < max_cond):
                return a

    def newSymmetricPositiveDefineMatrix_nD(shape, ran=(0., 10.), max_cond=4):
        n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
        return onp.array([newSymmetricPositiveDefineMatrix_2D(shape[-2:], ran, max_cond) for i in range(n)]).reshape(shape)


    rtol = 1e-3
    atol = 1e-5
    if dtype == 'float32':
        rtol = 1e-2
        atol = 1e-4

    test_cholesky = TestCholesky(upper)
    if hybridize:
        test_cholesky.hybridize()

    # Numerical issue:
    # When backpropagating through Cholesky decomposition, we need to compute the inverse
    # of L according to dA = 0.5 * L**(-T) * copyLTU(L**T * dL) * L**(-1) where A = LL^T.
    # The inverse is calculated by "trsm" method in CBLAS. When the data type is float32,
    # this causes numerical instability. It happens when the matrix is ill-conditioned.
    # In this example, the issue occurs frequently if the symmetric positive definite input
    # matrix A is constructed by A = LL^T + \epsilon * I. A proper way of testing such
    # operators involving numerically unstable operations is to use well-conditioned random
    # matrices as input. Here we test Cholesky decomposition for FP32 and FP64 separately.
    # See rocBLAS:
    # https://github.com/ROCmSoftwarePlatform/rocBLAS/wiki/9.Numerical-Stability-in-TRSM

    # generate symmetric PD matrices
    if 0 in shape:
        data_np = np.ones(shape)
    else:
        data_np = newSymmetricPositiveDefineMatrix_nD(shape)

    # When dtype is np.FP32, truncation from FP64 to FP32 could also be a source of
    # instability since the ground-truth gradient is computed using FP64 data.
    data = np.array(data_np, dtype=dtype)
    data.attach_grad()
    with mx.autograd.record():
        L = test_cholesky(data)

    # check cholesky validity
    check_cholesky(L, data_np, upper)
    # check backward. backward does not support empty input
    if 0 not in L.shape:
        mx.autograd.backward(L)
        backward_expected = get_grad(L.asnumpy(), upper)
        assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)
    # check imperative once again
    L = np.linalg.cholesky(data, upper=upper)
    check_cholesky(L, data_np, upper)


@use_np
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('shape', [
    (0, 0),
    (4, 4),
    (2, 2),
    (1, 1),
    (2, 1, 1),
    (0, 1, 1),
    (6, 1, 1),
    (2, 3, 3, 3),
    (4, 2, 1, 1),
    (0, 5, 3, 3),
    (5, 0, 0, 0),
    (3, 3, 0, 0),
    (3, 5, 5),
])
@retry(3)
def test_np_linalg_inv(hybridize, dtype, shape):
    class TestInverse(HybridBlock):
        def __init__(self):
            super(TestInverse, self).__init__()

        def forward(self, data):
            return np.linalg.inv(data)

    def get_grad(A):
        if 0 in A.shape:
            return A

        dA = onp.ones_like(A)
        A_inv = onp.linalg.inv(A)
        dA_inv = -onp.matmul(onp.matmul(A_inv, dA), A_inv)
        return onp.swapaxes(dA_inv, -1, -2)

    def check_inv(A_inv, data_np):
        assert A_inv.shape == data_np.shape
        # catch error if numpy throws rank < 2
        try:
            A_expected = onp.linalg.inv(data_np)
        except Exception as e:
            print(data_np)
            print(data_np.shape)
            print(e)
        else:
            assert A_inv.shape == A_expected.shape
            assert_almost_equal(A_inv.asnumpy(), A_expected, rtol=rtol, atol=atol)

    atol = rtol = 1e-2

    test_inv = TestInverse()
    if hybridize:
        test_inv.hybridize()
    # generate well-conditioned matrices with small eigenvalues
    if 0 in shape:
        data_np = onp.ones(shape)
    else:
        n = int(np.prod(np.array(shape[:-2]))) if len(shape) > 2 else 1
        # eigenvalues
        D = onp.array([onp.diag(onp.random.uniform(-10., 10., shape[-1])) \
                         for i in range(n)]).reshape(shape)
        # orthogonal matrix through householder transformation
        I = onp.array([onp.eye(shape[-1]) for i in range(n)]).reshape(shape)
        v = onp.random.uniform(-10, 10,
                int(np.prod(np.array(shape[:-1])))).reshape(shape[:-1] + (1,))
        v = v / onp.linalg.norm(v, axis=-2, keepdims=True)
        v_T = onp.swapaxes(v, -1, -2)
        U = I - 2 * onp.matmul(v, v_T)
        data_np = onp.matmul(onp.matmul(U, D), onp.swapaxes(U, -1, -2))
    data = np.array(data_np, dtype=dtype)
    data.attach_grad()
    with mx.autograd.record():
        A_inv = test_inv(data)

    # check cholesky validity
    check_inv(A_inv, data_np)
    # check backward. backward does not support empty input
    mx.autograd.backward(A_inv)
    backward_expected = get_grad(data.asnumpy())
    assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)
    # check imperative once again
    A_inv = np.linalg.inv(data)
    check_inv(A_inv, data_np)


@use_np
def test_np_linalg_solve():
    class TestSolve(HybridBlock):
        def __init__(self):
            super(TestSolve, self).__init__()

        def forward(self, a, b):
            return np.linalg.solve(a, b)

    def check_solve(x, a_np, b_np):
        try:
            x_expected = onp.linalg.solve(a_np, b_np)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print("b", b_np)
            print("b shape:", b_np.shape)
            print(e)
        else:
            assert x.shape == x_expected.shape
            assert_almost_equal(x, x_expected)

    def newInvertibleMatrix_2D(shape, max_cond=4):
        while 1:
            # generate well-conditioned matrices with small eigenvalues
            D = onp.diag(onp.random.uniform(-1.0, 1.0, shape[-1]))
            I = onp.eye(shape[-1]).reshape(shape)
            v = onp.random.uniform(-10., 10., shape[-1]).reshape(shape[:-1] + (1,))
            v = v / onp.linalg.norm(v, axis=-2, keepdims=True)
            v_T = onp.swapaxes(v, -1, -2)
            U = I - 2 * onp.matmul(v, v_T)
            a = onp.matmul(U, D)
            if (onp.linalg.cond(a, 2) < max_cond):
                return a

    def newInvertibleMatrix_nD(shape, max_cond=4):
        n = int(np.prod(np.array(shape[:-2]))) if len(shape) > 2 else 1
        return onp.array([newInvertibleMatrix_2D(shape[-2:]) for i in range(n)]).reshape(shape)

    def get_grad_b(A, X):
        dX = onp.ones_like(X)
        A_inv = onp.linalg.inv(A)
        A_inv_trans = onp.swapaxes(A_inv, -1, -2)
        return onp.matmul(A_inv_trans, dX)

    shapes = [
        (0, 0),
        (1, 1),
        (3, 3),
        (4, 4),
        (3, 2, 2),
        (1, 0, 0),
        (0, 1, 1),
        (0, 5, 3, 3),
        (5, 0, 0, 0),
        (2, 2, 5, 5)
    ]
    nrhs = (-1, 0, 1, 2, 3)
    dtypes = ['float32', 'float64']
    for hybridize, shape, dtype, nrh in itertools.product([False, True], shapes, dtypes, nrhs):
        test_solve = TestSolve()
        if hybridize:
            test_solve.hybridize()

        if 0 in shape:
            a = onp.ones(shape)
            b = onp.ones(shape)
        else:
            shape_a = shape
            shape_b = list(shape_a)
            if nrh == -1:
                shape_b[-1] = 1
            else :
                shape_b[-1] = nrh
            a = newInvertibleMatrix_nD(shape_a)
            x = onp.random.randn(*shape_b)
            b = onp.matmul(a, x)
        a = np.array(a, dtype=dtype)
        b = np.array(b, dtype=dtype)
        a.attach_grad()
        b.attach_grad()
        with mx.autograd.record():
            mx_out = test_solve(a, b)
        # check solve validity
        assert mx_out.shape == b.shape
        check_solve(mx_out, a, b)

        # check backward. backward does not support empty input
        if 0 not in mx_out.shape:
            if nrh != -1:
                mx.autograd.backward(mx_out)
                b_backward_expected = get_grad_b(a.asnumpy(), mx_out.asnumpy())
                a_backward_expected = -onp.matmul(b_backward_expected, onp.swapaxes(mx_out, -1, -2).asnumpy())
                assert_almost_equal(a.grad, a_backward_expected)
                assert_almost_equal(b.grad, b_backward_expected)

        # check imperative once again
        mx_out = np.linalg.solve(a, b)
        check_solve(mx_out, a, b)


def test_np_linalg_tensorinv():
    class TestTensorinv(HybridBlock):
        def __init__(self, ind=2):
            super(TestTensorinv, self).__init__()
            self._ind = ind

        def forward(self, a):
            return np.linalg.tensorinv(a, ind=self._ind)

    def check_tensorinv(inv_a, a_np, ind):
        try:
            inv_a_expected = onp.linalg.tensorinv(a_np, ind=ind)
        except Exception as e:
            print(a_np)
            print(a_np.shape)
            print(e)
        else:
            assert inv_a.shape == inv_a_expected.shape
            assert_almost_equal(inv_a, inv_a_expected)

    def newInvertibleMatrix_2D(shape, max_cond=4):
        while 1:
            # generate well-conditioned matrices with small eigenvalues
            D = onp.diag(onp.random.uniform(-1.0, 1.0, shape[-1]))
            I = onp.eye(shape[-1]).reshape(shape)
            v = onp.random.uniform(-10., 10., shape[-1]).reshape(shape[:-1] + (1,))
            v = v / onp.linalg.norm(v, axis=-2, keepdims=True)
            v_T = onp.swapaxes(v, -1, -2)
            U = I - 2 * onp.matmul(v, v_T)
            a = onp.matmul(U, D)
            if (onp.linalg.cond(a, 2) < max_cond):
                return a

    def get_grad_A(A, ind):
        inv_A = onp.linalg.tensorinv(A, ind)
        d_inv_A = onp.ones_like(inv_A)
        axes1 = len(A.shape) - ind
        axes2 = ind
        inv_A_trans_axes = tuple(onp.arange(len(A.shape)))[axes1:] + tuple(onp.arange(len(A.shape)))[:axes1]
        inv_A_trans = onp.transpose(inv_A, inv_A_trans_axes)
        temp_tensor = -onp.tensordot(inv_A_trans, d_inv_A, axes = axes1)
        return onp.tensordot(temp_tensor, inv_A_trans, axes = axes2)

    shapes = [
        (1, 1, 1),
        (1, 2, 2),
        (1, 6, 2, 3),
        (1, 10, 2, 5),
        (1, 12, 3, 4),
        (2, 1, 1),
        (2, 1, 1, 1),
        (2, 2, 5, 5, 2),
        (2, 1, 6, 3, 2),
        (2, 1, 8, 4, 2),
        (2, 12, 1, 3, 4, 1),
        (3, 1, 1, 1),
        (3, 2, 3, 1, 6),
        (3, 3, 2, 1, 2, 3, 1)
    ]
    dtypes = ['float32', 'float64']
    for hybridize, shape, dtype, in itertools.product([False, True], shapes, dtypes):
        ind = shape[0]
        test_tensorinv = TestTensorinv(ind=ind)
        if hybridize:
            test_tensorinv.hybridize()

        prod_front = 1
        prod_back = 1
        for k in shape[1:ind + 1]:
            prod_front *= k
        for k in shape[1 + ind:]:
            prod_back *= k
        a_shape = (prod_back, prod_front)
        a = newInvertibleMatrix_2D(a_shape)
        a_shape = shape[1:]
        inv_a_shape = shape[(1 + ind):] + shape[1:(ind + 1)]
        a = np.array(a.reshape(a_shape), dtype=dtype)
        a.attach_grad()
        with mx.autograd.record():
            mx_out = test_tensorinv(a)
        # check tensorinv validity
        assert mx_out.shape == inv_a_shape
        check_tensorinv(mx_out, a, ind)

        # check tensorinv backward
        if 0 not in mx_out.shape:
            mx.autograd.backward(mx_out)
            grad_A_expected = get_grad_A(a.asnumpy(), ind)
            assert_almost_equal(a.grad, grad_A_expected)

    # check imperative once again
    mx_out = np.linalg.tensorinv(a, ind)
    check_tensorinv(mx_out, a, ind)


@use_np
def test_np_linalg_tensorsolve():
    class TestTensorsolve(HybridBlock):
        def __init__(self, axes):
            super(TestTensorsolve, self).__init__()
            self._axes = axes

        def forward(self, a, b):
            return np.linalg.tensorsolve(a, b, axes=self._axes)

    def get_tensorsolve_backward(a_np, b_np, mx_out_np, a_axes, a_origin_axes, a_trans_shape):
        if (a_np.ndim == 0 or b_np.ndim == 0) or (a_np.ndim == b_np.ndim):
            a_shape = a_np.shape
            b_shape = b_np.shape
            a_np = a_np.reshape((1, 1))
            b_np = b_np.reshape((1,))
            mx_out_np = mx_out_np.reshape((1,))
            dx = onp.ones_like(mx_out_np)
            inv_a_temp_np = onp.linalg.inv(a_np)
            grad_b = inv_a_temp_np[0][0] * dx[0]
            grad_a = -grad_b * mx_out_np[0]
            return grad_a.reshape(a_shape), grad_b.reshape(b_shape)
        else:
            dx = onp.ones_like(mx_out_np)
            a_np = a_np.transpose(a_axes)
            ind = a_np.ndim - mx_out_np.ndim
            tensorinv_a_np = onp.linalg.tensorinv(a_np, ind=ind)
            a_trans_axes = list(range(a_np.ndim))[a_np.ndim - ind:] + list(range(a_np.ndim))[:a_np.ndim - ind]
            trans_tensorinv_a_np = tensorinv_a_np.transpose(a_trans_axes)
            grad_b = onp.tensordot(trans_tensorinv_a_np, dx, axes=dx.ndim)
            grad_a = onp.tensordot(grad_b, mx_out_np, axes=0)
            grad_a = grad_a.transpose(a_origin_axes)
            return -grad_a, grad_b.reshape(b_np.shape)

    def check_tensorsolve(x, a_np, b_np, axes):
        try:
            x_expected = onp.linalg.tensorsolve(a_np, b_np, axes=axes)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print("b", b_np)
            print("b shape:", b_np.shape)
            print(e)
        else:
            assert x.shape == x_expected.shape
            assert_almost_equal(x, x_expected)

    def shapeInfer(a_shape, b_shape, axes=None):
        # b_shape - Right-hand tensor shape, which can be of any shape.
        a_ndim = len(a_shape)
        b_ndim = len(b_shape)
        a_trans_shape = list(a_shape)
        a_axes = list(range(0, a_ndim))
        if axes is not None:
            for k in axes:
                a_axes.remove(k)
                a_axes.insert(a_ndim, k)
            for k in range(a_ndim):
                a_trans_shape[k] = a_shape[a_axes[k]]
        x_shape = a_trans_shape[-(a_ndim - b_ndim):]
        prod = 1
        for k in x_shape:
            prod *= k
        if prod * prod != onp.prod(a_shape):
            raise ValueError("a is not square")
        if prod != onp.prod(b_shape):
            raise ValueError("a's shape and b's shape dismatch")
        return a_axes, (prod, prod), tuple(a_trans_shape), tuple(x_shape)

    def newInvertibleMatrix_2D(shape, max_cond=4):
        while 1:
            # generate well-conditioned matrices with small eigenvalues
            D = onp.diag(onp.random.uniform(-1.0, 1.0, shape[-1]))
            I = onp.eye(shape[-1]).reshape(shape)
            v = onp.random.uniform(-1., 1., shape[-1]).reshape(shape[:-1] + (1,))
            v = v / onp.linalg.norm(v, axis=-2, keepdims=True)
            v_T = onp.swapaxes(v, -1, -2)
            U = I - 2 * onp.matmul(v, v_T)
            a = onp.matmul(U, D)
            if (onp.linalg.cond(a, 2) < max_cond):
                return a

    shapes = [
        # a_shape.ndim <= 6,
        # (a_shape, b_shape, axes)
        ((), (), None),                     # a.ndim == 0, b.ndim == 0, with axes must be None
        ((), (1, 1, 1), None),              # a.ndim == 0, b.ndim != 0, with axes must be None
        ((1, 1, 1), (), None),              # a.ndim != 0, b.ndim == 0, with axes == None
        ((1, 1, 1), (), (0, 1, 2)),         # a.ndim != 0, b.ndim == 0, with axes != None
        ((1, 1, 1), (1, 1, 1), None),       # a.ndim != 0, b.ndim != 0, a.ndim == b.ndim with axes == None
        ((1, 1, 1), (1, 1, 1), (2, 0, 1)),  # a.ndim != 0, b.ndim != 0, a.ndim == b.ndim with axes != None
        ((1, 1), (1,), None),               # a.ndim != 0, b.ndim != 0, a.ndim > b.ndim
        ((1, 1), (1, 1, 1, 1, 1), None),    # a.ndim != 0, b.ndim != 0, a.ndim < b.ndim - a.ndim
        ((4, 4), (4,), None),
        ((6, 2, 3), (6,), None),
        ((2, 3, 6), (6,), (0, 1)),
        ((3, 4, 2, 3, 2), (3, 4), None),
        ((2, 1, 4, 2, 4), (2, 4), (0, 1, 2)),
        ((2, 3, 3, 4, 2), (3, 4), (0, 2, 4)),
        ((1, 3, 3, 4, 4), (1, 3, 4), (1, 3)),
        ((1, 12, 4, 1, 3), (1, 2, 1, 2, 1, 3, 1), None),
        ((1, 4, 1, 12, 3), (1, 2, 1, 2, 1, 3, 1), (1, 2, 4)),
    ]
    dtypes = ['float32', 'float64']
    for hybridize in [True, False]:
        for dtype in dtypes:
            for a_shape, b_shape, axes in shapes:
                test_tensorsolve = TestTensorsolve(axes)
                if hybridize:
                    test_tensorsolve.hybridize()

                a_axes, mat_shape, a_trans_shape, x_shape = shapeInfer(a_shape, b_shape, axes)
                # generate coefficient tensor a and right side tensor b
                if (len(a_shape) == 0 or len(b_shape) == 0) or (len(a_shape) == len(b_shape)):
                    a_np = onp.asarray(1).astype(dtype).reshape(a_shape)
                    b_np = onp.asarray(2).astype(dtype).reshape(b_shape)
                else:
                    a_np = newInvertibleMatrix_2D(mat_shape, max_cond=3).reshape(a_trans_shape)
                    x_np = onp.random.randn(*x_shape)
                    b_np = onp.tensordot(a_np, x_np, axes=len(x_shape))

                # resume original shape of tensor a
                a_origin_axes = list(range(a_np.ndim))
                if axes is not None:
                    for k in range(a_np.ndim):
                        a_origin_axes[a_axes[k]] = k
                a_np = a_np.transpose(a_origin_axes)
                a = np.array(a_np, dtype=dtype).reshape(a_shape)
                b = np.array(b_np, dtype=dtype).reshape(b_shape)
                a.attach_grad()
                b.attach_grad()

                with mx.autograd.record():
                    mx_out = test_tensorsolve(a, b)
                # check tensorsolve validity
                assert mx_out.shape == x_shape
                check_tensorsolve(mx_out, a.asnumpy(), b.asnumpy(), axes)

                # check backward
                if len(a_shape) != 0 and len(b_shape) != 0:
                    mx.autograd.backward(mx_out)
                    grad_a_expected, grad_b_expected = get_tensorsolve_backward(
                        a.asnumpy(), b.asnumpy(), mx_out.asnumpy(), a_axes, a_origin_axes, a_trans_shape)
                    assert_almost_equal(a.grad, grad_a_expected)
                    assert_almost_equal(b.grad, grad_b_expected)

                # check imperative once again
                mx_out = test_tensorsolve(a, b)
                check_tensorsolve(mx_out, a.asnumpy(), b.asnumpy(), axes)


@use_np
def test_np_linalg_lstsq():
    class TestLstsq(HybridBlock):
        def __init__(self, rcond):
            super(TestLstsq, self).__init__()
            self._rcond = rcond

        def forward(self, a, b, rcond='warn'):
            return np.linalg.lstsq(a, b, rcond=self._rcond)

    def check_lstsq(a_np, b_np, rcond_np, x, residuals, rank, s):
        try:
            if rcond_np == 'warn':
                rcond_np = -1
            x_expected, residuals_expected, rank_expected, s_expected = onp.linalg.lstsq(a_np, b_np, rcond_np)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print("b:", b_np)
            print("b shape:", b_np.shape)
            print(e)
        else:
            assert x.shape == x_expected.shape
            assert residuals.shape == residuals_expected.shape
            assert rank.shape == rank_expected.shape
            assert s.shape == s_expected.shape
            assert_almost_equal(x.asnumpy(), x_expected, rtol=rtol, atol=atol)
            assert_almost_equal(residuals.asnumpy(), residuals_expected, rtol=rtol, atol=atol)
            assert_almost_equal(rank.asnumpy(), rank_expected, rtol=rtol, atol=atol)
            assert_almost_equal(s.asnumpy(), s_expected, rtol=rtol, atol=atol)

    shapes = [
        ((4, 0), (4,)),   # ncol == 0
        ((4, 0), (4, 2)), # ncol == 0
        ((0, 2), (0,)),   # nrow == 0
        ((0, 2), (0, 4)), # nrow == 0
        ((4, 2), (4, 0)), # nrhs == 0
        ((4, 4), (4, 0)), # nrhs == 0
        ((4, 6), (4, 0)), # nrhs == 0
        ((0, 0), (0, 4)), # nrow == 0, ncol == 0
        ((0, 2), (0, 0)), # nrow == 0, nrhs == 0
        ((4, 0), (4, 0)), # ncol == 0, nrhs == 0
        ((0, 0), (0,)),   # nrow == 0, ncol == 0, nrhs = none
        ((0, 0), (0, 0)), # nrow == 0, ncol == 0, nrhs = 0
        ((2, 1), (2,)),
        ((4, 1), (4,)),
        ((4, 2), (4,)),
        ((4, 4), (4,)),
        ((1, 4), (1, 4)),
        ((4, 2), (4, 1)),
        ((4, 2), (4, 3)),
        ((4, 4), (4, 3)),
        ((4, 6), (4, 3)),
    ]
    rconds = [None, "random", "warn"]
    dtypes = ['float32', 'float64']
    for rcond, hybridize in itertools.product(rconds, [True, False]):
        for dtype in dtypes:
            for a_shape, b_shape in shapes:
                rtol = 1e-2 if dtype == 'float32' else 1e-3
                atol = 1e-4 if dtype == 'float32' else 1e-5
                if rcond == "random":
                    rcond = onp.random.uniform(100, 200)
                test_lstsq = TestLstsq(rcond)
                if hybridize:
                    test_lstsq.hybridize()
                a_np = onp.random.uniform(-10.0, 10.0, a_shape)
                b_np = onp.random.uniform(-10.0, 10.0, b_shape)
                a = np.array(a_np, dtype=dtype)
                b = np.array(b_np, dtype=dtype)
                x, residuals, rank, s = test_lstsq(a, b)
                # check lstsq validity
                check_lstsq(a_np, b_np, rcond, x, residuals, rank, s)


@use_np
def test_np_linalg_matrix_rank():
    class TestMatrixRank(HybridBlock):
        def __init__(self, hermitian):
            super(TestMatrixRank, self).__init__()
            self._hermitian = hermitian

        def forward(self, M, tol=None):
            return np.linalg.matrix_rank(M, tol, hermitian=self._hermitian)

    def check_matrix_rank(rank, a_np, tol, hermitian):
        try:
            rank_expected = onp.linalg.matrix_rank(a_np, tol=tol, hermitian=hermitian)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print(e)
        else:
            if a_np.ndim < 2:
                assert rank.shape == onp.asarray(rank_expected).shape
            else:
                assert rank.shape == rank_expected.shape
            assert_almost_equal(rank.asnumpy(), rank_expected, rtol=rtol, atol=atol)

    shapes = [
        ((), ()),
        ((1,), (1,)),
        ((3,), (1,)),
        ((1, 1), ()),
        ((1, 1), (1,)),
        ((3, 3), (1,)),
        ((3, 4), (1,)),
        ((4, 3), ()),
        ((4, 3), (1,)),
        ((4, 3), (2,)),
        ((4, 3), (2, 3,)),
        ((2, 1, 1), ()),
        ((2, 1, 1), (1,)),
        ((2, 3, 3), (2,)),
        ((2, 3, 4), (1,)),
        ((2, 4, 3), (2,)),
        ((2, 3, 1, 1), ()),
        ((2, 3, 1, 1), (1, 1)),
        ((2, 3, 1, 1), (2, 1)),
        ((2, 3, 4, 4), (1, 3)),
        ((2, 3, 4, 5), (2, 1)),
        ((2, 3, 5, 4), (1, 3)),
        ((2, 3, 1, 1), (2, 3)),
        ((2, 3, 4, 4), (2, 3)),
        ((2, 3, 4, 5), (2, 3)),
        ((2, 3, 5, 4), (2, 3)),
    ]
    dtypes = ['float32', 'float64']
    for dtype in dtypes:
        for a_shape, tol_shape in shapes:
            for tol_is_none, hybridize in itertools.product([True, False], [True, False]):
                rtol = 1e-3
                atol = 1e-5
                test_matrix_rank = TestMatrixRank(hermitian=False)
                if hybridize:
                    test_matrix_rank.hybridize()

                a_np = onp.asarray(onp.random.uniform(-10., 10., a_shape))
                a = np.array(a_np, dtype=dtype)
                if tol_is_none:
                    rank = test_matrix_rank(a)
                    # check matrix_rank validity
                    check_matrix_rank(rank, a.asnumpy(), tol=None, hermitian=False)
                else:
                    tol_np = onp.random.uniform(10., 20., tol_shape)
                    tol = np.array(tol_np, dtype=dtype)
                    rank = test_matrix_rank(a, tol)
                    # check matrix_rank validity
                    check_matrix_rank(rank, a.asnumpy(), tol.asnumpy(), hermitian=False)


@use_np
@pytest.mark.parametrize('shape', [
    (),
    (1,),
    (0, 1, 2),
    (0, 1, 2),
    (0, 1, 2),
    (4, 5, 6, 7),
    (4, 5, 6, 7),
    (4, 5, 6, 7),
])
def test_np_linalg_matrix_transpose(shape):
    class TestMatTranspose(HybridBlock):
        def __init__(self):
            super(TestMatTranspose, self).__init__()

        def forward(self, x):
            return np.linalg.matrix_transpose(x)

    data_np = onp.random.uniform(size=shape)
    data_mx = np.array(data_np, dtype=data_np.dtype)
    if data_mx.ndim < 2:
        assertRaises(ValueError, np.linalg.matrix_transpose, data_mx)
        return
    ret_np = onp.swapaxes(data_np, -1, -2)
    ret_mx = np.linalg.matrix_transpose(data_mx)
    assert same(ret_mx.asnumpy(), ret_np)

    net = TestMatTranspose()
    for hybrid in [False, True]:
        if hybrid:
            net.hybridize()
        ret_mx = net(data_mx)
        assert same(ret_mx.asnumpy(), ret_np)
    
    assert same(data_mx.mT.asnumpy(), ret_np)


@use_np
def test_np_linalg_pinv():
    class TestPinv(HybridBlock):
        def __init__(self, hermitian):
            super(TestPinv, self).__init__()
            self._hermitian = hermitian

        def forward(self, a, rcond=1e-15):
            return np.linalg.pinv(a, rcond, hermitian=self._hermitian)

    def check_pinv(x, a_np, rcond_np, hermitian, use_rcond):
        try:
            if use_rcond:
                x_expected = onp.linalg.pinv(a_np, rcond_np, hermitian=hermitian)
            else:
                x_expected = onp.linalg.pinv(a_np, hermitian=hermitian)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            if use_rcond:
                print("rcond_np", rcond_np)
                print("b rcond_np:", rcond_np.shape)
            print(e)
        else:
            assert x.shape == x_expected.shape
            assert_almost_equal(x.asnumpy(), x_expected, rtol=rtol, atol=atol)

    shapes = [
        ((1, 1), ()),
        ((5, 5), ()),
        ((5, 6), ()),
        ((6, 5), ()),
        ((2, 3, 3), (1,)),
        ((2, 3, 3), (2,)),
        ((2, 3, 4), (2,)),
        ((2, 4, 3), (1,)),
        ((4, 5, 6), ()),
        ((4, 5, 6), (1,)),
        ((4, 6, 5), (4,)),
        ((2, 2, 4, 3), (1,)),
        ((2, 2, 4, 3), (2,)),
        ((2, 2, 4, 3), (1, 1)),
        ((2, 2, 4, 3), (1, 2)),
        ((2, 2, 4, 3), (2, 1)),
        ((2, 2, 4, 3), (2, 2)),
        ((2, 2, 3, 4), (1,)),
        ((2, 2, 3, 4), (2,)),
        ((2, 2, 3, 4), (1, 1)),
        ((2, 2, 3, 4), (1, 2)),
        ((2, 2, 3, 4), (2, 1)),
        ((2, 2, 3, 4), (2, 2)),
    ]
    dtypes = ['float32', 'float64']
    for dtype in dtypes:
        for a_shape, rcond_shape in shapes:
            for use_rcond, hybridize in itertools.product([True, False], [True, False]):
                rtol = 1e-2 if dtype == 'float32' else 1e-3
                atol = 1e-4 if dtype == 'float32' else 1e-5
                hermitian = False
                test_pinv = TestPinv(hermitian)
                if hybridize:
                    test_pinv.hybridize()

                a_np = onp.random.uniform(-10.0, 10.0, a_shape)
                a_np = onp.array(a_np, dtype=dtype)
                rcond_np = onp.random.uniform(0., 0.1, rcond_shape)
                rcond_np = onp.array(rcond_np, dtype=dtype)
                a = np.array(a_np, dtype=dtype)
                rcond = np.array(rcond_np, dtype=dtype)
                if use_rcond:
                    mx_out = test_pinv(a, rcond)
                else:
                    mx_out = test_pinv(a)

                # check tensorsolve validity
                check_pinv(mx_out, a.asnumpy(), rcond.asnumpy(), hermitian, use_rcond)


@use_np
def test_np_linalg_eigvals():
    class TestEigvals(HybridBlock):
        def __init__(self):
            super(TestEigvals, self).__init__()

        def forward(self, a):
            return np.linalg.eigvals(a)

    def check_eigvals(x, a_np):
        try:
            x_expected = onp.linalg.eigvals(a_np)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print(e)
        else:
            assert x.shape == x_expected.shape
            if 0 not in x.shape:
                n = int(onp.prod(x.shape[:-1])) if len(shape) > 1 else 1
                x = x.reshape(n, -1)
                x_expected = x_expected.reshape(n, -1)
                for i in range(n):
                    x1 = onp.sort(x[i].asnumpy())
                    x2 = onp.sort(x_expected[i])
                    assert_almost_equal(x1, x2, rtol=rtol, atol=atol)

    shapes = [
        (0, 0),
        (1, 1),
        (3, 3),
        (5, 5),
        (1, 0, 0),
        (0, 4, 4),
        (1, 4, 4),
        (2, 4, 4),
        (5, 5, 5),
        (1, 1, 4, 4),
        (2, 3, 4, 4)
    ]
    dtypes = ['float32', 'float64', 'uint8', 'int8', 'int32', 'int64']
    UPLOs = ['L', 'U']
    for hybridize in [True, False]:
        for shape, dtype in itertools.product(shapes, dtypes):
            rtol = 1e-2 if dtype == 'float32' else 1e-3
            atol = 1e-4 if dtype == 'float32' else 1e-5
            test_eigvals = TestEigvals()
            if hybridize:
                test_eigvals.hybridize()
            if 0 in shape:
                a_np = onp.ones(shape)
            else:
                if dtype == 'uint8' or dtype == 'int8' or dtype == 'int32' or dtype == 'int64':
                    n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
                    a_np = onp.array([onp.diag(onp.random.randint(1, 10, size=shape[-1])) for i in range(n)]).reshape(shape)
                else:
                    a_np = new_matrix_with_real_eigvals_nd(shape)
            a = np.array(a_np, dtype=dtype)
            # check eigvals validity
            mx_out = test_eigvals(a)
            check_eigvals(mx_out, a.asnumpy())

            # check imperative once again
            mx_out = test_eigvals(a)
            check_eigvals(mx_out, a.asnumpy())


@use_np
def test_np_linalg_eigvalsh():
    class TestEigvalsh(HybridBlock):
        def __init__(self, upper):
            super(TestEigvalsh, self).__init__()
            self._upper = upper

        def forward(self, a):
            return np.linalg.eigvalsh(a, upper=self._upper)

    def check_eigvalsh(w, a_np, upper):
        try:
            w_expected = onp.linalg.eigvalsh(a_np, upper)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print(e)
        else:
            assert w.shape == w_expected.shape
            assert_almost_equal(w, w_expected, rtol=rtol, atol=atol)

    def new_matrix_from_sym_matrix_nd(sym_a, upper):
        shape = sym_a.shape
        if 0 in shape:
            return sym_a
        n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
        a = sym_a.reshape(n, shape[-2], shape[-1])
        for idx in range(n):
            for i in range(shape[-2]):
                for j in range(shape[-1]):
                    if ((upper == True and i > j) or (upper == False and i < j)):
                        a[idx][i][j] = onp.random.uniform(-10., 10.)
        return a.reshape(shape)

    shapes = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (1, 0, 0),
        (0, 4, 4),
        (1, 4, 4),
        (2, 4, 4),
        (5, 5, 5),
        (1, 1, 4, 4),
        (2, 3, 4, 4)
    ]
    dtypes = ['float32', 'float64', 'uint8', 'int8', 'int32', 'int64']
    uppers = [True, False]
    for hybridize in [True, False]:
        for shape, dtype, upper in itertools.product(shapes, dtypes, uppers):
            rtol = 1e-2 if dtype == 'float32' else 1e-3
            atol = 1e-4 if dtype == 'float32' else 1e-5
            test_eigvalsh = TestEigvalsh(upper)
            if hybridize:
                test_eigvalsh.hybridize()
            if 0 in shape:
                a_np = onp.ones(shape)
            else:
                if dtype == 'uint8' or dtype == 'int8' or dtype == 'int32' or dtype == 'int64':
                    n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
                    a_np = onp.array([onp.diag(onp.random.randint(1, 10, size=shape[-1])) for i in range(n)], dtype=dtype).reshape(shape)
                else:
                    a_np = new_sym_matrix_with_real_eigvals_nd(shape)
                    a_np = new_matrix_from_sym_matrix_nd(a_np, upper)
            a = np.array(a_np, dtype=dtype)
            # check eigvalsh validity
            mx_out = test_eigvalsh(a)
            check_eigvalsh(mx_out, a.asnumpy(), upper)

            # check imperative once again
            mx_out = test_eigvalsh(a)
            check_eigvalsh(mx_out, a.asnumpy(), upper)


@use_np
def test_np_linalg_eig():
    class TestEig(HybridBlock):
        def __init__(self):
            super(TestEig, self).__init__()

        def forward(self, a):
            return np.linalg.eig(a)

    def check_eig(w, v, a_np):
        try:
            w_expected, v_expected = onp.linalg.eig(a_np)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print(e)
        else:
            assert w.shape == w_expected.shape
            assert v.shape == v_expected.shape
            if 0 not in a_np.shape:
                n = int(onp.prod(w.shape[:-1])) if len(shape) > 1 else 1
                N = a_np.shape[-1]
                w = w.reshape(n, N)
                w_expected = w_expected.reshape(n, N)
                v = v.reshape(n, N, N)
                v_expected = v_expected.reshape(n, N, N)
                a_np = a_np.reshape(n, N, N)
                for i in range(n):
                    # check eigenvector
                    ai = a_np[i]
                    vi = (v[i].asnumpy()).T
                    wi = w[i].asnumpy()
                    for j in range(N):
                        assert_almost_equal(wi[j] * vi[j], onp.matmul(ai, vi[j]), rtol=rtol, atol=atol)

                    # check eigenvalues
                    w1 = onp.sort(w[i].asnumpy())
                    w2 = onp.sort(w_expected[i])
                    assert_almost_equal(w1, w2, rtol=rtol, atol=atol)

    shapes = [
        (0, 0),
        (1, 1),
        (3, 3),
        (5, 5),
        (1, 0, 0),
        (0, 4, 4),
        (1, 4, 4),
        (2, 4, 4),
        (5, 5, 5),
        (1, 1, 4, 4),
        (2, 3, 4, 4)
    ]
    dtypes = ['float32', 'float64', 'uint8', 'int8', 'int32', 'int64']
    for hybridize in [True, False]:
        for shape, dtype in itertools.product(shapes, dtypes):
            rtol = 1e-2 if dtype == 'float32' else 1e-3
            atol = 1e-4 if dtype == 'float32' else 1e-5
            test_eig = TestEig()
            if hybridize:
                test_eig.hybridize()
            if 0 in shape:
                a_np = onp.ones(shape)
            else:
                if dtype == 'uint8' or dtype == 'int8' or dtype == 'int32' or dtype == 'int64':
                    n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
                    a_np = onp.array([onp.diag(onp.random.randint(1, 10, size=shape[-1])) for i in range(n)]).reshape(shape)
                else:
                    a_np = new_matrix_with_real_eigvals_nd(shape)
            a = np.array(a_np, dtype=dtype)
            # check eig validity
            mx_w, mx_v = test_eig(a)
            check_eig(mx_w, mx_v, a.asnumpy())

            # check imperative once again
            mx_w, mx_v = test_eig(a)
            check_eig(mx_w, mx_v, a.asnumpy())


@use_np
def test_np_linalg_eigh():
    class TestEigh(HybridBlock):
        def __init__(self, upper):
            super(TestEigh, self).__init__()
            self.upper = uppers

        def forward(self, a):
            return np.linalg.eigh(a, upper=self.upper)

    def check_eigh(w, v, a_np, upper):
        try:
            w_expected, v_expected = onp.linalg.eigh(a_np, upper)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print(e)
        else:
            assert w.shape == w_expected.shape
            assert v.shape == v_expected.shape
            # check eigenvalues.
            assert_almost_equal(w, w_expected, rtol=rtol, atol=atol)
            # check eigenvectors.
            w_shape, v_shape, a_sym_np = get_sym_matrix_nd(a_np, upper)
            w_np = w.asnumpy()
            v_np = v.asnumpy()
            if 0 not in a_np.shape:
                w_np = w_np.reshape(w_shape)
                v_np = v_np.reshape(v_shape)
                a_sym_np = a_sym_np.reshape(v_shape)
                for i in range(w_shape[0]):
                    for j in range(w_shape[1]):
                        assert_almost_equal(onp.dot(a_sym_np[i], v_np[i][:, j]), w_np[i][j] * v_np[i][:, j], rtol=rtol, atol=atol)

    def get_sym_matrix_nd(a_np, upper):
        a_res_np = a_np
        shape = a_np.shape
        if 0 not in a_np.shape:
            n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
            nrow, ncol = shape[-2], shape[-1]
            a_np = a_np.reshape(n, nrow, ncol)
            a_res_np = a_np
            for idx in range(n):
                for i in range(nrow):
                    for j in range(ncol):
                        if ((upper == False and i < j) or (upper == True and i > j)):
                            a_res_np[idx][i][j] = a_np[idx][j][i]
            return (n, nrow), (n, nrow, ncol), a_res_np.reshape(shape)
        else :
            return (0, 0), (0, 0, 0), a_res_np.reshape(shape)

    def new_matrix_from_sym_matrix_nd(sym_a, upper):
        shape = sym_a.shape
        if 0 in shape:
            return sym_a
        n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
        a = sym_a.reshape(n, shape[-2], shape[-1])
        for idx in range(n):
            for i in range(shape[-2]):
                for j in range(shape[-1]):
                    if ((upper == True and i > j) or (upper == False and i < j)):
                        a[idx][i][j] = onp.random.uniform(-10., 10.)
        return a.reshape(shape)

    shapes = [
        (0, 0),
        (1, 1),
        (3, 3),
        (5, 5),
        (1, 0, 0),
        (0, 4, 4),
        (1, 4, 4),
        (2, 4, 4),
        (5, 5, 5),
        (1, 1, 4, 4),
        (2, 3, 4, 4)
    ]
    dtypes = ['float32', 'float64', 'uint8', 'int8', 'int32', 'int64']
    uppers = [True, False]
    for hybridize in [True, False]:
        for shape, dtype, upper in itertools.product(shapes, dtypes, uppers):
            rtol = 1e-2 if dtype == 'float32' else 1e-3
            atol = 1e-4 if dtype == 'float32' else 1e-5
            test_eigh = TestEigh(upper)
            if hybridize:
                test_eigh.hybridize()
            if 0 in shape:
                a_np = onp.ones(shape)
            else:
                if dtype == 'uint8' or dtype == 'int8' or dtype == 'int32' or dtype == 'int64':
                    n = int(onp.prod(shape[:-2])) if len(shape) > 2 else 1
                    a_np = onp.array([onp.diag(onp.random.randint(1, 10, size=shape[-1])) for i in range(n)], dtype=dtype).reshape(shape)
                else:
                    a_np = new_sym_matrix_with_real_eigvals_nd(shape)
                    a_np = new_matrix_from_sym_matrix_nd(a_np, upper)
            a = np.array(a_np, dtype=dtype)
            # check eigh validity
            w, v = test_eigh(a)
            check_eigh(w, v, a.asnumpy(), upper)

            # check imperative once again
            w, v = test_eigh(a)
            check_eigh(w, v, a.asnumpy(), upper)


@use_np
def test_np_linalg_det():
    class TestDet(HybridBlock):
        def __init__(self):
            super(TestDet, self).__init__()

        def forward(self, a):
            return np.linalg.det(a)

    # test non zero size input
    tensor_shapes = [
        (2, 0, 2, 2),
        (4, 4),
        (0, 2, 2, 2),
        (3, 3, 3),
        (0, 2, 2),
        (2, 2, 2, 2, 2),
        (1, 1),
    ]
    types = [onp.float32, onp.float64]
    grad_reqs = ['write', 'add', 'null']

    for hybridize, dtype, shape, grad_req in itertools.product([True, False], types, tensor_shapes, grad_reqs):
        a_shape = (1,) + shape
        test_det = TestDet()
        if hybridize:
            test_det.hybridize()
        a = rand_ndarray(shape=a_shape, dtype=dtype).as_np_ndarray()
        a.attach_grad(grad_req)
        np_out = onp.linalg.det(a.asnumpy())
        with mx.autograd.record():
            mx_out = test_det(a)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-1, atol=1e-1)
        if grad_req != 'null':
            mx_out.backward()

        # Test imperative once again
        mx_out = np.linalg.det(a)
        np_out = onp.linalg.det(a.asnumpy())
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-1, atol=1e-1)

        # test numeric gradient
        a_sym = mx.sym.Variable("a").as_np_ndarray()
        mx_sym = mx.sym.np.linalg.det(a_sym).as_nd_ndarray()
        if 0 not in shape and grad_req != 'null':
            check_numeric_gradient(mx_sym, [a.as_nd_ndarray()], rtol=1e-1, atol=1e-1, dtype=dtype)


@use_np
@retry(3)
@pytest.mark.parametrize('grad_req', ['write', 'add', 'null'])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('a_shape', [
    (2, 0, 2, 2),
    (5, 5),
    (0, 2, 2, 2),
    (3, 3, 3),
    (0, 3, 3),
    (2, 2, 2, 2, 2),
    (1, 1)
])
@pytest.mark.xfail('win' in sys.platform, reason="Flaky test even with very high tolerance, tracked in #18184")
def test_np_linalg_slogdet(a_shape, grad_req, dtype, hybridize):
    class TestSlogdet(HybridBlock):
        def __init__(self):
            super(TestSlogdet, self).__init__()

        def forward(self, a):
            return np.linalg.slogdet(a)

    test_slogdet = TestSlogdet()
    if hybridize:
        test_slogdet.hybridize()
    a = rand_ndarray(shape=a_shape, dtype=dtype).as_np_ndarray()
    a.attach_grad(grad_req)

    np_out = onp.linalg.slogdet(a.asnumpy())
    with mx.autograd.record():
        mx_out = test_slogdet(a)
    assert mx_out[0].shape == np_out[0].shape
    assert mx_out[1].shape == np_out[1].shape
    assert_almost_equal(mx_out[0].asnumpy(), np_out[0], rtol=1e-1, atol=1e-1)
    assert_almost_equal(mx_out[1].asnumpy(), np_out[1], rtol=1e-1, atol=1e-1)
    if grad_req != 'null':
        mx_out[1].backward()

    # Test imperative once again
    mx_out = np.linalg.slogdet(a)
    np_out = onp.linalg.slogdet(a.asnumpy())
    assert_almost_equal(mx_out[0].asnumpy(), np_out[0], rtol=1e-1, atol=1e-1)
    assert_almost_equal(mx_out[1].asnumpy(), np_out[1], rtol=1e-1, atol=1e-1)


@use_np
def test_np_vstack():
    class TestVstack(HybridBlock):
        def __init__(self):
            super(TestVstack, self).__init__()

        def forward(self, a, *args):
            return np.vstack([a] + list(args))

    def g(data):
        return onp.ones_like(data)

    configs = [
        ((), (), ()),
        ((2), (2), (2)),
        ((0), (0), (0)),
        ((2, 2), (3, 2), (0, 2)),
        ((2, 3), (1, 3), (4, 3)),
        ((2, 2, 2), (3, 2, 2), (1, 2, 2)),
        ((0, 1, 1), (4, 1, 1), (5, 1, 1)),
        ((2), (0, 2), (2, 2))
    ]
    types = ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']
    for config in configs:
        for hybridize in [True, False]:
            for dtype in types:
                test_vstack = TestVstack()
                if hybridize:
                    test_vstack.hybridize()
                rtol = 1e-3
                atol = 1e-5
                v = []
                v_np = []
                for i in range(3):
                    v_np.append(onp.array(onp.random.uniform(-10.0, 10.0, config[i]), dtype=dtype))
                    v.append(mx.nd.array(v_np[i]).as_np_ndarray())
                    v[i].attach_grad()
                expected_np = onp.vstack(v_np)
                with mx.autograd.record():
                    mx_out = test_vstack(*v)
                assert mx_out.shape == expected_np.shape
                assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)

                # Test gradient
                mx_out.backward()
                for i in range(3):
                    expected_grad = g(v_np[i])
                    assert_almost_equal(v[i].grad.asnumpy(), expected_grad, rtol=rtol, atol=atol)

                # Test imperative once again
                mx_out = np.vstack(v)
                expected_np = onp.vstack(v_np)
                assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


@use_np
def test_np_full():
    class TestFull(HybridBlock):
        def __init__(self, shape, dtype=None):
            super(TestFull, self).__init__()
            self._shape = shape
            self._dtype = dtype

        def forward(self, a):
            return np.full(self._shape, a, dtype=self._dtype)

    configs = [
        ((3, 4), 2.0),
        ((0, 3), 2.0),
        ((2, 3), True),
        ((3, 0), False),
        ((3, 4), np.array(2.0)),
        ((0, 3), np.array(2.0)),
        ((2, 3), np.array([1, 2, 3], dtype=np.float32)),
        ((2, 3), np.array([1, 2, 3], dtype=np.int64)),
        ((0, 3), np.array([1, 2, 3], dtype=np.float32)),
        ((0, 3), np.array([1, 2, 3], dtype=np.int64)),
    ]

    rtol, atol = 1e-3, 1e-5
    dtypes = ['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool']
    for shape, fill_value in configs:
        for hybridize in [True, False]:
            for dtype in dtypes:
                if isinstance(fill_value, np.ndarray):
                    test_full = TestFull(shape, dtype=dtype)
                    if hybridize:
                        test_full.hybridize()
                    mx_out = test_full(fill_value)
                    expected_np = onp.full(shape, fill_value.asnumpy(), dtype=dtype)
                    assert mx_out.shape == expected_np.shape
                    assert mx_out.dtype == expected_np.dtype
                    assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)

                # Test imperative once again
                mx_out = np.full(shape, fill_value, dtype=dtype)
                if isinstance(fill_value, np.ndarray):
                    expected_np = onp.full(shape, fill_value.asnumpy(), dtype=dtype)
                else:
                    expected_np = onp.full(shape, fill_value, dtype=dtype)
                assert mx_out.shape == expected_np.shape
                assert mx_out.dtype == expected_np.dtype
                assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


@use_np
@pytest.mark.skip(reason='Skipped as the test is flaky and the feature causes curand error. Tracked in #18100')
def test_np_full_like():
    class TestFullLike(HybridBlock):
        def __init__(self, fill_value, dtype, device):
            super(TestFullLike, self).__init__()
            self._fill_value = fill_value
            self._dtype = dtype
            self._device = device

        def forward(self, x, *args, **kwargs):
            return np.full_like(x, self._fill_value, dtype=self._dtype, device=self._device)

    if StrictVersion(platform.python_version()) < StrictVersion('3.0.0'):
        return

    dtypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int8', 'bool']
    shapes = [
        (),
        (1,),
        (4, 3),
        (4, 5),
        (2, 1),
        (6, 5, 6),
        (4, 2, 1, 2),
        (5, 1, 3, 3),
        (3, 3, 1, 0),
    ]
    # numpy.full_like operator in py2 cannot handle shape like (5, 0, 3) properly
    fill_values = [0, 1, 2, 3, 4, 5, 6, True, False]
    flags = [True, False]
    for fill_value, dtype, shape, hybridize in itertools.product(
        fill_values, dtypes, shapes, flags):
        param_dtype = onp.random.choice(dtypes)
        a = np.random.uniform(low=0, high=100, size=shape, dtype='float64').astype(dtype)
        test = TestFullLike(fill_value, param_dtype, npx.current_device())
        expected_ret = onp.full_like(a.asnumpy(), fill_value=fill_value, dtype=param_dtype)
        if hybridize:
            test.hybridize()
        ret = test(a)
        assert_almost_equal(ret.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)

        # check imperative again
        ret = np.full_like(a, fill_value, param_dtype)
        assert_almost_equal(ret.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)


@use_np
def test_np_roll():
    class TestRoll(HybridBlock):
        def __init__(self, shift=None, axis=None):
            super(TestRoll, self).__init__()
            self._shift = shift
            self._axis = axis

        def forward(self, x):
            return np.roll(x, shift=self._shift, axis=self._axis)

    dtypes = ['int32', 'int64', 'float16', 'float32', 'float64']
    configs = [
        ((), (3,), None),
        ((1,), (-3,), None),
        ((20,), (-3,), None),
        ((3,), (2,), 0),
        ((2, 3, 4), (12,), (1,)),
        ((2, 3, 4), (10, -10), (0, 1)),
        ((2, 3, 4, 5), (0, 1), (-1, 2)),
        ((2, 3, 0, 1), (0, 1), (-1, 2)),
        ((2, 3, 4, 5), 10, (0, 2)),
    ]
    i_dtype = {"float32" : onp.float32,
               "float64" : onp.float64
               }
    for dtype in dtypes:
        for config in configs:
            for hybridize in [False, True]:
                shape, shift, axis = config[0], config[1], config[2]
                x = rand_ndarray(shape=shape, dtype=dtype).as_np_ndarray()
                net = TestRoll(shift=shift, axis=axis)
                np_out = onp.roll(x.asnumpy(), shift=shift, axis=axis)
                if hybridize:
                    net.hybridize()
                x.attach_grad()
                with mx.autograd.record():
                    mx_out = net(x)
                assert mx_out.shape == np_out.shape
                mx_out.backward()
                assert same(mx_out.asnumpy(), np_out)
                assert same(x.grad.shape, x.shape)
                assert same(x.grad.asnumpy(), onp.ones(shape))

                # test imperativen
                np_out = onp.roll(x.asnumpy(), shift=shift, axis=axis)
                mx_out = np.roll(x, shift=shift, axis=axis)
                assert same(mx_out.asnumpy(), np_out)

                # test numeric
                if dtype in ['float32', 'float64'] and len(shape)> 0 and  onp.prod(shape) > 0:
                    x_sym = mx.sym.Variable("x").as_np_ndarray()
                    mx_sym = mx.sym.np.roll(x_sym, shift=shift, axis=axis).as_nd_ndarray()
                    check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                           numeric_eps=1e-3, rtol=1e-3, atol=1e-5, dtype=i_dtype[dtype])


@use_np
def test_np_trace():
    class TestTrace(HybridBlock):
        def __init__(self, axis1, axis2, offset):
            super(TestTrace, self).__init__()
            self._axis1 = axis1
            self._axis2 = axis2
            self._offset = offset

        def forward(self, data):
            return np.trace(data, axis1=self._axis1, axis2=self._axis2, offset=self._offset)

    def g(data, axis1, axis2, offset):
        idx = onp.indices(data.shape)
        ret = onp.zeros_like(data)
        ret[idx[axis1] + offset == idx[axis2]] = 1.0
        return ret

    shapes = [
        (3, 3),
        (3, 4),
        (0, 0),
        (3, 3, 3),
        (0, 0, 0),
        (2, 2, 4, 3),
        (2, 2, 4, 3),
        (2, 0, 3, 0),
        (2, 0, 2, 3)
    ]
    offsets = range(-5, 5)
    dtypes = ['int32', 'float16', 'float32', 'float64']
    for hybridize in [True, False]:
        for shape in shapes:
            ndim = len(shape)
            for axis1 in range(-ndim, ndim):
                for axis2 in range(-ndim, ndim):
                    if (axis1 + ndim) % ndim != (axis2 + ndim) % ndim:
                        for offset in offsets:
                            for dtype in dtypes:
                                if dtype == 'float16':
                                    rtol = atol = 1e-2
                                else:
                                    rtol = atol = 1e-5
                                test_trace = TestTrace(axis1, axis2, offset)
                                if hybridize:
                                    test_trace.hybridize()
                                data_np = onp.random.uniform(-10.0, 10.0, shape)
                                data = mx.nd.array(data_np, dtype=dtype)
                                data_np = data.asnumpy()
                                data.attach_grad()
                                expected_np = onp.trace(data_np, axis1=axis1, axis2=axis2, offset=offset)
                                with mx.autograd.record():
                                    out_mx = test_trace(data.as_np_ndarray())
                                assert out_mx.shape == expected_np.shape
                                assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)
                                out_mx.backward()
                                backward_expected = g(data_np, axis1=axis1, axis2=axis2, offset=offset)
                                assert_almost_equal(data.grad.asnumpy(), backward_expected, rtol=rtol, atol=atol)

                                # Test imperative once again
                                data = mx.nd.array(data_np, dtype=dtype)
                                out_mx = np.trace(data.as_np_ndarray(), axis1=axis1, axis2=axis2, offset=offset)
                                assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)

    # bad params
    params = [
        ([], 0, 1, 0),
        ([2], 0, 1, 0),
        ([3, 2, 2], 1, 1, 1),
        ([3, 2, 2], 0, -4, 1)
    ]
    for shape, axis1, axis2, offset in params:
        data_np = onp.random.uniform(-1.0, 1.0, shape)
        data_mx = mx.nd.array(data_np)
        try:
            output = np.trace(data_mx.as_np_ndarray(), axis1=axis1, axis2=axis2, offset=offset)
        except mx.base.MXNetError:
            continue
        assert False


@use_np
def test_np_windows():
    class TestWindows(HybridBlock):
        def __init__(self, func, M):
            super(TestWindows, self).__init__()
            self._func = func
            self._M = M

        def forward(self, x, *args, **kwargs):
            op = getattr(np, self._func)
            assert op is not None
            return x + op(M=self._M)

    configs = [-10, -3, -1, 0, 1, 6, 10, 20]
    dtypes = ['float32', 'float64']
    funcs = ['hanning', 'hamming', 'blackman']
    for config in configs:
        for dtype in dtypes:
            for func in funcs:
                x = np.zeros(shape=(), dtype=dtype)
                for hybridize in [False, True]:
                    np_func = getattr(onp, func)
                    mx_func = TestWindows(func, M=config)
                    np_out = np_func(M=config).astype(dtype)
                    if hybridize:
                        mx_func.hybridize()
                    mx_out = mx_func(x)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                    # test imperative
                    mx_out = getattr(np, func)(M=config)
                    np_out = np_func(M=config).astype(dtype)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_flip():
    class TestFlip(HybridBlock):
        def __init__(self, axis):
            super(TestFlip, self).__init__()
            self.axis = axis

        def forward(self, x):
            return np.flip(x, self.axis)

    shapes = [(1, 2, 3), (1, 0), ()]
    types = ['int32', 'int64', 'float16', 'float32', 'float64']
    for hybridize in [True, False]:
        for oneType in types:
            rtol, atol=1e-3, 1e-5
            for shape in shapes:
                axis = random.randint(-len(shape), len(shape))
                if axis == len(shape):
                    axis = None
                test_flip = TestFlip(axis)
                if hybridize:
                    test_flip.hybridize()
                x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                x.attach_grad()
                np_out = onp.flip(x.asnumpy(), axis)
                with mx.autograd.record():
                    mx_out = test_flip(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                mx_out.backward()
                np_backward = onp.ones(np_out.shape)
                assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

                # Test imperative once again
                mx_out = np.flip(x, axis)
                np_out = onp.flip(x.asnumpy(), axis)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_flipud_fliplr():
    class TestFlipud(HybridBlock):
        def __init__(self):
            super(TestFlipud, self).__init__()

        def forward(self, x):
            return np.flipud(x)

    class TestFliplr(HybridBlock):
        def __init__(self):
            super(TestFliplr, self).__init__()

        def forward(self, x):
            return np.fliplr(x)

    shapes = [(1, 2, 3), (1, 0)]
    types = ['int32', 'int64', 'float16', 'float32', 'float64']
    for func in ['flipud', 'fliplr']:
        for hybridize in [True, False]:
            for oneType in types:
                rtol, atol=1e-3, 1e-5
                for shape in shapes:
                    if func == 'flipud':
                        test_flip = TestFlipud()
                    else:
                        test_flip = TestFliplr()
                    if hybridize:
                        test_flip.hybridize()
                    x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                    x.attach_grad()
                    if func == 'flipud':
                        np_out = onp.flipud(x.asnumpy())
                    else:
                        np_out = onp.fliplr(x.asnumpy())
                    with mx.autograd.record():
                        mx_out = test_flip(x)
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                    mx_out.backward()
                    np_backward = onp.ones(np_out.shape)
                    assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

                    # Test imperative once again
                    if func == 'flipud':
                        mx_out = np.flipud(x)
                        np_out = onp.flipud(x.asnumpy())
                    else:
                        mx_out = np.fliplr(x)
                        np_out = onp.fliplr(x.asnumpy())
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
@pytest.mark.flaky
def test_np_around():
    class TestAround(HybridBlock):
        def __init__(self, decimals):
            super(TestAround, self).__init__()
            self.decimals = decimals

        def forward(self, x):
            return np.around(x, self.decimals)

    shapes = [(), (1, 2, 3), (1, 0)]
    types = ['int32', 'int64', 'float32', 'float64']
    for hybridize in [True, False]:
        for oneType in types:
            rtol, atol = 1e-3, 1e-5
            for shape in shapes:
                for d in range(-5, 6):
                    test_around = TestAround(d)
                    if hybridize:
                        test_around.hybridize()
                    x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                    np_out = onp.around(x.asnumpy(), d)
                    mx_out = test_around(x)
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)

                    mx_out = np.around(x, d)
                    np_out = onp.around(x.asnumpy(), d)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_flatnonzero():
    class TestFlatnonzero(HybridBlock):
        def __init__(self):
            super(TestFlatnonzero, self).__init__()

        def forward(self, a):
            return np.flatnonzero(a)

    shapes = [(1,), (4, 3), (4, 5), (2, 1), (6, 5, 6), (4, 2, 1, 2),
              (5, 1, 3, 3), (3, 3, 1, 0),]
    types = ['int32', 'int64', 'float32', 'float64']
    hybridizes = [True, False]
    for hybridize, oneType, shape in itertools.product(hybridizes, types, shapes):
        rtol, atol = 1e-3, 1e-5
        test_flatnonzero = TestFlatnonzero()
        if hybridize:
            test_flatnonzero.hybridize()
        x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
        np_out = onp.flatnonzero(x.asnumpy())
        mx_out = test_flatnonzero(x)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)

        mx_out = np.flatnonzero(x)
        np_out = onp.flatnonzero(x.asnumpy())
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_round():
    class TestRound(HybridBlock):
        def __init__(self, func, decimals):
            super(TestRound, self).__init__()
            self.func = func
            self.decimals = decimals

        def forward(self, x):
            return getattr(np, self.func)(x, self.decimals)

    shapes = [(), (1, 2, 3), (1, 0)]
    types = ['int32', 'int64', 'float32', 'float64']
    funcs = ['round', 'round_']
    for hybridize, oneType, func in itertools.product([True, False], types, funcs):
        rtol, atol = 1e-3, 1e-5
        for shape in shapes:
            for d in range(-5, 6):
                test_round = TestRound(func, d)
                if hybridize:
                    test_round.hybridize()
                x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                np_out = getattr(onp, func)(x.asnumpy(), d)
                mx_out = test_round(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)

                mx_out = getattr(mx.np, func)(x, d)
                np_out = getattr(onp, func)(x.asnumpy(), d)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_nonzero():
    class TestNonzero(HybridBlock):
        def __init__(self):
            super(TestNonzero, self).__init__()

        def forward(self, x):
            return npx.nonzero(x)

    types = ['int32', 'int64', 'float64', 'float32', 'float16']
    for hybridize in [True, False]:
        for shape in [(), (1, 2, 3), (1, 0)]:
            for oneType in types:
                rtol, atol = 1e-3, 1e-5
                test_nonzero = TestNonzero()
                if hybridize:
                    test_nonzero.hybridize()
                x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                np_out = onp.nonzero(x.asnumpy())
                np_out = onp.transpose(np_out)
                mx_out = test_nonzero(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol, atol)

                # Test imperative once again
                mx_out = npx.nonzero(x)
                np_out = onp.nonzero(x.asnumpy())
                np_out = onp.transpose(np_out)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol, atol)


@use_np
def test_np_unique():
    class TestUnique(HybridBlock):
        def __init__(self, return_index=False, return_inverse=False, return_counts=False, axis=None):
            super(TestUnique, self).__init__()
            self._return_index = return_index
            self._return_inverse = return_inverse
            self._return_counts = return_counts
            self._axis = axis

        def forward(self, a):
            return np.unique(a, self._return_index, self._return_inverse, self._return_counts, self._axis)

    configs = [
        ((), True, True, True, None),
        ((1, ), True, True, True, -1),
        ((5, ), False, False, False, 0),
        ((5, ), True, False, False, 0),
        ((5, ), True, True, False, 0),
        ((5, ), True, True, True, 0),
        ((5, ), True, True, True, None),
        ((5, 4), True, True, True, None),
        ((5, 4), True, True, True, -1),
        ((5, 0, 4), True, True, True, None),
        ((0, 0, 0), True, True, True, None),
        # ((5, 3, 4), True, True, True, -1), # waiting for numpy 1.18, details in pr 14255
        ((5, 3, 4), True, True, True, None),
        ((5, 3, 4), True, True, True, 1),
    ]
    for dtype in ['float32', 'float64', 'int8', 'uint8', 'int32', 'int64']:
        for hybridize in [False, True]:
            for config in configs:
                test_unique = TestUnique(*config[1:])
                if hybridize:
                    test_unique.hybridize()
                x = onp.random.uniform(-8.0, 8.0, size=config[0])
                x = np.array(x, dtype=dtype)
                np_out = onp.unique(x.asnumpy(), *config[1:])
                mx_out = test_unique(x)
                if (len(mx_out)) == 1:
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                else:
                    for i in range(len(mx_out)):
                        assert mx_out[i].shape == np_out[i].shape
                        assert_almost_equal(mx_out[i].asnumpy(), np_out[i], rtol=1e-3, atol=1e-5)

                # Test imperative once again
                mx_out = np.unique(x, *config[1:])
                np_out = onp.unique(x.asnumpy(), *config[1:])
                if (len(mx_out)) == 1:
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                else:
                    for i in range(len(mx_out)):
                        assert mx_out[i].shape == np_out[i].shape
                        assert_almost_equal(mx_out[i].asnumpy(), np_out[i], rtol=1e-3, atol=1e-5)


@use_np
@pytest.mark.parametrize('shape,index,inverse,counts', [
    ((), True, True, True),
    ((1, ), True, True, True),
    ((5, ), True, True, True),
    ((5, ), True, True, True),
    ((5, 4), True, True, True),
    ((5, 0, 4), True, True, True),
    ((0, 0, 0), True, True, True),
    ((5, 3, 4), True, True, True),
])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int8', 'uint8', 'int32', 'int64'])
@pytest.mark.parametrize('hybridize', [False, True])
def test_np_unique_all(shape, index, inverse, counts, dtype, hybridize):
    class TestUniqueAll(HybridBlock):
        def __init__(self):
            super(TestUniqueAll, self).__init__()

        def forward(self, a):
            return np.unique_all(a)

    test_unique = TestUniqueAll()
    if hybridize:
        test_unique.hybridize()
    x = onp.random.uniform(-8.0, 8.0, size=shape)
    x = np.array(x, dtype=dtype)
    np_out = onp.unique(x.asnumpy(), return_index=index, return_inverse=inverse, return_counts=counts)
    mx_out = test_unique(x)
    for i in range(len(mx_out)):
        assert mx_out[i].shape == np_out[i].shape
        assert_almost_equal(mx_out[i].asnumpy(), np_out[i], rtol=1e-3, atol=1e-5)

    # Test imperative once again
    mx_out = np.unique_all(x)
    np_out = onp.unique(x.asnumpy(), return_index=index, return_inverse=inverse, return_counts=counts)
    assert mx_out.values.shape == np_out[0].shape
    assert_almost_equal(mx_out.values.asnumpy(), np_out[0], rtol=1e-3, atol=1e-5)
    assert mx_out.indices.shape == np_out[1].shape
    assert_almost_equal(mx_out.indices.asnumpy(), np_out[1], rtol=1e-3, atol=1e-5)
    assert mx_out.inverse_indices.shape == np_out[2].shape
    assert_almost_equal(mx_out.inverse_indices.asnumpy(), np_out[2], rtol=1e-3, atol=1e-5)
    assert mx_out.counts.shape == np_out[3].shape
    assert_almost_equal(mx_out.counts.asnumpy(), np_out[3], rtol=1e-3, atol=1e-5)


@use_np
@pytest.mark.parametrize('shape,index,inverse,counts', [
    ((), False, True, False),
    ((1, ), False, True, False),
    ((5, ), False, True, False),
    ((5, ), False, True, False),
    ((5, 4), False, True, False),
    ((5, 0, 4), False, True, False),
    ((0, 0, 0), False, True, False),
    ((5, 3, 4), False, True, False),
])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int8', 'uint8', 'int32', 'int64'])
@pytest.mark.parametrize('hybridize', [False, True])
def test_np_unique_inverse(shape, index, inverse, counts, dtype, hybridize):
    class TestUniqueInverse(HybridBlock):
        def __init__(self):
            super(TestUniqueInverse, self).__init__()

        def forward(self, a):
            return np.unique_inverse(a)

    test_unique = TestUniqueInverse()
    if hybridize:
        test_unique.hybridize()
    x = onp.random.uniform(-8.0, 8.0, size=shape)
    x = np.array(x, dtype=dtype)
    np_out = onp.unique(x.asnumpy(), return_index=index, return_inverse=inverse, return_counts=counts)
    mx_out = test_unique(x)
    for i in range(len(mx_out)):
        assert mx_out[i].shape == np_out[i].shape
        assert_almost_equal(mx_out[i].asnumpy(), np_out[i], rtol=1e-3, atol=1e-5)

    # Test imperative once again
    mx_out = np.unique_inverse(x)
    np_out = onp.unique(x.asnumpy(), return_index=index, return_inverse=inverse, return_counts=counts)
    assert mx_out.values.shape == np_out[0].shape
    assert_almost_equal(mx_out.values.asnumpy(), np_out[0], rtol=1e-3, atol=1e-5)
    assert mx_out.inverse_indices.shape == np_out[1].shape
    assert_almost_equal(mx_out.inverse_indices.asnumpy(), np_out[1], rtol=1e-3, atol=1e-5)


@use_np
@pytest.mark.parametrize('shape,index,inverse,counts', [
    ((), False, False, False),
    ((1, ), False, False, False),
    ((5, ), False, False, False),
    ((5, ), False, False, False),
    ((5, 4), False, False, False),
    ((5, 0, 4), False, False, False),
    ((0, 0, 0), False, False, False),
    ((5, 3, 4), False, False, False),
])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int8', 'uint8', 'int32', 'int64'])
@pytest.mark.parametrize('hybridize', [False, True])
def test_np_unique_values(shape, index, inverse, counts, dtype, hybridize):
    class TestUniqueValues(HybridBlock):
        def __init__(self):
            super(TestUniqueValues, self).__init__()

        def forward(self, a):
            return np.unique_values(a)

    test_unique = TestUniqueValues()
    if hybridize:
        test_unique.hybridize()
    x = onp.random.uniform(-8.0, 8.0, size=shape)
    x = np.array(x, dtype=dtype)
    np_out = onp.unique(x.asnumpy(), return_index=index, return_inverse=inverse, return_counts=counts)
    mx_out = test_unique(x)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    # Test imperative once again
    mx_out = np.unique_values(x)
    np_out = onp.unique(x.asnumpy(), return_index=index, return_inverse=inverse, return_counts=counts)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_take():
    configs = [
        ((4, 4), (4, 0), None),
        ((4, 4), (4, 0), 0),
        ((4, 4), (4, 0), 1),
        ((), (4, 0), None),
        ((), (5, ), None),
        ((), (4, 5), None),
        ((), (), None),
        ((3, 4), (), None),
        ((3, 4), (), 0),
        ((3, 4), (), 1),
        ((3, 4, 5), (), 2),
        ((3, 4, 5), (), -3),
    ]

    class TestTake(HybridBlock):
        def __init__(self, axis, mode):
            super(TestTake, self).__init__()
            self._axis = axis
            self._mode = mode

        def forward(self, a, indices):
            return np.take(a, indices, axis=self._axis, mode=self._mode)

    def grad_helper(grad_in, axis, idx, mode):
        k = 1 if axis == None else grad_in.shape[axis]
        if mode == 'clip':
            idx = 0 if idx < 0 else idx
            idx = k - 1 if idx >= k else idx
        else:
            idx = idx % k

        if axis == None:
            if grad_in.shape == ():
                grad_in += 1.0
            else:
                grad_in[idx] += 1.0
        elif axis == 0:
            if axis == len(grad_in.shape) - 1:
                grad_in[idx] += 1.0
            else:
                grad_in[idx, :] += 1.0
        elif axis == 1:
            if axis == len(grad_in.shape) - 1:
                grad_in[:, idx] += 1.0
            else:
                grad_in[:, idx, :] += 1.0
        elif axis == 2:
            if axis == len(grad_in.shape) - 1:
                grad_in[:, :, idx] += 1.0
            else:
                grad_in[:, :, idx, :] += 1.0
        elif axis == 3:
            if axis == len(grad_in.shape) - 1:
                grad_in[:, :, :, idx] += 1.0
            else:
                grad_in[:, :, :, idx, :] += 1.0
        elif axis == 4:
            grad_in[:, :, :, :, idx] += 1.0
        else:
            raise ValueError(f"axis {axis} is not supported...")

    def check_output_n_grad(data_shape, idx_shape, axis, mode):
        data_real = onp.random.normal(size=data_shape).astype('float32')
        idx_real = onp.random.randint(low=-100, high=100, size=idx_shape)

        assert same(np.take(np.array(data_real), np.array(idx_real), axis=axis, mode=mode).asnumpy(),
             onp.take(data_real, idx_real, axis=axis, mode=mode))

        grad_in = onp.zeros(data_shape, dtype='float32')

        test_take = TestTake(axis=axis, mode=mode)
        if hybridize:
            test_take.hybridize()
        x = np.array(data_real)
        x.attach_grad()
        with mx.autograd.record():
            mx_out = test_take(x, np.array(idx_real))
        assert same(mx_out.asnumpy(), onp.take(data_real, idx_real, axis=axis, mode=mode))

        if axis and axis < 0:
            axis += len(data_shape)

        if idx_real.size != 0:
            for i in onp.nditer(idx_real):
                grad_helper(grad_in, axis, i, mode)


        mx_out.backward()
        same(x.grad.asnumpy(), grad_in)

    for hybridize in [True, False]:
        for mode in ['clip', 'wrap']:
            for data_ndim in range(1, 5):
                for idx_ndim in range(1, 4):
                    for axis in range(-data_ndim, data_ndim):
                        data_shape = ()
                        for _ in range(data_ndim):
                            data_shape += (onp.random.randint(low=1, high=5), )
                        idx_shape = ()
                        for _ in range(idx_ndim):
                            idx_shape += (onp.random.randint(low=1, high=5), )
                        check_output_n_grad(data_shape, idx_shape, axis, mode)

            for config in configs:
                check_output_n_grad(config[0], config[1], config[2], mode)


def test_np_builtin_op_signature():
    import inspect
    from mxnet import _numpy_op_doc
    builtin_np_op_names = [name for name in get_all_registered_operators() if name.startswith('_np_')]
    for op_name in builtin_np_op_names:
        _op_from_doc = getattr(_numpy_op_doc, op_name, None)
        assert _op_from_doc is not None, "Failed to find documentation for operator {}. " \
                                         "Please add the documentation in _numpy_op_doc.py for this operator."\
            .format(op_name)
        op = _get_builtin_op(op_name)
        assert op is not None
        assert str(op.__signature__) == str(inspect.signature(_op_from_doc))


@use_np
def test_np_tril_indices():
    class TestTrilindices(HybridBlock):
        def __init__(self, n, k=0, m=None):
            super(TestTrilindices, self).__init__()
            self._n = n;
            self._k = k;
            if m is None:
                m = n
            self._m = m

        def forward(self, x, *args, **kwargs):
            return x, np.tril_indices(n=self._n, k=self._k, m=self._m)

    for n in onp.random.random_integers(-10, 50, 2):
        for k in onp.random.random_integers(-50, 50, 2):
            for m in onp.random.random_integers(-10, 50, 2):
                np_out = onp.tril_indices(n, k, m)
                for hybridize in [True, False]:
                    # dummy nparray for hybridize
                    x = np.ones((1,1))
                    test_trilindices = TestTrilindices(int(n), int(k), int(m))
                    if hybridize:
                        test_trilindices.hybridize()
                    mx_out = test_trilindices(x)[1]
                    assert len(mx_out) == 2
                    assert same(mx_out[0], np_out[0])
                    assert same(mx_out[1], np_out[1])
                    if n > 0 and m > 0 and hybridize is False:
                        np_data = onp.arange(n*m).reshape(n, m)
                        mx_data = np.array(np_data)
                        np_data[np_out] = -10
                        mx_data[mx_out] = -10
                        assert same(np_data, mx_data.asnumpy())


@use_np
def test_np_fill_diagonal():
    class TestFillDiagonal(HybridBlock):
        def __init__(self, val, wrap=False):
            super(TestFillDiagonal, self).__init__()
            self._val = val
            self._wrap= wrap

        def forward(self, x):
            return np.fill_diagonal(x, val=self._val, wrap=self._wrap)

    configs = [
        ((10, 10), 2),
        ((10, 10), -2),
        ((4, 10), -2),
        ((10, 4), 2),
        ((10, 10), [-2, 2]),
        ((10, 10), [-2, 2]),
        ((10, 5), [-2, 2, -1, -3]),
        ((100, 50), [-2, 2, -1, -3]),
        ((1000, 500), [-2, 2, -1, -3]),
        ((5, 10), [-2, 2, -1, -3]),
        ((50, 100), [-2, 2, -1, -3]),
        ((500, 1000), [-2, 2, -1, -3]),
        ((4, 4, 4), 2),
        ((4, 4, 4, 4), 2),
        ((4, 4, 4, 4, 4), [-1, 2]),
        ((4, 4, 4, 4, 4, 4, 4, 4), 2),
        ((5, 5, 5, 5, 5, 5, 5, 5), [-1, 2, -2]),
        ((6, 6, 6, 6, 6, 6, 6, 6), 2),
        ((7, 7, 7, 7, 7, 7, 7, 7), [-1, 2, -2]),
    ]
    dtypes = ['int8', 'int32', 'int64', 'float16', 'float32', 'float64']
    for dtype in dtypes:
        for config in configs:
            for wrap in [False, True]:
                np_data = onp.ones(config[0]).astype(dtype)
                mx_data = np.array(np_data, dtype=dtype)
                test_filldiagonal = TestFillDiagonal(config[1], wrap)
                test_filldiagonal(mx_data)
                onp.fill_diagonal(np_data, config[1], wrap)
                assert same(np_data, mx_data.asnumpy())


@use_np
def test_np_moveaxis():
    class TestMoveaxis(HybridBlock):
        def __init__(self, source=None, destination=None):
            super(TestMoveaxis, self).__init__()
            self._source = source
            self._destination= destination

        def forward(self, x):
            return np.moveaxis(x, source=self._source, destination=self._destination)

    dtypes = ['int32', 'int64', 'float16', 'float32', 'float64']
    for hybridize in [False, True]:
        for dtype in dtypes:
            for ndim in [0, 1, 2, 3, 4, 5, 6]:
                shape = rand_shape_nd(ndim, dim=5, allow_zero_size=True)
                np_data = onp.random.uniform(low=-100, high=100, size=shape).astype(dtype)
                mx_data = np.array(np_data, dtype=dtype)
                axis = [i for i in range(ndim)]
                random.shuffle(axis)
                for i in range(ndim):
                    source = random.sample(axis, i)
                    destination = random.sample(axis, i)

                    # test gluon
                    test_moveaxis = TestMoveaxis(source,destination)
                    if hybridize:
                        test_moveaxis.hybridize()
                    np_out = onp.moveaxis(np_data, source=source, destination=destination)
                    mx_data.attach_grad()
                    with mx.autograd.record():
                        mx_out = test_moveaxis(mx_data)
                    assert mx_out.shape == np_out.shape
                    mx_out.backward()
                    assert same(mx_data.grad.shape, mx_data.shape)
                    assert same(mx_data.grad.asnumpy(), onp.ones(shape))
                    # test imperative
                    np_out = onp.moveaxis(np_data, source=source, destination=destination)
                    mx_out = np.moveaxis(mx_data, source=source, destination= destination)
                    assert np_out.dtype == mx_out.dtype
                    assert same(mx_out.asnumpy(), np_out)


@use_np
def test_np_rot90():
    class TestTRot90(HybridBlock):
        def __init__(self, k=1, axes=(0, 1)):
            super(TestTRot90, self).__init__()
            self._k = k
            self._axes = axes

        def forward(self, a, *args):
            return np.rot90(a, self._k, self._axes)

    configs = [
        ((2, 3), 1, (0, 1)),
        ((2, 3), 3, (0, 1)),
        ((2, 3), 1, (1, 0)),
        ((2, 3), 2, (1, 0)),
        ((2, 3), 3, (1, 0)),
        ((2, 3), 0, (1, 0)),
        ((2, 3, 4, 5), 3, (1, 2)),
        ((2, 3, 4, 5), -3, (2, 3)),
        ((2, 3, 0, 5), -2, (2, 3)),
        ((2, 0, 0, 5), -3, (2, 3)),
        ((2, 3, 0, 5), 0, (2, 1)),
    ]
    dtypes = ['uint8', 'int8', 'int32', 'int64', 'float16', 'float32', 'float64']

    for config in configs:
        for dtype in dtypes:
            for hybridize in [True, False]:
                shape, k, axes = config[0], config[1], config[2]
                x = rand_ndarray(shape=shape, dtype=dtype).as_np_ndarray()
                net = TestTRot90(k=k, axes=axes)
                if hybridize:
                    net.hybridize()

                x.attach_grad()
                np_out = onp.rot90(x.asnumpy(), k=k, axes=axes)
                with mx.autograd.record():
                    mx_out = net(x)
                assert mx_out.shape == np_out.shape
                assert same(mx_out.asnumpy(), np_out)
                mx_out.backward()
                np_backward = onp.ones(shape, dtype)

                assert same(x.grad.asnumpy().shape, np_backward.shape)
                assert same(x.grad.asnumpy(), np_backward)

                np_out = onp.rot90(x.asnumpy(), k=k, axes=axes)
                mx_out = np.rot90(x, k=k, axes=axes)
                assert same(mx_out.asnumpy(), np_out)


@use_np
def test_np_hsplit():
    class TestHSplit(HybridBlock):
        def __init__(self, indices_or_sections):
            super(TestHSplit, self).__init__()
            self._indices_or_sections = indices_or_sections

        def forward(self, a, *args, **kwargs):
            return np.hsplit(a, indices_or_sections=self._indices_or_sections)

    shapes = [
        (10,),
        (3, 8, 5),
        (3, 0, 5),
        (3, 8, 5, 6),
        (3, 0, 5, 6),
    ]
    indices_or_sections_num = [
        (2, 4),
        (3, 3),
        (3,),
        (1,),
        2,
    ]
    for hybridize in [True, False]:
        for shape in shapes:
            for indices_or_sections in indices_or_sections_num:
                # test gluon
                test_hsplit = TestHSplit(indices_or_sections=indices_or_sections)
                if hybridize:
                    test_hsplit.hybridize()

                a = mx.nd.random.uniform(-1.0, 1.0, shape=shape).as_np_ndarray()
                a.attach_grad()
                expected_ret = onp.hsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                with mx.autograd.record():
                    y = test_hsplit(a)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                mx.autograd.backward(y)
                assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_outs = np.hsplit(a, indices_or_sections=indices_or_sections)
                np_outs = onp.hsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_dsplit():
    class TestDSplit(HybridBlock):
        def __init__(self, indices_or_sections):
            super(TestDSplit, self).__init__()
            self._indices_or_sections = indices_or_sections

        def forward(self, a, *args, **kwargs):
            return np.dsplit(a, indices_or_sections=self._indices_or_sections)

    shapes = [
        (2, 4, 6),
        (3, 0, 6),
        (2, 3, 0, 4),
    ]
    indices_or_sections_num = [
        (2, 4),
        (3, 3),
        (3,),
        (1,),
        2,
    ]
    for hybridize in [True, False]:
        for shape in shapes:
            for indices_or_sections in indices_or_sections_num:
                # test gluon
                test_dsplit = TestDSplit(indices_or_sections=indices_or_sections)
                if hybridize:
                    test_dsplit.hybridize()

                a = mx.nd.random.uniform(-1.0, 1.0, shape=shape).as_np_ndarray()
                a.attach_grad()
                expected_ret = onp.dsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                with mx.autograd.record():
                    y = test_dsplit(a)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                mx.autograd.backward(y)
                assert_almost_equal(a.grad.asnumpy(), onp.ones(a.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_outs = np.dsplit(a, indices_or_sections=indices_or_sections)
                np_outs = onp.dsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_einsum():
    class TestEinsum(HybridBlock):
        def __init__(self, subscripts, optimize):
            super(TestEinsum, self).__init__()
            self.subscripts = subscripts
            self.optimize = optimize

        def forward(self, *operands):
            return np.einsum(self.subscripts, *operands, optimize=self.optimize)

    def dbg(name, data):
        print('type of {} = {}'.format(name, type(data)))
        print('shape of {} = {}'.format(name, data.shape))
        print('{} = {}'.format(name, data))

    configs = [
        ('ii', [(5, 5)], lambda *args: (onp.eye(5),)),
        ('ii->i', [(5, 5)], lambda *args: (onp.eye(5),)),
        ('ij->i', [(5, 5)], lambda *args: (onp.ones((5, 5)),)),
        ('...j->...', [(5, 5)], lambda *args: (onp.ones((5, 5)),)),
        ('ji', [(2, 3)], lambda *args: (onp.ones((2, 3)),)),
        ('ij->ji', [(2, 3)], lambda *args: (onp.ones((2, 3)),)),
        ('i, i', [(5,), (5,)], lambda *args: (args[1], args[0])),
        ('ij, j', [(5, 5), (5,)], lambda *args: (onp.tile(args[1][None, :], [5, 1]),
                                                 args[0].sum(axis=0))),
        ('...j, j', [(5, 5), (5,)], lambda *args: (onp.tile(args[1][None, :], [5, 1]),
                                                   onp.sum(args[0], axis=0))),
        ('..., ...', [(), (2, 3)], lambda *args: (onp.sum(args[1], axis=None),
                                                  args[0] * onp.ones((2, 3)))),
        (', ij', [(), (2, 3)], lambda *args: (onp.sum(args[1], axis=None),
                                              args[0] * onp.ones((2, 3)))),
        ('i, j', [(2,), (5, )], lambda *args: (onp.sum(args[1], axis=None) * onp.ones(2),
                                               onp.sum(args[0], axis=None) * onp.ones(5))),
        ('ijk, jil->kl', [(3, 4, 5), (4, 3, 2)], lambda *args: (onp.tile(onp.transpose(onp.sum(args[1],
                                                                                               axis=-1))[:, :, None],
                                                                         [1, 1, 5]),
                                                                onp.tile(onp.transpose(onp.sum(args[0],
                                                                                               axis=-1))[:, :, None],
                                                                         [1, 1, 2]))),
        ('ii->i', [(3, 3)], lambda *args: (onp.eye(3),)),
        ('ki, jk->ij', [(3, 2), (4, 3)], lambda *args: (onp.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                        onp.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('ki, ...k->i...', [(3, 2), (4, 3)], lambda *args: (onp.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                            onp.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('k..., jk', [(3, 2), (4, 3)], lambda *args: (onp.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                      onp.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('ij, jk', [(5, 0), (0, 4)], lambda *args: (onp.empty((5, 0)), onp.empty((0, 4)))),
        (('ij,jk,kl->il'), [(2, 2), (2, 5), (5, 2)], lambda *args: (onp.dot(onp.ones((2, 2)), onp.dot(args[1], args[2]).T),
                                                                    onp.dot(args[0].T, onp.dot(onp.ones((2, 2)), args[2].T)),
                                                                    onp.dot(onp.dot(args[0], args[1]).T, onp.ones((2, 2))))),
        # broadcast bug
        ('ij, ij -> i', [(1, 4), (2, 4)], lambda *args: (onp.sum(args[1], axis=0)[None, :],
                                                         onp.tile(args[0], [2, 1]))),
        # one dimensim bug
        ('...ij, ...jk -> ...ik', [(1, 4), (4, 2)], lambda *args: (args[1].sum(axis=1)[None, :],
                                                                   onp.tile(args[0].sum(axis=0)[: ,None], [1, 2]))),
        ('...ij, ...jk -> ...ik', [(2, 4), (4, 2)], lambda *args: (onp.tile(args[1].sum(axis=1)[None, :], [2, 1]),
                                                                   onp.tile(args[0].sum(axis=0)[: ,None], [1, 2]))),
        ('...ij, ...jk -> ...ik', [(3, 2, 1, 4), (3, 2, 4, 2)], lambda *args: (
                                                            args[1].sum(axis=3)[:, :, None, :],
                                                            onp.tile(args[0].sum(axis=2)[:, :, :, None], [1, 1, 1, 2]))),
        ('...ij, ...ik -> ...jk', [(1, 1, 1, 4), (1, 1, 1, 3)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, :, None], [1, 1, 1, 4]),
                                                            onp.tile(args[0].sum(axis=3)[:, :, : ,None], [1, 1, 1, 3]))),
        ('...ij, ...jc -> ...ic', [(1, 1, 5, 3), (1, 1, 3, 2)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, None, :], [1, 1, 5, 1]),
                                                            onp.tile(args[0].sum(axis=2)[:, :, : ,None], [1, 1, 1, 2]))),
        ('...ij, ...jc -> ...ic', [(1, 2, 5, 4), (1, 2, 4, 2)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, None, :], [1, 1, 5, 1]),
                                                            onp.tile(args[0].sum(axis=2)[:, :, : ,None], [1, 1, 1, 2]))),
        ('...ij, ...jc -> ...ic', [(2, 1, 5, 4), (2, 1, 4, 2)], lambda *args: (
                                                            onp.tile(args[1].sum(axis=3)[:, :, None, :], [1, 1, 5, 1]),
                                                             onp.tile(args[0].sum(axis=2)[:, :, : ,None], [1, 1, 1, 2]))),
        # issue #16576
        # commented due to long running time
        # ('abiz,abjz->abij', [(64, 8, 128, 512), (64, 8, 128, 512)], lambda *args: (onp.matmul(onp.ones((64, 8, 128, 128)), args[1]),
        #                                                                            onp.matmul(onp.ones((64, 8, 128, 128)), args[0]))),
    ]
    dtypes = ['float32', 'float64', 'int32']
    acc_type = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64',
                'int32': 'int64'}
    for hybridize in [False, True]:
        for dtype in dtypes:
            for config in configs:
                for optimize in [False, True]:
                    rtol = 1e-2 if dtype == 'float16' else 1e-3
                    atol = 1e-4 if dtype == 'float16' else 1e-5
                    (subscripts, operands, get_grad) = config
                    test_einsum = TestEinsum(subscripts, optimize)
                    if hybridize:
                        test_einsum.hybridize()
                    x = []
                    x_np = []
                    for shape in operands:
                        tmp = onp.array(onp.random.uniform(-1.0, 1.0, shape), dtype=dtype)
                        x_np.append(tmp.astype(acc_type[dtype]))
                        x.append(np.array(tmp, dtype=dtype))
                        x[-1].attach_grad()
                    expected_np = onp.einsum(subscripts, *x_np, optimize=optimize).astype(dtype)
                    with mx.autograd.record():
                        out_mx = test_einsum(*x)
                    assert out_mx.shape == expected_np.shape
                    assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)
                    out_mx.backward()
                    for (iop, op) in enumerate(x):
                        assert_almost_equal(op.grad.asnumpy(), get_grad(*x_np)[iop], rtol=rtol, atol=atol)

                    # Test imperative once again
                    for op in x:
                        op.attach_grad()
                    with mx.autograd.record():
                        out_mx = np.einsum(subscripts, *x, optimize=optimize)
                    out_mx.backward()
                    expected_np = onp.einsum(subscripts, *x_np, optimize=optimize)
                    assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)
                    for (iop, op) in enumerate(x):
                        assert_almost_equal(op.grad.asnumpy(), get_grad(*x_np)[iop].astype(dtype), rtol=rtol, atol=atol)
    configs = [
        (('ij,jk,kl->il'), [(2, 2), (2, 5), (5, 2)]),
        (('ea,fb,abcd,gc,hd->efgh'), [(5, 5), (5, 5), (5, 5, 5, 5), (5, 5), (5, 5)]),
    ]
    dtypes = ['int32', 'float32', 'float64']
    for hybridize in [False, True]:
        for dtype in dtypes:
            for config in configs:
                (subscripts, operands) = config
                rtol = 1e-2 if dtype == 'float16' else 1e-3
                atol = 1e-3 if dtype == 'float16' else 1e-4
                grad = []
                x_np = []
                for shape in operands:
                    x_np.append(onp.array(onp.random.uniform(-2.0, 2.0, shape),
                                          dtype=dtype))
                for optimize in [False, True]:
                    x = []
                    for iop in range(len(operands)):
                        x.append(np.array(x_np[iop], dtype=dtype))
                        x[-1].attach_grad()
                    test_einsum = TestEinsum(subscripts, optimize)
                    if hybridize:
                        test_einsum.hybridize()
                    expected_np = onp.einsum(subscripts, *[op.astype(acc_type[dtype]) for op in x_np],
                                             optimize=optimize).astype(dtype)
                    with mx.autograd.record():
                        out_mx = test_einsum(*x)
                    assert out_mx.shape == expected_np.shape
                    assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)
                    out_mx.backward()
                    cur_grad = []
                    for op in x:
                        cur_grad.append(op.grad.asnumpy())
                    grad.append(cur_grad)
                for iop in range(len(grad[0])):
                    assert_almost_equal(grad[0][iop], grad[1][iop], rtol=rtol, atol=atol)


@use_np
@pytest.mark.skip(reason='Skipped as the test is flaky and the feature causes curand error. Tracked in #18100')
def test_np_diagflat():
    class TestDiagflat(HybridBlock):
        def __init__(self, k=0):
            super(TestDiagflat,self).__init__()
            self._k = k
        def forward(self, a):
            return np.diagflat(a, k=self._k)
    shapes = [(2,),5 , (1,5), (2,2), (2,5), (3,3), (4,3),(4,4,5)] # test_shapes, remember to include zero-dim shape and zero-size shapes
    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64] # remember to include all meaningful data types for the operator
    range_k = 6
    for hybridize,shape,dtype, in itertools.product([False,True],shapes,dtypes):
        rtol = 1e-2 if dtype == np.float16 else 1e-3
        atol = 1e-4 if dtype == np.float16 else 1e-5

        for k in range(-range_k,range_k):
            test_diagflat = TestDiagflat(k)
            if hybridize:
                test_diagflat.hybridize()

            x = np.random.uniform(-1.0,1.0, size = shape).astype(dtype)
            x.attach_grad()

            np_out = onp.diagflat(x.asnumpy(), k)
            with mx.autograd.record():
                mx_out = test_diagflat(x)

            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(),np_out,rtol = rtol, atol = atol)

            mx_out.backward()
            # Code to get the reference backward value
            np_backward = np.ones(shape)
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

            # Test imperative once again
            mx_out = np.diagflat(x, k)
            np_out = onp.diagflat(x.asnumpy(), k)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_pad():
    class TestPad(HybridBlock):
        def __init__(self, pad_width, mode='constant'):
            super(TestPad,self).__init__()
            self._pad_width = pad_width
            self._mode = mode
        def forward(self, A, **kwargs):
            return np.pad(A, self._pad_width, mode=self._mode, **kwargs)

    shapes = [6, (1,5), (2,2), (2,2), (3,3), (2,3), (3,4,5)]
    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64]
    mode = ['constant', 'reflect', 'symmetric', 'edge', 'minimum', 'maximum']
    for hybridize, shape, dtype, in itertools.product([False,True], shapes, dtypes):
        rtol = 1e-2 if dtype == np.float16 else 1e-3
        atol = 1e-4 if dtype == np.float16 else 1e-5

        for m in mode:
            x = np.random.uniform(-1.0, 1.0, size = shape).astype(dtype)
            pw = ()
            if (type(shape) == int):
                pw += (2,3)
            else:
                for _ in range(len(shape)):
                    pw += ((2,3),)
            test_pad = TestPad(pw, m)
            if hybridize:
                test_pad.hybridize()
            x.attach_grad()

            if(m != 'constant'):
                np_out = onp.pad(x.asnumpy(), pw, mode=m)
            else:
                np_out = onp.pad(x.asnumpy(), pw, mode=m, constant_values=0)
            with mx.autograd.record():
                mx_out = test_pad(x)

            # code to get the reference value
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol = rtol, atol = atol)

            # test gradient
            if m == "constant":
                device = mx.device.current_device()
                x = mx.np.random.uniform(-1.0, 1.0, size=shape)
                x = mx.np.array(x, device=device)
                for grad_req in ['write', 'add']:
                    x.attach_grad(grad_req)
                    if grad_req == 'add':
                        init_grad = mx.np.random.uniform(-1.0, 1.0, size=shape, device=device)
                        x.grad[:] = init_grad
                    with mx.autograd.record():
                        mx_out = mx.np.pad(x, pad_width=pw, mode="constant")
                        out_grad = mx.np.random.normal(0, 1, mx_out.shape)
                        out_grad = mx.np.array(out_grad, device=device)
                        loss = mx_out * out_grad
                        loss = loss.sum()
                        loss.backward()
                    gt_in_grad = mx.np.pad(mx.np.ones_like(x.grad), pad_width=pw, mode="constant") * mx.np.array(out_grad, device=device)
                    mx_grad = x.grad
                    if grad_req == 'add':
                        assert_almost_equal(mx.np.pad(mx_grad - init_grad, pad_width=pw, mode="constant"), gt_in_grad.asnumpy(), rtol=rtol, atol=atol)
                    else:
                        assert_almost_equal(mx.np.pad(mx_grad, pad_width=pw, mode="constant"), gt_in_grad.asnumpy(), rtol=rtol, atol=atol)


@use_np
def test_np_rand():
    # Test shapes.
    shapes = [
        (3, 3),
        (3, 4),
        (0, 0),
        (3, 3, 3),
        (0, 0, 0),
        (2, 2, 4, 3),
        (2, 2, 4, 3),
        (2, 0, 3, 0),
        (2, 0, 2, 3)
    ]
    dtypes = ['float16', 'float32', 'float64']
    for dtype in dtypes:
        for shape in shapes:
            data_mx = np.random.rand(*shape, dtype=dtype)
            assert data_mx.shape == shape

    # Test random generator.
    device = mx.device.current_device()
    samples = 1000000
    trials = 8
    num_buckets = 10
    lower = 0.0
    upper = 1.0
    for dtype in ['float16', 'float32', 'float64']:
        buckets, probs = gen_buckets_probs_with_ppf(
            lambda x: ss.uniform.ppf(x, lower, upper), num_buckets)
        # Quantize bucket boundaries to reflect the actual dtype
        # and adjust probs accordingly
        buckets = np.array(buckets, dtype=dtype).tolist()
        probs = [(ss.uniform.cdf(buckets[i][1], lower, upper) -
                  ss.uniform.cdf(buckets[i][0], lower, upper))
                 for i in range(num_buckets)]

        def generator_mx(x): return np.random.rand(
            samples, device=device, dtype=dtype).asnumpy()
        verify_generator(generator=generator_mx, buckets=buckets,
                         probs=probs, nsamples=samples, nrepeat=trials)
        generator_mx_same_seed =\
            lambda x: onp.concatenate(
                [np.random.rand(x // 10, device=device, dtype=dtype).asnumpy()
                    for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets,
                         probs=probs, nsamples=samples, nrepeat=trials)


@use_np
def test_np_true_divide():
    shapes = [
        [()],
        [(0,)],
        [(2, 0, 3)],
        [(0, 0, 0)],
        [(10,)],
        [(3, 4)],
        [(2, 3, 4)],
        [(2, 3, 4, 5)],
        [(2, 3, 4, 5, 6)],
        [(0,), (0,)],
        [(0,), (1,)],
        [(2, 0, 3), (1, 1)],
        [(), (2, 3)],
        [(2, 3), ()],
        [(2, 3, 1), (1, 4)],
        [(2, 1, 4, 1), (3, 1, 5)],
    ]
    dtypes = [np.bool, np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64]
    itypes = [np.bool, np.int8, np.uint8, np.int32, np.int64]
    ftypes = [np.float16, np.float32, np.float64]
    for shape_pair, dtype in itertools.product(shapes, dtypes):
        a = np.random.uniform(3, 50, size=shape_pair[0]).astype(dtype)
        b = np.random.uniform(3, 50, size=shape_pair[-1]).astype(dtype)
        out_mx = a / b
        if onp.issubdtype(dtype, onp.integer) or (dtype is np.bool):
            assert out_mx.dtype == np.float32
        else:
            assert out_mx.dtype == dtype
        out_np = onp.true_divide(a.asnumpy(), b.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

        val = onp.random.randint(3, 50)
        out_mx = a / val
        out_np = onp.true_divide(a.asnumpy(), val)
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

        out_mx = val / a
        out_np = onp.true_divide(val, a.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

    for shape_pair, itype, ftype in itertools.product(shapes, itypes, ftypes):
        i_ = np.random.uniform(3, 50, size=shape_pair[0]).astype(itype)
        f_ = np.random.uniform(3, 50, size=shape_pair[-1]).astype(ftype)

        out_mx = i_ / f_
        assert out_mx.dtype == ftype
        out_np = onp.true_divide(i_.asnumpy(), f_.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

        out_mx = f_ / i_
        assert out_mx.dtype == ftype
        out_np = onp.true_divide(f_.asnumpy(), i_.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)


@use_np
def test_np_column_stack():
    class TestColumnStack(HybridBlock):
        def __init__(self):
            super(TestColumnStack, self).__init__()

        def forward(self, a, *args):
            return np.column_stack([a] + list(args))

    def g(data):
        return onp.ones_like(data)

    configs = [
        ((), (), ()),
        ((2), (2), (2)),
        ((0), (0), (0)),
        ((0, 3, 0), (0, 0, 0), (0, 1, 0)),
        ((2, 2), (2, 1), (2, 3)),
        ((4, 3), (4, 0), (4, 1)),
        ((2, 2, 2), (2, 4, 2), (2, 2, 2)),
        ((0, 1, 1), (0, 1, 1), (0, 1, 1))
    ]
    types = ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']
    for config, hybridize, dtype in itertools.product(configs, [True, False], types):
        test_column_stack = TestColumnStack()
        if hybridize:
            test_column_stack.hybridize()
        rtol = 1e-3
        atol = 1e-5
        v = []
        v_np = []
        for i in range(3):
            v_np.append(onp.array(onp.random.uniform(-10.0, 10.0, config[i]), dtype=dtype))
            v.append(mx.nd.array(v_np[i]).as_np_ndarray())
            v[i].attach_grad()
        expected_np = onp.column_stack(v_np)
        with mx.autograd.record():
            mx_out = test_column_stack(*v)
        assert mx_out.shape == expected_np.shape
        assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)

        # Test gradient
        mx_out.backward()
        for i in range(3):
            expected_grad = g(v_np[i])
            assert_almost_equal(v[i].grad.asnumpy(), expected_grad, rtol=rtol, atol=atol)

        # Test imperative once again
        mx_out = np.column_stack(v)
        expected_np = onp.column_stack(v_np)
        assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


def test_npx_reshape():
    class TestNumpyXReshape(HybridBlock):
        def __init__(self, newshape, reverse):
            super(TestNumpyXReshape, self).__init__()
            self._newshape = newshape
            self._reverse = reverse

        def forward(self, a, *args, **kwargs):
            return npx.reshape(a, self._newshape, reverse=self._reverse)

    test_cases = [
        [(2, 3, 5, 5),  (-2, -1),         False, (2, 75)],
        [(2, 3, 5, 5),  (-2, -2, -1),     False, (2, 3, 25)],
        [(5, 3, 4, 5),  (-2, -1, -2),     False, (5, 15, 4)],
        [(2, 3, 5, 4),  (-1, -2, -2),     False, (8, 3, 5)],
        [(2, 3, 5, 5),  (-2, -2, -2, -2), False, (2, 3, 5, 5)],
        [(2, 1, 4, 5),  (-2, -3, -2, -2), False, (2, 4, 5)],
        [(1, 1, 4, 1),  (-3, -3, -2, -2), False, (4, 1)],
        [(1, 1, 1, 1),  (-3, -3, -3, -3), False, ()],
        [(2, 4, 5, 3),  (-1, 2, 2, 1),    False, (30, 2, 2, 1)],
        [(2, 3, 5, 6),  (-4,),            False, (2, 3, 5, 6)],
        [(2, 3, 5, 6),  (6, 1, -4),       False, (6, 1, 5, 6)],
        [(2, 3, 5, 6),  (-5, -5),         False, (6, 30)],
        [(2, 3, 5, 6),  (-5, -1),         False, (6, 30)],
        [(64,),         (-6, 16, 4),      False, (16, 4)],
        [(64,),         (-6, 16, -1),     False, (16, 4)],
        [(64, 1, 2, 3), (-6, 16, -1, -4), False, (16, 4, 1, 2, 3)],
        [(8, 5, 4, 6),  (-4, -1, 3, -6),  True,  (8, 5, 4, 2, 3)]
    ]
    for hybridize in [True, False]:
        for shape, newshape, reverse, expected_ret_shape in test_cases:
            for grad_req in ['write', 'add']:
                # test gluon
                test_reshape = TestNumpyXReshape(newshape=newshape, reverse=reverse)
                if hybridize:
                    test_reshape.hybridize()

                a = mx.np.random.uniform(-1, 1, shape).astype(np.float32)
                init_a_grad = mx.np.random.uniform(-1, 1, shape).astype(np.float32)
                a.attach_grad(grad_req=grad_req)
                if grad_req == 'add':
                    a.grad[:] = init_a_grad
                with mx.autograd.record():
                    y = test_reshape(a)
                assert y.shape == expected_ret_shape,\
                    'y.shape={}, expected_ret_shape={}'.format(y.shape, expected_ret_shape)
                assert_almost_equal(y.asnumpy(), a.asnumpy().reshape(expected_ret_shape), rtol=1e-3, atol=1e-5)

                # test backward
                mx.autograd.backward(y)
                expected_grad = onp.ones(shape)
                if grad_req == 'add':
                    expected_grad += init_a_grad.asnumpy()
                assert_almost_equal(a.grad.asnumpy(), expected_grad, rtol=1e-3, atol=1e-5)

                # test imperative
                npx_out = npx.reshape(a, newshape, reverse=reverse)
                expected_out = onp.reshape(a.asnumpy(), expected_ret_shape)
                assert_almost_equal(npx_out.asnumpy(), expected_out, rtol=1e-3, atol=1e-5)


@use_np
def test_np_share_memory():
    ops = [np.shares_memory, np.may_share_memory]
    # reshape not support boolean types
    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64]
    for op in ops:
        for dt in dtypes:
            x = np.zeros([13, 21, 23, 22], dtype=dt)
            assert not op(x[0,:,:,:], x[1,:,:,:])
            assert not op(x[2,:,:,:], x[3,:,:,:])
            assert not op(x[2:5,0,0,0], x[3:4,0,0,0])
            assert not op(x[2:5,0,0,0], x[4:7,0,0,0])
            assert op(x[0,0,0,2:5], x[0,0,0,3:4])
            assert op(x[0,6,0,2:5], x[0,6,0,4:7])
            assert not op(x[0,5,0,2:5], x[0,6,0,4:7])

            for adt in dtypes:
                assert not op(x, np.ones((5, 0), dtype=adt))
                assert not op(np.ones((5, 0), dtype=adt), x)
                assert not op(np.ones((5, 0), dtype=dt), np.ones((0, 3, 0), dtype=adt))


def test_np_median():
    class TestMedian(HybridBlock):
        def __init__(self, axis=None, keepdims=False):
            super(TestMedian, self).__init__()
            self._axis = axis
            self._keepdims = keepdims

        def forward(self, a):
            return np.median(a, axis=self._axis, keepdims=self._keepdims)

    flags = [True, False]
    dtypes = ['float16', 'float32', 'float64']
    qtypes = ['float32', 'float64']
    tensor_shapes = [
        ((2, 3), None),
        ((2, 3, 4, 5), 3),
        ((2, 3, 4), (0, 2)),
        ((2, 3, 4), 1)
    ]

    for hybridize, keepdims, (a_shape, axis), dtype in \
        itertools.product(flags, flags, tensor_shapes, dtypes):
        atol = 3e-4 if dtype == 'float16' else 1e-4
        rtol = 3e-2 if dtype == 'float16' else 1e-2
        test_median = TestMedian(axis=axis, keepdims=keepdims)
        if hybridize:
            test_median.hybridize()
        a = np.random.uniform(-1.0, 1.0, size=a_shape)
        np_out = onp.median(a.asnumpy(), axis=axis, keepdims=keepdims)
        mx_out = test_median(a)

        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)

        mx_out = np.median(a, axis=axis, keepdims=keepdims)
        np_out = onp.median(a.asnumpy(), axis=axis, keepdims=keepdims)

        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)


@use_np
def test_np_quantile():
    class TestQuantile(HybridBlock):
        def __init__(self, axis=None, interpolation='linear', keepdims=False):
            super(TestQuantile, self).__init__()
            self._axis = axis
            self._interpolation = interpolation
            self._keepdims = keepdims

        def forward(self, a, q):
            return np.quantile(a, q, axis=self._axis, interpolation=self._interpolation, keepdims=self._keepdims)

    class TestQuantileScalar(HybridBlock):
        def __init__(self, q=None, axis=None, interpolation='linear', keepdims=False):
            super(TestQuantileScalar, self).__init__()
            self._q = q
            self._axis = axis
            self._interpolation = interpolation
            self._keepdims = keepdims

        def forward(self, a):
            return np.quantile(a, self._q, axis=self._axis, interpolation=self._interpolation, keepdims=self._keepdims)

    flags = [True, False]
    interpolation_options = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
    dtypes = [np.int32, np.int64, np.float16, np.float32, np.float64]
    qtypes = [np.float32, np.float64]
    tensor_shapes = [
        ((2, 3), (), None),
        ((2, 3, 4, 5), (), 3),
        ((2, 3, 4), (3,), (0, 2)),
        ((2, 3, 4), (3,), 1)
    ]
    for hybridize, keepdims, q_scalar, (a_shape, q_shape, axis), interpolation, dtype in \
        itertools.product(flags, flags, flags, tensor_shapes, interpolation_options, dtypes):
        if dtype == np.float16 and interpolation == 'linear': continue
        atol = 3e-4 if dtype == np.float16 else 1e-4
        rtol = 3e-2 if dtype == np.float16 else 1e-2
        a = np.random.uniform(-10.0, 10.0, size=a_shape).astype(dtype)
        qtype = random.choice(qtypes)
        q = np.random.uniform(0, 1.0, size=q_shape).astype(qtype)
        np_q = q.asnumpy()
        if q_scalar and q_shape == ():
            q = q.item()
            np_q = q
            test_quantile = TestQuantileScalar(q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        else:
            test_quantile = TestQuantile(axis=axis, interpolation=interpolation, keepdims=keepdims)
        if hybridize:
            test_quantile.hybridize()
        mx_out = test_quantile(a) if (q_scalar and q_shape == ()) else test_quantile(a, q)
        np_out = onp.quantile(a.asnumpy(), np_q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)

        mx_out = np.quantile(a, q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        np_out = onp.quantile(a.asnumpy(), np_q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)


@use_np
def test_np_percentile():
    class TestPercentile(HybridBlock):
        def __init__(self, axis=None, interpolation='linear', keepdims=False):
            super(TestPercentile, self).__init__()
            self._axis = axis
            self._interpolation = interpolation
            self._keepdims = keepdims

        def forward(self, a, q):
            return np.percentile(a, q, axis=self._axis, interpolation=self._interpolation, keepdims=self._keepdims)

    class TestPercentileScalar(HybridBlock):
        def __init__(self, q=None, axis=None, interpolation='linear', keepdims=False):
            super(TestPercentileScalar, self).__init__()
            self._q = q
            self._axis = axis
            self._interpolation = interpolation
            self._keepdims = keepdims

        def forward(self, a):
            return np.percentile(a, self._q, axis=self._axis, interpolation=self._interpolation, keepdims=self._keepdims)

    flags = [True, False]
    interpolation_options = ['linear', 'lower', 'higher', 'nearest', 'midpoint']
    dtypes = [np.int32, np.int64, np.float16, np.float32, np.float64]
    qtypes = [np.float32, np.float64]
    tensor_shapes = [
        ((2, 3), (), None),
        ((2, 3, 4, 5), (), 3),
        ((2, 3, 4, 5), (), (0, 1, 2)),
        ((2, 3, 4, 5), (), (-1, -2)),
        ((2, 3, 4), (3,), (0, 2)),
        ((2, 3, 4), (3,), 1)
    ]
    for hybridize, keepdims, q_scalar, (a_shape, q_shape, axis), interpolation, dtype in \
        itertools.product(flags, flags, flags, tensor_shapes, interpolation_options, dtypes):
        if dtype == np.float16 and interpolation == 'linear': continue
        atol = 3e-4 if dtype == np.float16 else 1e-4
        rtol = 3e-2 if dtype == np.float16 else 1e-2
        a = np.random.uniform(-10.0, 10.0, size=a_shape).astype(dtype)
        qtype = random.choice(qtypes)
        q = np.random.uniform(0, 1.0, size=q_shape).astype(qtype)
        np_q = q.asnumpy()
        if q_scalar and q_shape == ():
            q = q.item()
            np_q = q
            test_percentile = TestPercentileScalar(q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        else:
            test_percentile = TestPercentile(axis=axis, interpolation=interpolation, keepdims=keepdims)
        if hybridize:
            test_percentile.hybridize()
        mx_out = test_percentile(a) if (q_scalar and q_shape == ()) else test_percentile(a, q)
        np_out = onp.percentile(a.asnumpy(), np_q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)

        mx_out = np.percentile(a, q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        np_out = onp.percentile(a.asnumpy(), np_q, axis=axis, interpolation=interpolation, keepdims=keepdims)
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)


@use_np
def test_np_diff():
    def np_diff_backward(ograd, n, axis):
        res = ograd
        for _ in range(n):
            res = onp.negative(onp.diff(res, n=1, axis=axis, prepend=0, append=0))
        return res

    class TestDiff(HybridBlock):
        def __init__(self, n=1, axis=-1):
            super(TestDiff, self).__init__()
            self._n = n
            self._axis = axis

        def forward(self, a):
            return np.diff(a, n=self._n, axis=self._axis)

    shapes = [tuple(random.randrange(10) for i in range(random.randrange(6))) for j in range(5)]
    for hybridize in [True, False]:
        for shape in shapes:
            for axis in [i for i in range(-len(shape), len(shape))]:
                for n in [i for i in range(0, shape[axis]+1)]:
                    test_np_diff = TestDiff(n=n, axis=axis)
                    if hybridize:
                        test_np_diff.hybridize()
                    for itype in [onp.float16, onp.float32, onp.float64]:
                        # note the tolerance shall be scaled by the input n
                        if itype == onp.float16:
                            rtol = atol = 1e-2*len(shape)*n
                        else:
                            rtol = atol = 1e-5*len(shape)*n
                        x = rand_ndarray(shape).astype(itype).as_np_ndarray()
                        x.attach_grad()
                        np_out = onp.diff(x.asnumpy(), n=n, axis=axis)
                        with mx.autograd.record():
                            mx_out = test_np_diff(x)
                        assert mx_out.shape == np_out.shape
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                        mx_out.backward()
                        if (np_out.size == 0):
                            np_backward = onp.zeros(shape)
                        else:
                            np_backward = np_diff_backward(onp.ones(np_out.shape, dtype=itype), n=n, axis=axis)
                        assert x.grad.shape == np_backward.shape
                        assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

                        mx_out = np.diff(x, n=n, axis=axis)
                        np_out = onp.diff(x.asnumpy(), n=n, axis=axis)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_ediff1d():
    def np_diff_backward(size, shape):
        if size <= 1:
            return onp.zeros(shape)
        else:
            ret = onp.ones(size - 1)
            return onp.negative(onp.diff(ret, n=1, axis=-1, prepend=0, append=0)).reshape(shape)

    # case 1: when both `to_begin` and `to_end` are arrays
    class TestEDiff1DCASE1(HybridBlock):
        def __init__(self):
            super(TestEDiff1DCASE1, self).__init__()

        def forward(self, a, b, c):
            return np.ediff1d(a, to_end=b, to_begin=c)

    # case 2: only `to_end` is array but `to_begin` is scalar/None
    class TestEDiff1DCASE2(HybridBlock):
        def __init__(self, to_begin=None):
            super(TestEDiff1DCASE2, self).__init__()
            self._to_begin = to_begin

        def forward(self, a, b):
            return np.ediff1d(a, to_end=b, to_begin=self._to_begin)

    # case 3: only `to_begin` is array but `to_end` is scalar/None
    class TestEDiff1DCASE3(HybridBlock):
        def __init__(self, to_end=None):
            super(TestEDiff1DCASE3, self).__init__()
            self._to_end = to_end

        def forward(self, a, b):
            return np.ediff1d(a, to_end=self._to_end, to_begin=b)

    # case 4: both `to_begin` and `to_end` are scalar/None
    class TestEDiff1DCASE4(HybridBlock):
        def __init__(self, to_end=None, to_begin=None):
            super(TestEDiff1DCASE4, self).__init__()
            self._to_begin = to_begin
            self._to_end = to_end

        def forward(self, a):
            return np.ediff1d(a, to_end=self._to_end, to_begin=self._to_begin)

    rtol = 1e-3
    atol = 1e-5
    mapper = {(True, True): TestEDiff1DCASE1,
              (False, True): TestEDiff1DCASE2,
              (True, False): TestEDiff1DCASE3,
              (False, False): TestEDiff1DCASE4}
    hybridize_list = [True, False]
    shape_list = [(), (1,), (2, 3), 6, (7, 8), 10, (4, 0, 5)]
    # dtype_list = [np.int32, np.int64, np.float16, np.float32, np.float64]
    dtype_list = [np.float16, np.float32, np.float64]
    append_list = [1, 2, None, (1, 2, 4), (4, 3), (), (5, 0), (6)]

    for hybridize, dtype, shape, to_begin, to_end in itertools.product(hybridize_list, dtype_list,
                shape_list, append_list, append_list):
        mx_arr = np.random.randint(5, size=shape).astype(dtype)
        np_arr = mx_arr.asnumpy()
        kwargs = {}
        mx_args = [mx_arr]
        np_args = [np_arr]
        mx_args_imperative = [mx_arr]

        if isinstance(to_end, tuple):
            to_end = np.random.randint(5, size=to_end).astype(dtype)
            mx_args.append(to_end)
            np_args.append(to_end.asnumpy())
        else:
            kwargs["to_end"] = to_end
            np_args.append(to_end)
        mx_args_imperative.append(to_end)

        if isinstance(to_begin, tuple):
            to_begin = np.random.randint(5, size=to_begin).astype(dtype)
            mx_args.append(to_begin)
            np_args.append(to_begin.asnumpy())
        else:
            kwargs["to_begin"] = to_begin
            np_args.append(to_begin)
        mx_args_imperative.append(to_begin)

        from mxnet.numpy import ndarray as np_ndarray
        input_type = (isinstance(to_begin, np_ndarray), isinstance(to_end, np_ndarray))
        test_np_ediff1d = mapper[input_type](**kwargs)

        if hybridize:
            test_np_ediff1d.hybridize()

        np_out = onp.ediff1d(*np_args)
        for arg in mx_args:
            arg.attach_grad()

        with mx.autograd.record():
            mx_out = test_np_ediff1d(*mx_args)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)
        # test imperative
        mx_out_imperative = np.ediff1d(*mx_args_imperative)
        assert mx_out_imperative.shape == np_out.shape
        assert_almost_equal(mx_out_imperative.asnumpy(), np_out, atol=atol, rtol=rtol)

        mx_out.backward()
        if dtype in [np.float16, np.float32, np.float64]:
            for idx, arg in enumerate(mx_args):
                if idx == 0:
                    assert_almost_equal(arg.grad.asnumpy(), np_diff_backward(arg.size, arg.shape), atol=atol, rtol=rtol)
                else:
                    assert_almost_equal(arg.grad.asnumpy(), np.ones_like(arg), atol=atol, rtol=rtol)


@use_np
def test_np_column_stack():
    class TestColumnStack(HybridBlock):
        def __init__(self):
            super(TestColumnStack, self).__init__()

        def forward(self, a, *args):
            return np.column_stack([a] + list(args))

    def g(data):
        return onp.ones_like(data)

    configs = [
        ((), (), ()),
        ((2), (2), (2)),
        ((0), (0), (0)),
        ((0, 3, 0), (0, 0, 0), (0, 1, 0)),
        ((2, 2), (2, 1), (2, 3)),
        ((4, 3), (4, 0), (4, 1)),
        ((2, 2, 2), (2, 4, 2), (2, 2, 2)),
        ((0, 1, 1), (0, 1, 1), (0, 1, 1))
    ]
    types = ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']
    for config, hybridize, dtype in itertools.product(configs, [True, False], types):
        test_column_stack = TestColumnStack()
        if hybridize:
            test_column_stack.hybridize()
        rtol = 1e-3
        atol = 1e-5
        v = []
        v_np = []
        for i in range(3):
            v_np.append(onp.array(onp.random.uniform(-10.0, 10.0, config[i]), dtype=dtype))
            v.append(mx.nd.array(v_np[i]).as_np_ndarray())
            v[i].attach_grad()
        expected_np = onp.column_stack(v_np)
        with mx.autograd.record():
            mx_out = test_column_stack(*v)
        assert mx_out.shape == expected_np.shape
        assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)

        # Test gradient
        mx_out.backward()
        for i in range(3):
            expected_grad = g(v_np[i])
            assert_almost_equal(v[i].grad.asnumpy(), expected_grad, rtol=rtol, atol=atol)

        # Test imperative once again
        mx_out = np.column_stack(v)
        expected_np = onp.column_stack(v_np)
        assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


@use_np
@pytest.mark.skip(reason='Test hangs. Tracked in #18144')
def test_np_resize():
    class TestResize(HybridBlock):
        def __init__(self, new_shape):
            super(TestResize, self).__init__()
            self._new_shape = new_shape

        def forward(self, x, *args, **kwargs):
            return np.resize(x, self._new_shape)

    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool_]
    shape_config = [
        [(), (2, 3)],
        [(2, 3), (2,)],
        [(2, 3), 2],
        [(2, 0, 1), (2, 2)],
        [(2, 0, 1), (3, 4, 5)],
        [((1,)), ()],
    ]
    flags = [True, False]
    for dtype, shape_pair, hybridize in itertools.product(dtypes, shape_config, flags):
        a = np.random.uniform(low=0, high=100, size=shape_pair[0], dtype='float64').astype(dtype)
        test = TestResize(shape_pair[1])
        if hybridize:
            test.hybridize()
        ret = test(a)
        expected_ret = onp.resize(a.asnumpy(), shape_pair[1])
        assert_almost_equal(ret.asnumpy(), expected_ret, atol=1e-5, rtol=1e-5, use_broadcast=False)

        # check imperative again
        ret = np.resize(a, shape_pair[1])
        assert_almost_equal(ret.asnumpy(), expected_ret, atol=1e-5, rtol=1e-5, use_broadcast=False)


@use_np
def test_np_diag():
    class TestDiag(HybridBlock):
        def __init__(self, k=0):
            super(TestDiag, self).__init__()
            self._k = k

        def forward(self, a):
            return np.diag(a, k=self._k)

    shapes = [(), (2,), (1, 5), (2, 2), (2, 5), (3, 3), (4, 3)]
    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64]
    range_k = 6
    combination = itertools.product([False, True], shapes, dtypes, list(range(-range_k, range_k)))
    for hybridize, shape, dtype, k in combination:
        rtol = 1e-2 if dtype == np.float16 else 1e-3
        atol = 1e-4 if dtype == np.float16 else 1e-5
        test_diag = TestDiag(k)
        if hybridize:
            test_diag.hybridize()
        x = np.random.uniform(-2.0, 2.0, size=shape).astype(dtype) if len(shape) != 0 else np.array(())
        x.attach_grad()
        np_out = onp.diag(x.asnumpy(), k)
        with mx.autograd.record():
            mx_out = test_diag(x)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)

        # check backward function
        mx_out.backward()
        if len(shape) == 0:
            np_backward = np.array(())
        elif len(shape) == 1:
            np_backward = np.ones(shape[0])
        else:
            np_backward = np.zeros(shape)
            h = shape[0]
            w = shape[1]
            if k > 0:
                w -= k
            else:
                h += k
            s = min(w, h)
            if s > 0:
                if k >= 0:
                    for i in range(s):
                        np_backward[0+i][k+i] = 1
                else:
                    for i in range(s):
                        np_backward[-k+i][0+i] = 1
        assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

        # Test imperative once again
        mx_out = np.diag(x, k)
        np_out = onp.diag(x.asnumpy(), k)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
@pytest.mark.parametrize('config', [
    [(1, 5), (0, 1)], [(2, 2), (0, 1)],
    [(2, 5), (0, 1)], [(5, 5), (0, 1)],
    [(2, 2, 2), (0, 1)], [(2, 4, 4), (0, 2)],
    [(3, 3, 3), (1, 2)], [(4, 8, 8), (1, 2)],
    [(4, 4, 4, 4), (1, 2)], [(5, 6, 7, 8), (2, 3)],
    [(6, 7, 8, 9, 10), (3, 4)]
])
@pytest.mark.parametrize('k', [0, 2, 4, 6])
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('call_by_instance', [True, False])
def test_np_diagonal(config, k, dtype, hybridize, call_by_instance):
    class TestDiagonal(HybridBlock):
        def __init__(self, k=0, axis1=0, axis2=1, call_by_instance=False):
            super(TestDiagonal, self).__init__()
            self._k = k
            self._axis1 = axis1
            self._axis2 = axis2
            self._call_by_instance = call_by_instance

        def forward(self, a):
            if self._call_by_instance:
                return a.diagonal(self._k, self._axis1, self._axis2)
            else:
                return np.diagonal(a, self._k, self._axis1, self._axis2)

    rtol = 1e-2 if dtype == np.float16 else 1e-3
    atol = 1e-4 if dtype == np.float16 else 1e-5
    shape, (axis1, axis2) = config
    x = np.random.uniform(-5.0, 5.0, size=shape).astype(dtype)
    x.attach_grad()
    test_diagonal = TestDiagonal(k, axis1, axis2, call_by_instance)
    if hybridize:
        test_diagonal.hybridize()

    if call_by_instance:
        np_out = x.asnumpy().diagonal(offset=k, axis1=axis1, axis2=axis2)
    else:
        np_out = onp.diagonal(x.asnumpy(), offset=k, axis1=axis1, axis2=axis2)
    with mx.autograd.record():
        mx_out = test_diagonal(x)
    assert mx_out.shape == np_out.shape
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)

    # check backward function
    mx_out.backward()
    size_out = np_out.size
    shape_out = np_out.shape
    ndim = len(shape)
    h = shape[axis1]
    w = shape[axis2]
    np_backward_slice = onp.zeros((h, w))
    np_backward = onp.zeros(shape)
    if k > 0:
        w -= k
    else:
        h += k
    s = min(w, h)
    if s > 0:
        if k >= 0:
            for i in range(s):
                np_backward_slice[0+i][k+i] = 1
        else:
            for i in range(s):
                np_backward_slice[-k+i][0+i] = 1
        ileading = int(size_out/s)
        array_temp = onp.array([np_backward_slice for i in range(ileading)])
        array_temp = array_temp.reshape(shape_out[:-1] + (shape[axis1], shape[axis2]))
        axis_idx = [i for i in range(ndim-2)]
        axis_idx[axis1:axis1] = [ndim - 2]
        axis_idx[axis2:axis2] = [ndim - 1]
        np_backward = onp.transpose(array_temp, tuple(axis_idx))
    assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

    # Test imperative once again
    mx_out = np.diagonal(x, k, axis1, axis2)
    np_out = onp.diagonal(x.asnumpy(), offset=k, axis1=axis1, axis2=axis2)
    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
def test_np_nan_to_num():
    def take_ele_grad(ele):
        if onp.isinf(ele) or onp.isnan(ele):
            return 0
        return 1
    def np_nan_to_num_grad(data):
        shape = data.shape
        arr = list(map(take_ele_grad,data.flatten()))
        return onp.array(arr).reshape(shape)

    class TestNanToNum(HybridBlock):
        def __init__(self, copy=True, nan=0.0, posinf=None, neginf=None):
            super(TestNanToNum, self).__init__()
            self.copy = copy
            self.nan = nan
            self.posinf = posinf
            self.neginf = neginf
            # necessary initializations

        def forward(self, a):
            return np.nan_to_num(a, self.copy, self.nan, self.posinf, self.neginf)

    src_list = [
        onp.nan,
        onp.inf,
        -onp.inf,
        1,
        [onp.nan],
        [onp.inf],
        [-onp.inf],
        [1],
        [1,2,3,4,-1,-2,-3,-4,0],
        [onp.nan, onp.inf, -onp.inf],
        [onp.nan, onp.inf, -onp.inf, -574, 0, 23425, 24234,-5],
        [onp.nan, -1, 0, 1],
        [[-433, 0, 456, onp.inf], [-1, -onp.inf, 0, 1]]
    ]

    dtype_list = ['float16', 'float32', 'float64']
    # [nan, inf, -inf]
    param_list = [[None, None, None], [0, 1000, -100], [0.0, 9999.9, -9999.9]]
    # Inplace operations are not supported when recording in deferred compute mode
    # copy_list = [True, False]
    copy_list = [True]
    hybridize_list = [True, False]
    atol, rtol = 1e-5, 1e-3

    src_dtype_comb = list(itertools.product(src_list,dtype_list))
    # check the dtype = int case in both imperative and sympolic expression
    src_dtype_comb.append((1,'int32'))
    src_dtype_comb.append(([234, 0, -40],'int64'))

    combinations = itertools.product(hybridize_list, src_dtype_comb, copy_list, param_list)

    numpy_version = onp.version.version
    for [hybridize, src_dtype, copy, param] in combinations:
        src, dtype = src_dtype
        # np.nan, np.inf, -np.int are float type
        x1 = mx.nd.array(src, dtype=dtype).as_np_ndarray().asnumpy()
        x2 = mx.nd.array(src, dtype=dtype).as_np_ndarray()
        x3 = mx.nd.array(src, dtype=dtype).as_np_ndarray()

        expected_grad = np_nan_to_num_grad(x1)
        x2.attach_grad()
        # with optional parameters or without
        if param[0] !=None and numpy_version>="1.17":
            test_np_nan_to_num = TestNanToNum(copy=copy, nan=param[0], posinf=param[1], neginf=param[2])
            np_out = onp.nan_to_num(x1, copy=copy, nan=param[0], posinf=param[1], neginf=param[2])
            mx_out = np.nan_to_num(x3, copy=copy, nan=param[0], posinf=param[1], neginf=param[2])
        else:
            test_np_nan_to_num = TestNanToNum(copy=copy)
            np_out = onp.nan_to_num(x1, copy=copy)
            mx_out = np.nan_to_num(x3, copy=copy)

        assert_almost_equal(mx_out.asnumpy(), np_out, rtol, atol)
        # check the inplace operation when copy = False
        # if x1.shape = 0, onp.array will not actually execute copy logic
        # only check x3 from np.nan_to_num instead of x2 from gluon
        if copy == False and x1.shape!=():
            assert x1.shape == x3.asnumpy().shape
            assert x1.dtype == x3.asnumpy().dtype
            assert_almost_equal(x1, x3.asnumpy(), rtol=rtol, atol=atol)
        # gluon does not support nan_to_num when copy=False
        # backward will check int type and if so, throw error
        # if not this case, test gluon
        if not (hybridize== False and copy == False) and ('float' in dtype):
            if hybridize:
                test_np_nan_to_num.hybridize()
            with mx.autograd.record():
                mx_out_gluon = test_np_nan_to_num(x2)
            assert_almost_equal(mx_out_gluon.asnumpy(), np_out, rtol, atol)
            mx_out_gluon.backward()
            assert_almost_equal(x2.grad.asnumpy(), expected_grad, rtol=1e-3, atol=1e-5)

        # Test imperative once again
        # if copy = False, the value of x1 and x2 has changed
        if copy == True:
            np_out = onp.nan_to_num(x1)
            mx_out = np.nan_to_num(x3)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@use_np
def test_np_unary_bool_funcs():
    def check_unary_func(func):
        class TestUnary(HybridBlock):
            def __init__(self, func):
                super(TestUnary, self).__init__()
                self._func = func

            def forward(self, a):
                return getattr(np, self._func)(a)

        src_list = [
            onp.nan,
            onp.inf,
            -onp.inf,
            float('inf'),
            float('-inf'),
            float("nan"),
            onp.array(0)/0,  # nan
            0.0 * onp.inf,  # nan
            onp.inf/onp.inf,  # nan
            onp.inf - onp.inf,  # nan
            onp.array(1)/0,  # inf
            0 + np.inf,  # inf
            1,
            [onp.nan],
            [onp.inf],
            [-onp.inf],
            [onp.array(0)/0],
            [-onp.array(0)/0],
            [onp.inf - onp.inf],  # nan
            [1],
            [1,2,3,4,-1,-2,-3,-4,0],
            [onp.nan, onp.inf, -onp.inf],
            [onp.nan, onp.inf, -onp.inf, -574, 0, 23425, 24234,-5],
            [onp.nan, -1, 0, 1, float('inf'), float('-inf'), float('nan')],
            [[-433, 0, 456, onp.inf], [-1, -onp.inf, 0, 1]]
        ]

        np_func = getattr(onp, func)
        mx_func = TestUnary(func)
        dtype_list = ['float16', 'float32', 'float64']
        hybridize_list = [True, False]
        atol, rtol = 1e-5, 1e-3

        for [hybridize, dtype, src] in itertools.product(hybridize_list, dtype_list, src_list):
            mx_data = mx.np.array(src, dtype=dtype)
            np_data = mx_data.asnumpy()

            if hybridize:
                mx_func.hybridize()
            with mx.autograd.record():
                mx_out= mx_func(mx_data)

            assert mx_out.dtype == np.bool_

            np_out = np_func(np_data)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol, atol)
            # test imperative
            mx_out_imperative = getattr(mx.np, func)(mx_data)
            assert_almost_equal(mx_out_imperative.asnumpy(), np_out, rtol, atol)
            # if `out` is given and dtype == np.bool
            mx_x = np.ones_like(mx_data).astype(np.bool)
            np_x = mx_x.asnumpy()
            getattr(mx.np, func)(mx_data, mx_x)
            np_func(np_data, np_x)
            assert_almost_equal(mx_out_imperative .asnumpy(), np_out, rtol, atol)
            # if `out` is given but dtype mismatches
            mx_y = np.ones_like(mx_data)
            assertRaises(TypeError, getattr(np, func), mx_data, out=mx_y)

            assertRaises(NotImplementedError, getattr(np, func), mx_data, where=False)
            assertRaises(NotImplementedError, getattr(np, func), mx_data,  subok=False)
            assertRaises(NotImplementedError, getattr(np, func), mx_data,  dtype=onp.int8)
            assertRaises(TypeError, getattr(np, func), mx_data,  dtype="abcdefg")
            assertRaises(NotImplementedError, getattr(np, func), mx_data,  casting='safe')
            assertRaises(TypeError, getattr(np, func), mx_data,  casting='mxnet')
            assertRaises(NotImplementedError, getattr(np, func), mx_data,  order='C')
            assertRaises(NotImplementedError, getattr(np, func), mx_data,  order='mxnet')

        # test special shape and dtype
        shape_list = [(), (1,), (2, 3), (4, 0, 5), 6, (7, 8), None]
        dtype_list = ['int32', 'int64', 'float16', 'float32', 'float64']
        for [hybridize, dtype, shape] in itertools.product(hybridize_list, dtype_list, shape_list):
            mx_data = mx.np.random.randint(low=-1, high=1, size=shape).astype(dtype)
            np_data = mx_data.asnumpy()

            if hybridize:
                mx_func.hybridize()
            with mx.autograd.record():
                mx_out= mx_func(mx_data)

            assert mx_out.dtype == np.bool_

            np_out = np_func(np_data)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol, atol)
            mx_out_imperative = getattr(mx.np, func)(mx_data)
            assert_almost_equal(mx_out_imperative .asnumpy(), np_out, rtol, atol)

    check_unary_func("isnan")
    check_unary_func("isinf")
    check_unary_func("isposinf")
    check_unary_func("isneginf")
    check_unary_func("isfinite")


@use_np
def test_np_polyval():
    class TestPolyval(HybridBlock):
        def __init__(self):
            super(TestPolyval, self).__init__()

        def forward(self, p, x, *args, **kwargs):
            return np.polyval(p, x)

    def polyval_grad(p, x):
        x_shape = x.shape
        x = x.reshape((x.size, 1))
        x = onp.broadcast_to(x, (x.size, p.size))
        exp = onp.arange(p.size-1, -1, -1)
        p_grad = onp.power(x, exp)
        coeff = exp-1
        coeff[-1] = 0
        x_grad = onp.power(x, coeff) * p * exp
        p_grad = onp.sum(p_grad, axis=0)
        x_grad = onp.sum(x_grad, axis=-1).reshape(x_shape)
        return (p_grad, x_grad)

    dtypes = ['float32', 'float64', 'int32', 'int64']
    x_shapes = [
        (5,),
        (10),
        (3, 3),
        (3, 4),
        (3, 3, 3),
        (2, 2, 4, 3),
        (2, 0, 2, 3)
    ]
    flags = [True, False]
    for dtype, x_shape, hybridize in itertools.product(dtypes, x_shapes, flags):
        p_shape = (random.randint(1, 8),)
        test_polyval = TestPolyval()
        if hybridize:
            test_polyval.hybridize()
        rtol = 1e-2
        atol = 1e-4
        if dtype in ['int32', 'int64']:
            p = np.random.randint(-16, 16, p_shape, dtype=dtype)
            x = np.random.randint(-5, 5, x_shape, dtype=dtype)
        else:
            p = np.random.uniform(-1.0, 1.0, size=p_shape, dtype=dtype)
            x = np.random.uniform(-1.0, 1.0, size=x_shape, dtype=dtype)

        p.attach_grad()
        x.attach_grad()
        np_out = onp.polyval(p.asnumpy(), x.asnumpy())
        with mx.autograd.record():
            mx_out = test_polyval(p, x)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)

        mx_out.backward()
        if dtype in ['float16', 'float32', 'float64']:
            p_grad, x_grad = polyval_grad(p.asnumpy(), x.asnumpy())
            assert_almost_equal(p.grad.asnumpy(), p_grad, atol=atol, rtol=rtol)
            assert_almost_equal(x.grad.asnumpy(), x_grad, atol=atol, rtol=rtol)

        mx_out = np.polyval(p, x)
        np_out = onp.polyval(p.asnumpy(), x.asnumpy())
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)


@use_np
def test_np_where():
    class TestWhere(HybridBlock):
        def __init__(self):
            super(TestWhere, self).__init__()

        def forward(self, cond, x, y):
            return np.where(cond, x, y)

    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool]
    shape_configs = [
        [(), (2, 3), (4, 1, 3)],
        [(), (4, 1, 3), (2, 3)],
        [(2, 3), (4, 1, 3), ()],
        [(4, 1, 3), (2, 3), ()],
        [(2, 3), (), (4, 1, 3)],
        [(2, 3), (2, 3), (2, 3)],
        [(2, 3), (2, 1), (2, 3)],
        [(2, 1), (2, 3), (2, 3)],
        [(2, 3), (2, 3), (2, 1)]
    ]
    flags = [True, False]
    for ctype, dtype, shape_pair, hybridize in itertools.product(dtypes, dtypes, shape_configs, flags):
        cond = np.round(np.random.uniform(low=0, high=2, size=shape_pair[0], dtype='float64')).astype(ctype)
        x = np.random.uniform(low=0, high=100, size=shape_pair[1], dtype='float64').astype(dtype)
        y = np.random.uniform(low=0, high=100, size=shape_pair[2], dtype='float64').astype(dtype)
        cond.attach_grad()
        x.attach_grad()
        y.attach_grad()
        test_mod = TestWhere()
        if hybridize:
            test_mod.hybridize()
        with mx.autograd.record():
            ret = test_mod(cond, x, y)

        assert same(ret.asnumpy(), onp.where(cond.asnumpy(), x.asnumpy(), y.asnumpy()))
        if dtype in [np.float16, np.float32, np.float64]:
            ret.backward()
            assert same(cond.grad.asnumpy(), onp.zeros(shape_pair[0], dtype=ctype))

            xgrad = x.grad.asnumpy()
            npgrad = collapse_sum_like((onp.broadcast_to(cond.asnumpy(), ret.shape) != 0).astype(dtype), shape_pair[1])
            npgrad = npgrad.astype(xgrad.dtype)
            assert same(xgrad, npgrad)

        # check imperative again
        ret = np.where(cond, x, y)
        assert same(ret.asnumpy(), onp.where(cond.asnumpy(), x.asnumpy(), y.asnumpy()))

        # check scalar case
        if dtype in [np.float16, np.float32, np.float64]:
            # lscalar
            with mx.autograd.record():
                ret_lscalar = np.where(cond, 1, x)
            assert same(ret_lscalar.asnumpy(), onp.where(cond.asnumpy(), 1, x.asnumpy()))
            ret_lscalar.backward()

            xgrad = x.grad.asnumpy()
            npgrad = collapse_sum_like((onp.broadcast_to(cond.asnumpy(), ret_lscalar.shape) == 0).astype(dtype), shape_pair[1])
            npgrad = npgrad.astype(xgrad.dtype)
            assert same(xgrad, npgrad)
            # rscalar
            with mx.autograd.record():
                ret_rscalar = np.where(cond, x, 1)
            assert same(ret_rscalar.asnumpy(), onp.where(cond.asnumpy(), x.asnumpy(), 1))
            ret_rscalar.backward()

            xgrad = x.grad.asnumpy()
            npgrad = collapse_sum_like((onp.broadcast_to(cond.asnumpy(), ret_rscalar.shape) != 0).astype(dtype), shape_pair[1])
            npgrad = npgrad.astype(xgrad.dtype)
            assert same(xgrad, npgrad)

        # check both scalar case
        x = onp.random.randint(0, 100)
        y = onp.random.randint(0, 100)
        mx_out = np.where(cond, x, y)
        np_out = onp.where(cond, x, y)
        assert same(mx_out, np_out)


@use_np
def test_np_expand_dims():
    class TestExpandDims(HybridBlock):
        def __init__(self, axis):
            super(TestExpandDims, self).__init__()
            self._axis = axis

        def forward(self, x):
            return np.expand_dims(x, self._axis)

    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool]
    shapes = [
        (),
        (0,),
        (0, 1),
        (3,),
        (1, 2, 3),
    ]
    flags = [True, False]
    for dtype, shape, hybridize in itertools.product(dtypes, shapes, flags):
        ndim = len(shape)
        for axis in range(-ndim-1, ndim+1):
            x_np = onp.random.uniform(0, 100, size=shape).astype(dtype)
            expected = onp.expand_dims(x_np, axis)
            for req in ['write', 'add']:
                test_expand_dims = TestExpandDims(axis)
                if hybridize:
                    test_expand_dims.hybridize()

                x = np.array(x_np)
                x.attach_grad(req)
                initial_grad = np.random.uniform(0, 10, size=x.shape).astype(x.dtype)
                x.grad[()] = initial_grad
                with mx.autograd.record():
                    y = test_expand_dims(x)
                y.backward()

                assert_almost_equal(y.asnumpy(), expected, use_broadcast=False)
                if req == 'null':
                    assert same(x.grad.asnumpy(), initial_grad.asnumpy())
                elif req == 'write':
                    assert same(x.grad.asnumpy(), onp.ones_like(x.asnumpy()))
                else:
                    assert_almost_equal(x.grad.asnumpy(), initial_grad.asnumpy() + onp.ones_like(initial_grad.asnumpy()),
                                        atol=1e-2 if dtype is np.float16 else 1e-4,
                                        rtol=1e-2 if dtype is np.float16 else 1e-4,
                                        use_broadcast=False)

                # check imperative again
                y = np.expand_dims(x, axis)
                assert_almost_equal(y.asnumpy(), expected, use_broadcast=False)


@use_np
@pytest.mark.parametrize('ishape', [
    2, 5,
    (), (1,), (4,),
    (2, 2), (2, 4), (3, 5),
    (2, 2, 2), (2, 3, 2), (2, 3, 4),
])
@pytest.mark.parametrize('rshape', [
    10, (15,),
    (3, 4), (4, 5),
    (2,3,4)
])
@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.int32, np.int64])
@pytest.mark.parametrize('hybridize', [True, False])
def test_np_unravel_index(ishape, rshape, dtype, hybridize):
    class TestUnravel_index(HybridBlock):
        def __init__(self, shape, order='C') :
            super(TestUnravel_index, self).__init__()
            self._shape = shape
            self._order = order

        def forward(self, a):
            return np.unravel_index(a, self._shape, self._order)


    rtol = 1e-2 if dtype == np.float16 else 1e-3
    atol = 1e-4 if dtype == np.float16 else 1e-5
    test_unravel_index = TestUnravel_index(rshape)
    if hybridize:
        test_unravel_index.hybridize()
    if type(ishape) == int and hybridize:
        x = np.array([ishape], dtype=dtype)
        np_out = onp.unravel_index(x.asnumpy(), rshape)
    else:
        x = np.random.uniform(0, 8, size=ishape).astype(dtype)
        np_out = onp.unravel_index(x.asnumpy(), rshape)
    mx_out = test_unravel_index(x)
    assert len(mx_out) == len(np_out)
    for elem_mx, elem_np in zip(mx_out, np_out):
        assert elem_mx.asnumpy().shape == elem_np.shape
        assert_almost_equal(elem_mx.asnumpy(), elem_np, rtol=rtol, atol=atol)
    # no backward function for unravel_index operator

    # Test imperative once again
    mx_out = np.unravel_index(x, rshape)
    np_out = onp.unravel_index(x.asnumpy(), rshape)
    print(np_out)
    assert len(mx_out) == len(np_out)
    for elem_mx, elem_np in zip(mx_out, np_out):
        assert elem_mx.asnumpy().shape == elem_np.shape
        assert_almost_equal(elem_mx.asnumpy(), elem_np, rtol=rtol, atol=atol)


@use_np
def test_np_diag_indices_from():
    class TestDiag_indices_from(HybridBlock):
        def __init__(self) :
            super(TestDiag_indices_from, self).__init__()

        def forward(self, a):
            return np.diag_indices_from(a)

    dtypes = [np.int8, np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64]
    shapes = [(2, 2), (4, 4), (5, 5, 5), (6, 6, 6, 6), (8, 8, 8, 8)]
    combinations = itertools.product([False, True], dtypes, shapes)
    for hybridize, dtype, shape in combinations:
        rtol = 1e-2 if dtype == np.float16 else 1e-3
        atol = 1e-4 if dtype == np.float16 else 1e-5
        test_diag_indices_from = TestDiag_indices_from()
        if hybridize:
            test_diag_indices_from.hybridize()
        x = np.random.uniform(-8, 8, size=shape).astype(dtype)
        mx_out = test_diag_indices_from(x)
        np_out = onp.diag_indices_from(x.asnumpy())
        assert len(mx_out) == len(np_out)
        for elem_mx, elem_np in zip(mx_out, np_out):
            assert elem_mx.asnumpy().shape == elem_np.shape
            assert_almost_equal(elem_mx.asnumpy(), elem_np, rtol=rtol, atol=atol)
        # no backward function for diag_indices_from operator

        # Test imperative once again
        mx_out = np.diag_indices_from(x)
        np_out = onp.diag_indices_from(x.asnumpy())
        assert len(mx_out) == len(np_out)
        for elem_mx, elem_np in zip(mx_out, np_out):
            assert elem_mx.asnumpy().shape == elem_np.shape
            assert_almost_equal(elem_mx.asnumpy(), elem_np, rtol=rtol, atol=atol)


@use_np
def test_np_interp():
    class TestInterp(HybridBlock):
        def __init__(self, left=None, right=None, period=None):
            super(TestInterp, self).__init__()
            self._left = left
            self._right = right
            self._period = period

        def forward(self, x, xp, fp):
            return np.interp(x, xp, fp, left=self._left, right=self._right, period=self._period)

    class TestInterpScalar(HybridBlock):
        def __init__(self, x=None, left=None, right=None, period=None):
            super(TestInterpScalar, self).__init__()
            self._x = x
            self._left = left
            self._right = right
            self._period = period

        def forward(self, xp, fp):
            return np.interp(self._x, xp, fp, left=self._left, right=self._right, period=self._period)

    xtypes = [np.int64, np.float32, np.float64]
    dtypes = [np.int32, np.int64, np.float32, np.float64]
    xshapes = [
        (), (3,), (5,), (20,),
        (2, 2), (4, 4), (8, 8),
        (5, 5, 5), (8, 0, 8)
    ]
    dsizes = [10, 30]
    periods = [None, 2*np.pi]
    lefts = [None, -10, 0]
    rights= [None, 20, 50]
    flags = [True, False]
    combinations = itertools.product(flags, flags, xshapes, dsizes, xtypes, dtypes, lefts, rights, periods)
    for hybridize, x_scalar, xshape, dsize, xtype, dtype, left, right, period in combinations:
        rtol = 1e-3
        atol = 1e-5
        if period is not None:
            x = np.random.uniform(-np.pi, np.pi, size=xshape).astype(xtype)
            xp = np.random.uniform(0, 2*np.pi, size=dsize)
            fp = np.sin(xp)
        else:
            x = np.random.uniform(0, 100, size=xshape).astype(xtype)
            xp = np.sort(np.random.choice(100, dsize, replace=False).astype(dtype))
            fp = np.random.uniform(-50, 50, size=dsize).astype(dtype)
        np_x = x.asnumpy()
        if x_scalar and xshape == ():
            x = x.item()
            np_x = x
            test_interp = TestInterpScalar(x=x, left=left, right=right, period=period)
        else:
            test_interp = TestInterp(left=left, right=right, period=period)
        if hybridize:
            test_interp.hybridize()
        mx_out = test_interp(xp, fp) if (x_scalar and xshape == ()) else test_interp(x, xp, fp)
        np_out = onp.interp(np_x, xp.asnumpy(), fp.asnumpy(), left=left, right=right, period=period)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)

        mx_out = np.interp(x, xp, fp, left=left, right=right, period=period)
        np_out = onp.interp(np_x ,xp.asnumpy(), fp.asnumpy(), left=left, right=right, period=period)
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=atol, rtol=rtol)


@use_np
def test_np_bincount():
    class TestBincount(HybridBlock):
        def __init__(self, minlength=0):
            super(TestBincount, self).__init__()
            self._minlength = minlength

        def forward(self, a):
            return np.bincount(a, None, self._minlength)

    class TestBincountWeights(HybridBlock):
        def __init__(self, minlength=0):
            super(TestBincountWeights, self).__init__()
            self._minlength = minlength

        def forward(self, a, weights):
            return np.bincount(a, weights, self._minlength)

    dtypes = [np.int8, np.uint8, np.int32, np.int64]
    weight_types = [np.int32, np.int64, np.float16, np.float32, np.float64]
    shapes = [(), (5,), (10,), (15,), (20,), (30,), (50,)]
    min_lengths = [0, 5, 20, 50]
    has_weights = [True, False]
    combinations = itertools.product([True, False], shapes, dtypes, weight_types, has_weights, min_lengths)
    for hybridize, shape, dtype, weight_type, has_weight, minlength in combinations:
        rtol = 1e-2 if weight_type == np.float16 else 1e-3
        atol = 1e-4 if weight_type == np.float16 else 1e-5
        if shape != ():
            data = np.random.uniform(0, 10, size=shape).astype(dtype)
            weights = np.random.uniform(0, 10, size=shape).astype(weight_type) if has_weight else None
        else:
            data = np.array(()).astype(dtype)
            weights = np.array(()).astype(weight_type) if has_weight else None
        weights_np = weights.asnumpy() if has_weight else None
        test_bincount = TestBincountWeights(minlength) if has_weight else TestBincount(minlength)
        if hybridize:
            test_bincount.hybridize()
        mx_out = test_bincount(data, weights) if has_weight else test_bincount(data)
        np_out = onp.bincount(data.asnumpy(), weights_np, minlength)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
        # No backward operation for operator bincount at this moment

        # Test imperative once again
        mx_out = np.bincount(data, weights, minlength)
        np_out = onp.bincount(data.asnumpy(), weights_np, minlength)
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@use_np
@pytest.mark.skip(reason='Test hangs. Tracked in #18144')
def test_np_empty_like():
    class TestEmptyLike(HybridBlock):
        def __init__(self, dtype, order, subok):
            super(TestEmptyLike, self).__init__()
            self._dtype = dtype
            self._order = order
            self._subok = subok

        def forward(self, x, *args, **kwargs):
            return np.empty_like(x, self._dtype, self._order, self._subok)

    if StrictVersion(platform.python_version()) < StrictVersion('3.0.0'):
        return

    dtypes = [None, 'float16', 'float32', np.int8, np.uint8, np.int32, np.int64,
              np.float16, np.float32, np.float64, np.bool_]
    shapes = [
        (),
        (1,),
        (5,),
        (4, 3),
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
    orders = ["C"]
    subok_list = [False]
    flags = [False]
    _np_version = onp.version.version
    for dtype, shape, hybridize, order, subok in itertools.product(dtypes, shapes, flags, orders, subok_list):
        prototype = np.random.uniform(low=0, high=100, size=shape, dtype='float64').astype(dtype)
        test = TestEmptyLike(dtype, order, subok)
        if StrictVersion(_np_version) >= StrictVersion('1.6.0'):
            expected_ret = onp.empty_like(prototype, dtype=dtype, order=order, subok=subok)
        else:
            expected_ret = onp.empty_like(prototype)
        if hybridize:
            test.hybridize()
        ret = test(prototype)
        assert ret.asnumpy().shape == expected_ret.shape

        # check imperative again
        ret = np.empty_like(prototype, dtype, order, subok)
        assert ret.asnumpy().shape == expected_ret.shape


@use_np
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('a_shape,b_shape,axes', [
    # - 2 x 2
    ((2,), (2,), (-1, -1, -1)),
    ((1, 2), (1, 2), (-1, -1, -1)),
    ((1, 2), (2, 2), (-1, -1, -1)),
    ((2, 2), (1, 2), (-1, -1, -1)),
    ((2, 2), (2, 2), (-1, -1, -1)),
    ((1, 2), (2, 2), (-1, 0, -1)),
    ((2, 2), (1, 2), (0, -1, -1)),
    ((2, 2), (2, 2), (0, 0, -1)),
    ((2, 2), (2, 2), (0, 0, 0)),
    ((5, 4, 3, 2), (5, 4, 3, 2), (-1, -1, -1)),
    ((1, 4, 3, 2), (5, 1, 3, 2), (-1, -1, -1)),
    ((5, 4, 3, 2), (5, 4, 3, 2), (-1, -1, 0)),
    ((2, 5, 4, 3), (5, 2, 4, 3), (0, 1, 2)),
    ((2, 5, 1, 3), (1, 2, 4, 3), (0, 1, 2)),
    # - 2 x 3
    ((2,), (3,), (-1, -1, -1)),
    ((1, 2,), (1, 3,), (-1, -1, -1)),
    ((2, 2,), (2, 3,), (0, -1, 0)),
    ((1, 2,), (2, 3,), (-1, -1, -1)),
    ((2, 2,), (1, 3,), (-1, -1, -1)),
    ((2, 1,), (3, 4,), (0, 0, 0)),
    ((2, 1, 3), (4, 3, 1), (0, 1, 2)),
    ((6, 5, 4, 2), (6, 5, 4, 3), (-1, -1, -1)),
    ((2, 6, 5, 4), (6, 5, 4, 3), (0, -1, 2)),
    ((2, 6, 5, 4), (6, 3, 5, 4), (0, 1, 2)),
    ((6, 2, 5, 4), (6, 5, 3, 4), (1, 2, 0)),
    ((6, 2, 1, 4), (1, 5, 3, 4), (1, 2, 0)),
    # - 3 x 2
    ((3,), (2,), (-1, -1, -1)),
    ((1, 3,), (1, 2,), (-1, -1, -1)),
    ((2, 3,), (2, 2,), (-1, 0, 0)),
    ((2, 3,), (1, 2,), (-1, -1, -1)),
    ((2, 3,), (1, 2,), (-1, -1, -1)),
    ((3, 4, 4), (1, 1, 2,), (0, -1, 0)),
    ((3, 4, 4), (1, 2, 1,), (0, 1, 2)),
    ((6, 5, 4, 3), (6, 5, 4, 2), (-1, -1, -1)),
    ((3, 6, 5, 4), (6, 5, 4, 2), (0, -1, 2)),
    ((3, 6, 5, 4), (6, 2, 5, 4), (0, 1, 2)),
    ((6, 3, 5, 4), (6, 5, 2, 4), (1, 2, 0)),
    ((6, 3, 1, 4), (1, 5, 2, 4), (1, 2, 0)),
    # - 3 x 3
    ((3,), (3,), (-1, -1, -1)),
    ((1, 3,), (1, 3,), (-1, -1, -1)),
    ((2, 3,), (3, 2,), (-1, 0, 0)),
    ((1, 3,), (3, 2,), (-1, 0, 0)),
    ((1, 3,), (3, 4,), (-1, 0, 0)),
    ((1, 1, 3,), (3, 2, 2), (-1, 0, 0)),
    ((1, 1, 2, 3,), (3, 2, 2, 2), (-1, 0, 0)),
    ((6, 5, 4, 3), (6, 5, 4, 3), (-1, -1, -1)),
    ((3, 6, 5, 4), (6, 5, 4, 3), (0, -1, 2)),
    ((3, 6, 5, 4), (6, 3, 5, 4), (0, 1, 2)),
    ((6, 3, 5, 4), (6, 5, 3, 4), (1, 2, 0)),
    ((6, 3, 1, 4), (1, 5, 3, 4), (1, 2, -1)),

    # - (a_shape, b_shape, None)
    ((2,), (2,), None),
    ((2,), (3,), None),
    ((3,), (2,), None),
    ((3,), (3,), None),
    ((5, 4, 3, 2), (5, 4, 3, 2), None),
    ((6, 5, 4, 2), (6, 5, 4, 3), None),
    ((6, 5, 4, 3), (6, 5, 4, 2), None),
    ((6, 5, 4, 3), (6, 5, 4, 3), None),
    ((1, 4, 3, 2), (5, 1, 3, 2), None),
    ((6, 1, 4, 2), (6, 5, 1, 3), None),
    ((6, 5, 1, 3), (1, 5, 4, 2), None),
    ((1, 5, 4, 3), (6, 5, 1, 3), None),

    # - (a_shape, b_shape, (a_axis, b_axis, c_axis, axis))
    ((2, 5, 4, 3), (2, 5, 4, 3), (-1, -1, -1, 0,)),
    ((6, 2, 5, 4), (6, 3, 5, 4), (-1, -1, -1, 1,)),
    ((6, 5, 3, 4), (6, 5, 2, 4), (-1, -1, -1, 2,)),
    ((6, 5, 4, 3), (6, 5, 4, 3), (-1, -1, -1, 3,)),
])
def test_np_cross(a_shape, b_shape, axes, dtype, hybridize):
    class TestNumpyCross(HybridBlock):
        def __init__(self, axisa=-1, axisb=-1, axisc=-1, axis=None):
            super(TestNumpyCross, self).__init__()
            self._axisa = axisa
            self._axisb = axisb
            self._axisc = axisc
            self._axis = axis

        def forward(self, a, b):
            return np.cross(a, b, self._axisa, self._axisb, self._axisc, self._axis)

    def check_np_cross(x, a_np, b_np, axises):
        try:
            if axises is None:
                x_expected = onp.cross(a_np, b_np)
            elif len(axises) == 4:
                (a_axis, b_axis, c_axis, axis,) = axises
                x_expected = onp.cross(a_np, b_np, axisa=a_axis, axisb=b_axis, axisc=c_axis, axis=axis)
            else:
                (a_axis, b_axis, c_axis,) = axises
                x_expected = onp.cross(a_np, b_np, axisa=a_axis, axisb=b_axis, axisc=c_axis)
        except Exception as e:
            print("a:", a_np)
            print("a shape:", a_np.shape)
            print("b:", b_np)
            print("b shape:", b_np.shape)
            print(e)
        else:
            assert x.shape == x_expected.shape
            assert_almost_equal(x.asnumpy(), x_expected, rtol=rtol, atol=atol)

    def check_not_use_broadcast(a_np, b_np, axises):
        a_shape = a_np.shape
        b_shape = b_np.shape
        if axises is None:
            return a_shape[:-1] == b_shape[:-1]
        elif len(axises) == 4:
            axis = axises[3]
            a_moveaxis_shape = onp.moveaxis(a_np, axis, -1).shape
            b_moveaxis_shape = onp.moveaxis(b_np, axis, -1).shape
            return a_moveaxis_shape[:-1] == b_moveaxis_shape[:-1]
        else:
            a_axis = axises[0]
            b_axis = axises[1]
            a_moveaxis_shape = onp.moveaxis(a_np, a_axis, -1).shape
            b_moveaxis_shape = onp.moveaxis(b_np, b_axis, -1).shape
            return a_moveaxis_shape[:-1] == b_moveaxis_shape[:-1]

    # calculate dL = gradC * dC
    def cal_dL(grad_c_move, dc_move):
        num = int(onp.prod(dc_move.shape))
        grad_c_move_1d = grad_c_move.reshape((num,))
        dc_move_1d = dc_move.reshape((num,))
        dL = onp.inner(grad_c_move_1d, dc_move_1d)
        return dL

    # get reduced axis index
    def get_reduce_axis(shape, broad_shape):
        axis = list()
        length = len(broad_shape) if len(shape) == len(broad_shape) + 1 else len(broad_shape) - 1
        for i in range(length):
            if shape[i] != broad_shape[i]:
                axis.append(i)
        return tuple(axis) if len(axis) > 0 else None

    # get grad_a and grad_b
    def get_cross_backward(a, b, axises):
        if axises == None:
            a_axis, b_axis, c_axis = (-1,) * 3
        elif len(axises) == 4:
            a_axis, b_axis, c_axis = (axises[-1],) * 3
        else:
            (a_axis, b_axis, c_axis) = axises
        c = onp.cross(a, b, axisa=a_axis, axisb=b_axis, axisc=c_axis)
        c_move = onp.moveaxis(c, c_axis, -1) if a.shape[a_axis] == 3 or b.shape[b_axis] == 3 else c
        grad_c_move = onp.ones(shape=c_move.shape, dtype=c_move.dtype)
        a_move = onp.moveaxis(a, a_axis, -1)
        b_move = onp.moveaxis(b, b_axis, -1)
        da_move = onp.random.uniform(-1., 1., size=a_move.shape)
        db_move = onp.random.uniform(-1., 1., size=b_move.shape)
        # dC = dA x B + A x dB
        dc_move = onp.cross(da_move, b_move) + onp.cross(a_move, db_move)
        # dL1 = Tr(grad_C.T * dC) = dL/dCi * dCi
        dL1 = cal_dL(grad_c_move, dc_move)
        # check cross backward.
        if a.shape[a_axis] == 2 and b.shape[b_axis] == 2:
            # Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
            shape = grad_c_move.shape if grad_c_move.ndim != 0 else (1,)
            grad_a_move = onp.empty(shape, dtype=a_move.dtype)
            grad_b_move = onp.empty(shape, dtype=b_move.dtype)
            grad_a_move = onp.expand_dims(grad_a_move, -1).repeat(2, axis=-1)
            grad_b_move = onp.expand_dims(grad_b_move, -1).repeat(2, axis=-1)
            a_move_0 = a_move[..., 0]
            a_move_1 = a_move[..., 1]
            b_move_0 = b_move[..., 0]
            b_move_1 = b_move[..., 1]
            grad_a_move_0 = grad_c_move * b_move_1
            grad_a_move_1 = grad_c_move * b_move_0
            if grad_a_move_1.ndim == 0:
                grad_a_move_1 = -grad_a_move_1
            else:
                onp.negative(grad_a_move_1, out=grad_a_move_1)
            grad_b_move_0 = grad_c_move * a_move_1
            grad_b_move_1 = grad_c_move * a_move_0
            if grad_b_move_0.ndim == 0:
                grad_b_move_0 = -grad_b_move_0
            else:
                onp.negative(grad_b_move_0, out=grad_b_move_0)
            grad_a_move[..., 0] = grad_a_move_0
            grad_a_move[..., 1] = grad_a_move_1
            grad_b_move[..., 0] = grad_b_move_0
            grad_b_move[..., 1] = grad_b_move_1
        else:
            # Case 4: a.shape[-1] == 3 and b.shape[-1] == 3, param.axisc is not ignored.
            grad_a_move = onp.cross(b_move, grad_c_move)
            grad_b_move = onp.cross(grad_c_move, a_move)
            if a.shape[a_axis] == 2:
                # Case 2: a.shape[-1] == 2 and b.shape[-1] == 3, param.axisc is not ignored.
                grad_a_move = onp.delete(grad_a_move, obj=-1, axis=-1)
            if b.shape[b_axis] == 2:
                # Case 3: a.shape[-1] == 3 and b.shape[-1] == 2, param.axisc is not ignored.
                grad_b_move = onp.delete(grad_b_move, obj=-1, axis=-1)

        if not check_not_use_broadcast(a, b, axises):
            a_broad_axis = get_reduce_axis(a_move.shape, c_move.shape)
            b_broad_axis = get_reduce_axis(b_move.shape, c_move.shape)
            if a_broad_axis is not None:
                grad_a_move_reduce = onp.ones_like(a_move)
                grad_a_move_reduce = onp.sum(grad_a_move, axis=a_broad_axis, out=grad_a_move_reduce, keepdims=True)
                grad_a_move = grad_a_move_reduce
            if b_broad_axis is not None:
                grad_b_move_reduce = onp.ones_like(b_move)
                grad_b_move_reduce = onp.sum(grad_b_move, axis=b_broad_axis, out=grad_b_move_reduce, keepdims=True)
                grad_b_move = grad_b_move_reduce
        # dL2 = dL/dAi * dAi + dL/dBi * dBi
        dL2 = cal_dL(grad_a_move, da_move) + cal_dL(grad_b_move, db_move)
        assert_almost_equal(dL1, dL2, rtol=rtol, atol=atol)
        # move working axis
        return onp.moveaxis(grad_a_move, -1, a_axis), onp.moveaxis(grad_b_move, -1, b_axis)

    rtol = 1e-3
    atol = 1e-5
    if axes is None:
        a_axis, b_axis, c_axis = (-1,) * 3
        test_numpy_cross = TestNumpyCross()
    elif len(axes) == 4:
        (a_axis, b_axis, c_axis, axis,) = axes
        test_numpy_cross = TestNumpyCross(axisa=a_axis, axisb=b_axis, axisc=c_axis, axis=axis)
    else:
        (a_axis, b_axis, c_axis,) = axes
        test_numpy_cross = TestNumpyCross(axisa=a_axis, axisb=b_axis, axisc=c_axis)
    if hybridize:
        test_numpy_cross.hybridize()
    a_np = onp.random.uniform(-10., 10., size=a_shape)
    b_np = onp.random.uniform(-10., 10., size=b_shape)
    a = np.array(a_np, dtype=dtype)
    b = np.array(b_np, dtype=dtype)
    a.attach_grad()
    b.attach_grad()

    # check cross validity
    with mx.autograd.record():
        mx_out = test_numpy_cross(a, b)
    check_np_cross(mx_out, a.asnumpy(), b.asnumpy(), axes)

    # check cross backward
    mx.autograd.backward(mx_out)
    grad_a_expected, grad_b_expected = get_cross_backward(a.asnumpy(), b.asnumpy(), axes)
    assert_almost_equal(a.grad.asnumpy(), grad_a_expected, rtol=rtol, atol=atol)
    assert_almost_equal(b.grad.asnumpy(), grad_b_expected, rtol=rtol, atol=atol)

    # check imperative once again
    mx_out = test_numpy_cross(a, b)
    check_np_cross(mx_out, a.asnumpy(), b.asnumpy(), axes)


@use_np
def test_np_rollaxis():
    class TestRollaxis(HybridBlock):
        def __init__(self, axis=0, start=0):
            super(TestRollaxis, self).__init__()
            self._axis = axis
            self._start = start

        def forward(self, a, *args, **kwargs):
            return np.rollaxis(a, axis=self._axis, start=self._start)

    dtypes = ['int32', 'int64', 'float16', 'float32', 'float64']
    for hybridize in [False, True]:
        for dtype in dtypes:
            for ndim in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                shape = rand_shape_nd(ndim, dim=5, allow_zero_size=True)
                np_data = onp.random.uniform(low=-100, high=100, size=shape).astype(dtype)
                mx_data = np.array(np_data, dtype=dtype)
                for axis in range(-ndim, ndim):
                    for start in range(-ndim, ndim + 1):
                        # test gluon
                        test_rollaxis = TestRollaxis(axis, start)
                        if hybridize:
                            test_rollaxis.hybridize()
                        np_out = onp.rollaxis(np_data, axis=axis, start=start)
                        mx_data.attach_grad()
                        with mx.autograd.record():
                            mx_out = test_rollaxis(mx_data)
                        assert mx_out.shape == np_out.shape
                        mx_out.backward()
                        assert same(mx_data.grad.shape, mx_data.shape)
                        assert same(mx_data.grad.asnumpy(), onp.ones(shape))
                        # test imperative
                        np_out = onp.rollaxis(np_data, axis=axis, start=start)
                        mx_out = np.rollaxis(mx_data, axis=axis, start=start)
                        assert np_out.dtype == mx_out.dtype
                        assert same(mx_out.asnumpy(), np_out)


@use_np
def test_npx_stop_gradient():
    class TestStopGradient(HybridBlock):
        def forward(self, a):
            return npx.stop_gradient(a)
    dtypes = ['float16', 'float32', 'float64']
    for hybridize in [False, True]:
        for dtype in dtypes:
            for grad_req in ['write', 'add']:
                dat = np.ones((10,), dtype=dtype)
                dat.attach_grad(grad_req)
                dat.grad[:] = 2
                old_grad = dat.grad.asnumpy()
                net = TestStopGradient()
                if hybridize:
                    net.hybridize()
                with mx.autograd.record():
                    out = net(dat)
                    out = out + dat
                    out.backward()
                new_grad = dat.grad.asnumpy()
                assert same(out.asnumpy(), dat.asnumpy() * 2)
                if grad_req == 'write':
                    assert_almost_equal(new_grad, onp.ones_like(dat, dtype=dtype))
                elif grad_req == 'add':
                    assert_almost_equal(new_grad, old_grad + 1)


def test_npx_broadcast_like_different_types():
    x = mx.np.zeros((2, 1))
    y = mx.np.ones((2, 2))

    y = mx.np.array(y).astype('int32')
    z = mx.npx.broadcast_like(x, y)
    assert_almost_equal(z.asnumpy(), np.array([[0,0],[0,0]]))
    assert x.dtype == z.dtype


@use_np
def test_np_elementwise_ops_on_misaligned_input():
    a = np.array([1,2,3,4], dtype='float16')
    b = np.array([1,2,3,4], dtype='float16')

    c = a[1:3]
    d = b[1:3]
    # Note: testing just elemwise_add since all elemwise_ops
    #       share the implementation
    c[:] = c + d
    mx.nd.waitall()

    a = np.array([1,2,3,4], dtype='float16')
    b = np.array([1,2,3,4], dtype='float16')

    c = a[0:3]
    d = b[0:3]
    c[:] = c + d
    mx.nd.waitall()
    assert a[3] == 4.0


@use_np
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('lead_dim', [2, 3, 4, 6, 10])
@pytest.mark.parametrize('both_ways', [False, True])
def test_np_broadcast_ops_on_misaligned_input(dtype, lead_dim, both_ways):
    shape = list(rand_shape_2d()) + [lead_dim]
    small_shape = [shape[0], 1, lead_dim]
    if both_ways:
        # Broadcast in both ways [1, K, L] x [M, 1, L]
        big_shape = [1, shape[1], lead_dim]
    else:
        big_shape = shape
    size = onp.product(shape)
    small_size = onp.product(small_shape)
    big_size = onp.product(big_shape)
    a = np.arange(5000)
    b = np.arange(5000)
    e = np.arange(5000)
    c = a[1:big_size + 1].reshape(tuple(big_shape))
    d = b[1:small_size + 1].reshape(tuple(small_shape))
    f = e[1:size + 1].reshape(tuple(shape))
    f[:] = c + d
    expected = c.asnumpy() + d.asnumpy()
    mx.nd.waitall()
    assert_almost_equal(f, expected)


@use_np
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('lead_dim', [2, 3, 4, 6, 10])
@pytest.mark.parametrize('both_ways', [False, True])
def test_np_broadcast_ops_on_misaligned_input_oneside(dtype, lead_dim, both_ways):
    shape = list(rand_shape_2d()) + [lead_dim]
    small_shape = [shape[0], shape[1], 1]
    if both_ways:
        # Broadcast in both ways [1, K, L] x [M, 1, 1]
        big_shape = [1, shape[1], lead_dim]
    else:
        big_shape = shape
    size = onp.product(shape)
    small_size = onp.product(small_shape)
    big_size = onp.product(big_shape)
    a = np.arange(5000)
    b = np.arange(5000)
    e = np.arange(5000)
    c = a[1:big_size + 1].reshape(tuple(big_shape))
    d = b[1:small_size + 1].reshape(tuple(small_shape))
    f = e[1:size + 1].reshape(tuple(shape))
    f[:] = c + d
    expected = c.asnumpy() + d.asnumpy()
    mx.nd.waitall()
    assert_almost_equal(f, expected)

@use_np
@pytest.mark.parametrize('num_batch', [1, 2])
@pytest.mark.parametrize('num_channel_data', [4, 8])
@pytest.mark.parametrize('num_deformable_group', [1, 2])
@pytest.mark.parametrize('input_height', [5, 6])
@pytest.mark.parametrize('input_width', [5, 6])
@pytest.mark.parametrize('dilate', [(1, 1), (2, 2)])
@pytest.mark.parametrize('grad_nodes', [['im_data'], ['offset_data'], ['weight']])
def test_modulated_deformable_convolution(num_batch, num_channel_data, num_deformable_group,
                                          input_height, input_width, dilate, grad_nodes):
    output_height = input_height
    output_width = input_width
    im_data = np.random.rand(num_batch, num_channel_data, input_height, input_width)
    offset_data = \
        np.random.rand(num_batch, num_deformable_group * 3 * 3 * 2, output_height, output_width)\
        * 0.8 + 0.1
    mask_data = np.random.rand(num_batch, num_deformable_group * 3 * 3, output_height, output_width)
    mask_data = 0.5 * (1 + np.tanh(0.5 * mask_data)) # sigmoid
    weight = np.random.normal(0, 0.001, (num_channel_data, num_channel_data, 3, 3))
    bias = np.zeros(num_channel_data)

    im_data_var = mx.symbol.Variable(name="im_data").as_np_ndarray()
    offset_data_var = mx.symbol.Variable(name="offset_data").as_np_ndarray()
    mask_data_var = mx.symbol.Variable(name="mask_data").as_np_ndarray()
    weight_var = mx.symbol.Variable(name="weight").as_np_ndarray()
    bias_var = mx.symbol.Variable(name="bias").as_np_ndarray()
    op = mx.sym.npx.modulated_deformable_convolution(name='test_op', data=im_data_var,
                                                     offset=offset_data_var, mask=mask_data_var,
                                                     weight=weight_var, bias=bias_var,
                                                     num_filter=num_channel_data, pad=dilate,
                                                     kernel=(3, 3), stride=(1, 1), dilate=dilate,
                                                     num_deformable_group=num_deformable_group)
    if grad_nodes[0] == 'offset_data':
        # wider tolerance needed for coordinate differential
        rtol, atol = 1.0, 1e-2
    else:
        rtol, atol = 0.05, 1e-3


@use_np
def test_broadcast_like_different_types():
    x = mx.np.zeros((2, 1))
    y = mx.np.ones((2, 2))

    y = mx.np.array(y).astype('int32')
    z = mx.npx.broadcast_like(x, y, 1, 1)
    assert_almost_equal(z.asnumpy(), np.array([[0,0],[0,0]]))
    assert x.dtype == z.dtype


@use_np
def test_np_apply_along_axis_fallback():
    data = np.random.randint(-100, 100, (2, 3))
    axis = 1
    func1d = lambda x: x.mean()
    np_y = onp.apply_along_axis(func1d, 1, data.asnumpy())
    y1 = np.apply_along_axis(func1d, 1, data)
    y2 = np.apply_along_axis(func1d, 1, arr=data)
    assert_almost_equal(y1.asnumpy(), np_y)
    assert y1.asnumpy().dtype == np_y.dtype
    assert_almost_equal(y2.asnumpy(), np_y)
    assert y2.asnumpy().dtype == np_y.dtype


def check_multihead_attention_selfatt(dtype):
    class TestSelfAtt1(mx.gluon.HybridBlock):
        def __init__(self):
            super().__init__()
            self.batch_size = 2
            self.qkv_length = 7  # length of a sequence
            self.qkv_dim = 9     # dimension of encoding
            self.num_heads = 3   # number of attention head
            self.head_dim = 5    # head size
            self.out_dim = 13 * self.num_heads
            self.qkv_units = self.num_heads * self.head_dim

            self.q_weight = Parameter('q_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.k_weight = Parameter('k_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.v_weight = Parameter('v_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.q_bias = Parameter('q_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.k_bias = Parameter('k_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.v_bias = Parameter('v_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)                                       
            self.out_weight = Parameter('out_weight', shape=(self.out_dim, self.qkv_units),
                                        init=None, dtype=dtype, allow_deferred_init=True)
            self.out_bias = Parameter('out_bias', shape=(self.out_dim,),
                                      init=None, dtype=dtype, allow_deferred_init=True)

        def forward(self, qkv):
            device = qkv.device
            qkv_weight = self.convert_weight(self.q_weight.data().to_device(device),
                                             self.k_weight.data().to_device(device),
                                             self.v_weight.data().to_device(device),
                                             self.num_heads)
            qkv_bias = self.convert_bias(self.q_bias.data().to_device(device),
                                         self.k_bias.data().to_device(device),
                                         self.v_bias.data().to_device(device),
                                         self.num_heads)
            qkv = np.transpose(qkv, axes=(1, 0, 2))
            qkv_proj = npx.fully_connected(qkv, weight=qkv_weight, bias=qkv_bias, flatten=False,
                                           num_hidden=self.qkv_units * 3, no_bias=False)
            att_score = npx.interleaved_matmul_selfatt_qk(qkv_proj, heads=self.num_heads)
            weighted_value = npx.interleaved_matmul_selfatt_valatt(qkv_proj, att_score, heads=self.num_heads)
            output = npx.fully_connected(weighted_value, weight=self.out_weight.data().to_device(device),
                                         bias=self.out_bias.data().to_device(device), flatten=False,
                                         num_hidden=self.out_dim, no_bias=False)
            return np.transpose(output, axes=(1, 0, 2)), att_score

        def convert_weight(self, q_weight, k_weight, v_weight, num_heads):
            q_weight = npx.reshape(q_weight, (num_heads, -1, -2), reverse=True)
            k_weight = npx.reshape(k_weight, (num_heads, -1, -2), reverse=True)
            v_weight = npx.reshape(v_weight, (num_heads, -1, -2), reverse=True)
            all_weights = np.concatenate([q_weight, k_weight, v_weight], axis=-2)
            all_weights = npx.reshape(all_weights, (-1, -2), reverse=True)
            return all_weights

        def convert_bias(self, q_bias, k_bias, v_bias, num_heads):
            q_bias = npx.reshape(q_bias, (num_heads, -1))
            k_bias = npx.reshape(k_bias, (num_heads, -1))
            v_bias = npx.reshape(v_bias, (num_heads, -1))
            all_bias = np.stack([q_bias, k_bias, v_bias], axis=1)
            all_bias = npx.reshape(all_bias, (-1,))
            return all_bias

    class TestSelfAtt2(mx.gluon.HybridBlock):
        def __init__(self):
            super().__init__()
            self.batch_size = 2
            self.qkv_length = 7  # length of a sequence
            self.qkv_dim = 9     # dimension of encoding
            self.num_heads = 3   # number of attention head
            self.head_dim = 5    # head size
            self.out_dim = 13 * self.num_heads
            self.qkv_units = self.num_heads * self.head_dim

            self.q_weight = Parameter('q_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.k_weight = Parameter('k_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.v_weight = Parameter('v_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.q_bias = Parameter('q_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.k_bias = Parameter('k_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.v_bias = Parameter('v_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)                                       
            self.out_weight = Parameter('out_weight', shape=(self.out_dim, self.qkv_units),
                                        init=None, dtype=dtype, allow_deferred_init=True)
            self.out_bias = Parameter('out_bias', shape=(self.out_dim,),
                                      init=None, dtype=dtype, allow_deferred_init=True)

        def forward(self, qkv):
            device = qkv.device
            q = npx.fully_connected(qkv, weight=self.q_weight.data().to_device(device),
                                    bias=self.q_bias.data().to_device(device), flatten=False,
                                    num_hidden=self.qkv_units, no_bias=False)
            k = npx.fully_connected(qkv, weight=self.k_weight.data().to_device(device),
                                    bias=self.k_bias.data().to_device(device), flatten=False,
                                    num_hidden=self.qkv_units, no_bias=False)
            v = npx.fully_connected(qkv, weight=self.v_weight.data().to_device(device),
                                    bias=self.v_bias.data().to_device(device), flatten=False,
                                    num_hidden=self.qkv_units, no_bias=False)
            q = npx.reshape(q, (-2, -2, self.num_heads, -1))
            q = np.transpose(q, axes=(0, 2, 1, 3))
            q = npx.reshape(q, (-1, -2, -2), reverse=True)
            k = npx.reshape(k, (-2, -2, self.num_heads, -1))
            k = np.transpose(k, axes=(0, 2, 1, 3))
            k = npx.reshape(k, (-1, -2, -2), reverse=True)
            q = q / np.sqrt(q.shape[-1])
            qkv = np.transpose(qkv, axes=(1, 0, 2))
            att_score = npx.batch_dot(q, k, transpose_b=True)

            v = npx.reshape(v, (-2, -2, self.num_heads, -1))
            v = np.transpose(v, axes=(0, 2, 1, 3))
            v = npx.reshape(v, (-1, -2, -2), reverse=True)
            weighted_value = npx.batch_dot(att_score, v)
            weighted_value = npx.reshape(weighted_value, (-1, self.num_heads, -2, -2),
                                         reverse=True)
            weighted_value = np.transpose(weighted_value, axes=(0, 2, 1, 3))
            weighted_value = npx.reshape(weighted_value, (-2, -2, -1))
            output = npx.fully_connected(weighted_value, weight=self.out_weight.data().to_device(device),
                                         bias=self.out_bias.data().to_device(device), flatten=False,
                                         num_hidden=self.out_dim, no_bias=False)
            return output, att_score

    qkv = np.random.uniform(size=(2, 7, 9), dtype=dtype)
    block1 = TestSelfAtt1()
    block2 = TestSelfAtt2()
    block1.initialize()
    block2.initialize()
    params1 = block1.collect_params()
    params2 = block2.collect_params()
    orig_params1 = copy.deepcopy(params1)
    for key, val in orig_params1.items():
        params2[key].set_data(copy.deepcopy(val.data()))
    block1.hybridize()
    block2.hybridize()
    with mx.autograd.record():
        out1, att_score1 = block1(qkv)
    out1.backward()
    with mx.autograd.record():
        out2, att_score2 = block2(qkv)
    out2.backward()
    grads1 = {k : v for k, v in params1.items()}
    grads2 = {k : v for k, v in params2.items()}
    assert_allclose(att_score1.asnumpy(), att_score2.asnumpy(), rtol=1e-2, atol=1e-3)
    assert_allclose(out1.asnumpy(), out2.asnumpy(), rtol=1e-2, atol=1e-3)

    for k in grads1.keys():
        assert(grads1[k].data().dtype == grads2[k].data().dtype)
        assert(grads1[k].data().shape == grads2[k].data().shape)
        assert_allclose(grads1[k].data().asnumpy(), grads2[k].data().asnumpy(), rtol=1e-2, atol=1e-3)


@use_np
@assert_raises_cuda_not_satisfied(min_version='9.1')
@pytest.mark.serial
def test_multihead_attention_selfatt():
    dtypes = ['float32']
    if mx.device.current_device().device_type == 'gpu':
        dtypes += ['float16']

    for dtype in dtypes:
        check_multihead_attention_selfatt(dtype=dtype)


def check_multihead_attention_encdec(dtype):
    class TestSelfAtt1(mx.gluon.HybridBlock):
        def __init__(self):
            super().__init__()
            self.batch_size = 2
            self.qkv_length = 7  # length of a sequence
            self.qkv_dim = 9     # dimension of encoding
            self.num_heads = 3   # number of attention head
            self.head_dim = 5    # head size
            self.out_dim = 13 * self.num_heads
            self.qkv_units = self.num_heads * self.head_dim

            self.q_weight = Parameter('q_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.k_weight = Parameter('k_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.v_weight = Parameter('v_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.q_bias = Parameter('q_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.k_bias = Parameter('k_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.v_bias = Parameter('v_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)                                       
            self.out_weight = Parameter('out_weight', shape=(self.out_dim, self.qkv_units),
                                        init=None, dtype=dtype, allow_deferred_init=True)
            self.out_bias = Parameter('out_bias', shape=(self.out_dim,),
                                      init=None, dtype=dtype, allow_deferred_init=True)

        def forward(self, q, kv):
            device = kv.device
            kv_weight = self.convert_weight(self.k_weight.data().to_device(device),
                                            self.v_weight.data().to_device(device),
                                            self.num_heads)
            kv_bias = self.convert_bias(self.k_bias.data().to_device(device),
                                        self.v_bias.data().to_device(device),
                                        self.num_heads)
            kv = np.transpose(kv, axes=(1, 0, 2))
            kv_proj = npx.fully_connected(kv, weight=kv_weight, bias=kv_bias, flatten=False,
                                          num_hidden=self.qkv_units * 2, no_bias=False)
            q = np.transpose(q, axes=(1, 0, 2))
            q_proj = npx.fully_connected(q, weight=self.q_weight.data().to_device(device),
                                         bias=self.q_bias.data().to_device(device), flatten=False,
                                         num_hidden=self.qkv_units, no_bias=False)
            att_score = npx.interleaved_matmul_encdec_qk(q_proj, kv_proj, heads=self.num_heads)
            weighted_value = npx.interleaved_matmul_encdec_valatt(kv_proj, att_score, heads=self.num_heads)
            output = npx.fully_connected(weighted_value, weight=self.out_weight.data().to_device(device),
                                         bias=self.out_bias.data().to_device(device), flatten=False,
                                         num_hidden=self.out_dim, no_bias=False)
            return np.transpose(output, axes=(1, 0, 2)), att_score

        def convert_weight(self, k_weight, v_weight, num_heads):
            k_weight = npx.reshape(k_weight, (num_heads, -1, -2), reverse=True)
            v_weight = npx.reshape(v_weight, (num_heads, -1, -2), reverse=True)
            all_weights = np.concatenate([k_weight, v_weight], axis=-2)
            all_weights = npx.reshape(all_weights, (-1, -2), reverse=True)
            return all_weights

        def convert_bias(self, k_bias, v_bias, num_heads):
            k_bias = npx.reshape(k_bias, (num_heads, -1))
            v_bias = npx.reshape(v_bias, (num_heads, -1))
            all_bias = np.stack([k_bias, v_bias], axis=1)
            all_bias = npx.reshape(all_bias, (-1,))
            return all_bias

    class TestSelfAtt2(mx.gluon.HybridBlock):
        def __init__(self):
            super().__init__()
            self.batch_size = 2
            self.qkv_length = 7  # length of a sequence
            self.qkv_dim = 9     # dimension of encoding
            self.num_heads = 3   # number of attention head
            self.head_dim = 5    # head size
            self.out_dim = 13 * self.num_heads
            self.qkv_units = self.num_heads * self.head_dim

            self.q_weight = Parameter('q_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.k_weight = Parameter('k_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.v_weight = Parameter('v_weight', shape=(self.qkv_units, self.qkv_dim),
                                      init=None, dtype=dtype, allow_deferred_init=True)
            self.q_bias = Parameter('q_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.k_bias = Parameter('k_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)
            self.v_bias = Parameter('v_bias', shape=(self.qkv_units,),
                                    init=None, dtype=dtype, allow_deferred_init=True)                                       
            self.out_weight = Parameter('out_weight', shape=(self.out_dim, self.qkv_units),
                                        init=None, dtype=dtype, allow_deferred_init=True)
            self.out_bias = Parameter('out_bias', shape=(self.out_dim,),
                                      init=None, dtype=dtype, allow_deferred_init=True)

        def forward(self, q, kv):
            device = kv.device
            q = npx.fully_connected(q, weight=self.q_weight.data().to_device(device),
                                    bias=self.q_bias.data().to_device(device), flatten=False,
                                    num_hidden=self.qkv_units, no_bias=False)
            k = npx.fully_connected(kv, weight=self.k_weight.data().to_device(device),
                                    bias=self.k_bias.data().to_device(device), flatten=False,
                                    num_hidden=self.qkv_units, no_bias=False)
            v = npx.fully_connected(kv, weight=self.v_weight.data().to_device(device),
                                    bias=self.v_bias.data().to_device(device), flatten=False,
                                    num_hidden=self.qkv_units, no_bias=False)
            q = npx.reshape(q, (-2, -2, self.num_heads, -1))
            q = np.transpose(q, axes=(0, 2, 1, 3))
            q = npx.reshape(q, (-1, -2, -2), reverse=True)
            k = npx.reshape(k, (-2, -2, self.num_heads, -1))
            k = np.transpose(k, axes=(0, 2, 1, 3))
            k = npx.reshape(k, (-1, -2, -2), reverse=True)
            q = q / np.sqrt(q.shape[-1])
            att_score = npx.batch_dot(q, k, transpose_b=True)

            v = npx.reshape(v, (-2, -2, self.num_heads, -1))
            v = np.transpose(v, axes=(0, 2, 1, 3))
            v = npx.reshape(v, (-1, -2, -2), reverse=True)
            weighted_value = npx.batch_dot(att_score, v)
            weighted_value = npx.reshape(weighted_value, (-1, self.num_heads, -2, -2),
                                         reverse=True)
            weighted_value = np.transpose(weighted_value, axes=(0, 2, 1, 3))
            weighted_value = npx.reshape(weighted_value, (-2, -2, -1))
            output = npx.fully_connected(weighted_value, weight=self.out_weight.data().to_device(device),
                                         bias=self.out_bias.data().to_device(device), flatten=False,
                                         num_hidden=self.out_dim, no_bias=False)
            return output, att_score

    q = np.random.uniform(size=(2, 7, 9), dtype=dtype)
    kv = np.random.uniform(size=(2, 7, 9), dtype=dtype)
    block1 = TestSelfAtt1()
    block2 = TestSelfAtt2()
    block1.initialize()
    block2.initialize()
    params1 = block1.collect_params()
    params2 = block2.collect_params()
    orig_params1 = copy.deepcopy(params1)
    for key, val in orig_params1.items():
        params2[key].set_data(copy.deepcopy(val.data()))
    block1.hybridize()
    block2.hybridize()
    with mx.autograd.record():
        out1, att_score1 = block1(q, kv)
    out1.backward()
    with mx.autograd.record():
        out2, att_score2 = block2(q, kv)
    out2.backward()
    grads1 = {k : v for k, v in params1.items()}
    grads2 = {k : v for k, v in params2.items()}
    assert_allclose(att_score1.asnumpy(), att_score2.asnumpy(), rtol=1e-2, atol=1e-3)
    assert_allclose(out1.asnumpy(), out2.asnumpy(), rtol=1e-2, atol=1e-3)

    for k in grads1.keys():
        assert(grads1[k].data().dtype == grads2[k].data().dtype)
        assert(grads1[k].data().shape == grads2[k].data().shape)
        assert_allclose(grads1[k].data().asnumpy(), grads2[k].data().asnumpy(), rtol=1e-2, atol=1e-3)


@use_np
@assert_raises_cuda_not_satisfied(min_version='9.1')
@pytest.mark.serial
def test_multihead_attention_encdec():
    dtypes = ['float32']
    if mx.device.current_device().device_type == 'gpu':
        dtypes += ['float16']

    for dtype in dtypes:
        check_multihead_attention_encdec(dtype=dtype)


@use_np
def test_add_n():
    data_shape = (2, 2)
    input_num = 5
    data = [np.random.uniform(size=data_shape) for i in range(input_num)]
    rslt = np.zeros(shape=data_shape)
    for i in range(input_num):
        rslt += data[i]
    add_n_rslt = npx.add_n(*data, out=data[0])
    assert_almost_equal(rslt.asnumpy(), add_n_rslt.asnumpy(), atol=1e-5)


@use_np
def test_slice_like():
    for ndim in range(1, 6):
        from_shape = onp.random.randint(1, 11, size=(ndim,))
        shape = [s + onp.random.randint(0, 3) for s in from_shape]
        for t in range(ndim):
            if t > 0:
                axes = onp.random.randint(0, ndim, size=t).tolist()
            else:
                axes = []
            idx = []
            for i in range(ndim):
                idx.append(slice(0, shape[i]))
                if i in axes or not axes:
                    idx[i] = slice(0, from_shape[i])

            if axes:
                pos = onp.random.randint(0, t)
                if axes[pos] > 0:
                    axes[pos] -= ndim  # negative index
            x = np.array(onp.random.normal(size=shape))
            x1 = np.array(onp.random.normal(size=from_shape))
            x.attach_grad()
            x1.attach_grad()
            with mx.autograd.record():
                y = npx.slice_like(data=x, shape_like=x1, axes=axes)
            y.backward()
            assert_allclose(x.asnumpy()[idx], y.asnumpy())

            xx = x.asnumpy()
            xx[:] = 0.0
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(x1.grad.asnumpy(), np.zeros_like(x1.grad).asnumpy())


@use_np
@pytest.mark.parametrize('shape,num_filter,num_group,kernel,pad', [
    ((1, 4, 15), 16, 2, (2,), (0,)),
    ((8, 4, 16), 16, 1, (3,), (1,)),

    ((1, 4, 15, 16), 16, 2, (2, 2), (0, 0)),
    ((8, 4, 16, 16), 16, 1, (3, 3), (1, 1)),

    ((1, 4, 3, 15, 16), 16, 2, (2, 2, 2), (0, 0, 0)),
    ((8, 4, 3, 16, 16), 16, 1, (3, 3, 3), (1, 1, 1))])
def test_npx_deconvolution(shape, num_filter, num_group, kernel, pad):
    if len(kernel) == 3 and mx.current_device().device_type == 'gpu':
        pytest.skip('Skipping deconvoluition 3D tests for GPU')

    class TestConv(mx.gluon.HybridBlock):
        def __init__(self, w):
            super().__init__()
            self.weight = w

        def forward(self, x, *args):
            return npx.convolution(x, self.weight.data(x.device), no_bias=True, kernel=kernel,
                                   pad=pad, num_filter=self.weight.shape[0], num_group=num_group)

    class TestDeconv(mx.gluon.HybridBlock):
        def __init__(self):
            super().__init__()
            self.weight = mx.gluon.Parameter('weight', shape=(shape[1], int(num_filter/num_group), 
                                                              *kernel))
            self.bias = mx.gluon.Parameter('bias', shape=num_filter)

        def forward(self, x, *args):
            return npx.deconvolution(x, self.weight.data(x.device), self.bias.data(x.device), kernel,
                                     pad=pad, num_filter=num_filter, num_group=num_group)
    
    deconvNet = TestDeconv()
    deconvNet.initialize()

    # test imperative
    deconvData = np.random.uniform(0, 1, size=shape)
    npx_out_imp = deconvNet(deconvData)

    # test symbolic
    deconvNet.hybridize()
    deconvNet(deconvData)
    npx_out_sym = deconvNet(deconvData)
    assert_almost_equal(npx_out_imp, npx_out_sym)

    # compare outputs with reference tensors generated using convolution
    convNet = TestConv(deconvNet.weight)
    convNet.initialize()
    convData = np.random.uniform(0, 1, size=npx_out_imp.shape)
    convData.attach_grad()
    with mx.autograd.record():
        convOut = convNet(convData)
        y = np.reshape(convOut, -1)
        y = np.sum(y)
    y.backward()
    
    deconvData = np.ones_like(convOut)  # gradient of convOut
    deconvBias = np.repeat(deconvNet.bias.data(), int(np.prod(np.array(convData.grad.shape[2:])).item()))
    deconvRefOut = np.copy(convData.grad) + deconvBias.reshape((convData.grad.shape[1:]))
    deconvData.attach_grad()
    with mx.autograd.record():
        deconvOut = deconvNet(deconvData)
    deconvOut.backward()

    convData = np.ones_like(deconvOut)
    deconvRefGrad = convNet(convData)

    assert_almost_equal(deconvOut, deconvRefOut)
    assert_almost_equal(deconvData.grad, deconvRefGrad)


@use_np
@pytest.mark.parametrize('dtype', np.floating_dtypes)
def test_np_finfo(dtype):
    mx_finfo_obj = np.finfo(dtype)
    np_finfo = onp.finfo(dtype)
    assert (mx_finfo_obj.bits, mx_finfo_obj.eps, mx_finfo_obj.max, mx_finfo_obj.min, mx_finfo_obj.smallest_normal) == \
        (np_finfo.bits, np_finfo.eps, np_finfo.max, np_finfo.min, np_finfo.tiny)


@use_np
@pytest.mark.parametrize('dtype', np.integer_dtypes)
def test_np_iinfo(dtype):
    mx_iinfo_obj = np.iinfo(dtype)
    np_iinfo = onp.iinfo(dtype)
    assert (mx_iinfo_obj.bits, mx_iinfo_obj.max, mx_iinfo_obj.min) == \
        (np_iinfo.bits, np_iinfo.max, np_iinfo.min)


@use_np
@pytest.mark.parametrize('input1', [d for d in np.numeric_dtypes + np.boolean_dtypes] + [np.ones((1,), dtype=d) for d in np.numeric_dtypes + np.boolean_dtypes])
@pytest.mark.parametrize('input2', [d for d in np.numeric_dtypes + np.boolean_dtypes])
def test_np_can_cast(input1, input2):
    np_input1 = input1
    np_input2 = input2
    if isinstance(input1, np.ndarray):
        np_input1 = input1.asnumpy()
    assert np.can_cast(input1, input2) == onp.can_cast(np_input1, np_input2)


@use_np
@pytest.mark.parametrize('nums', [1, 2, 3, 4, 10, 100])
def test_np_result_type(nums):
    PICK_LIST = np.numeric_dtypes + np.boolean_dtypes + [np.ones((1,), dtype=d) for d in np.numeric_dtypes + np.boolean_dtypes]
    import random
    inputs = [random.choice(PICK_LIST) for _ in range(nums)]

    try:
        promoted = np.result_type(*inputs)
    except Exception as e:
        with pytest.raises(TypeError):
            promoted = np.result_type(*inputs)


@use_np
@pytest.mark.parametrize('func,func2,dtypes,ref_grad,low,high', [
    ('abs', 'abs', 'numeric', lambda x: -1. * (x < 0) + (x > 0), -1.0, 1.0),
    ('acos', 'arccos', 'floating-point', lambda x: -1. / (1. - x ** 2.) ** (1. / 2.), -1.0, 1.0),
    ('acosh', 'arccosh', 'floating-point', lambda x: 1./(x**2 - 1.)**(1./2.), 2.0, 5.0),
    ('asin', 'arcsin', 'floating-point', lambda x: 1. / (1. - x ** 2) ** (1. / 2.), -1.0, 1.0),
    ('asinh', 'arcsinh', 'floating-point', lambda x: 1./(x**2 + 1.)**(1./2.), -1.0, 1.0),
    ('atan', 'arctan', 'floating-point', lambda x: 1. / (x ** 2. + 1.), -1.0, 1.0),
    ('atanh', 'arctanh', 'floating-point', lambda x: -1./(x**2 - 1.), -0.99, 0.99),
    ('bitwise_invert', 'invert', 'integer or boolean', None, -5, 5),
    ('ceil', 'ceil', 'numeric', None, -10.0, 10.0),
    ('cos', 'cos', 'floating-point', lambda x: -onp.sin(x), -1.0, 1.0),
    ('cosh', 'cosh', 'floating-point', lambda x: onp.sinh(x), -1.0, 1.0),
    ('exp', 'exp', 'floating-point', lambda x: onp.exp(x), -1.0, 1.0),
    ('expm1', 'expm1', 'floating-point', lambda x: onp.exp(x), -1.0, 1.0),
    ('floor', 'floor', 'numeric', None, -10.0, 10.0),
    ('log', 'log', 'floating-point', lambda x: 1.0 / x, 0.1, 5.0),
    ('log10', 'log10', 'floating-point', lambda x: 1.0 / (x * onp.log(10)), 0.1, 10.0),
    ('log1p', 'log1p', 'floating-point', lambda x: 1.0 / (1.0 + x), -0.9, 5.0),
    ('log2', 'log2', 'floating-point', lambda x: 1.0 / (x * onp.log(2)), 0.1, 2.0),
    ('logical_not', 'logical_not', 'boolean', None,  -1.0, 1.0),
    ('negative', 'negative', 'numeric', lambda x: -1. * onp.ones(x.shape), -1.0, 1.0),
    ('positive', 'positive', 'numeric', lambda x: onp.ones(x.shape), -1.0, 1.0),
    ('sign', 'sign', 'numeric', None, -1.0, 1.0),
    ('sin', 'sin', 'floating-point', lambda x: onp.cos(x), -1.0, 1.0),
    ('sinh', 'sinh', 'floating-point', lambda x: onp.cosh(x), -1.0, 1.0),
    ('sqrt', 'sqrt', 'floating-point', lambda x: 0.5 / onp.sqrt(x), 0.001, 10.0),
    ('square', 'square', 'numeric', lambda x: 2.0 * x, -1.0, 1.0),
    ('tan', 'tan', 'floating-point', lambda x: onp.tan(x) ** 2 + 1.0, -1.0, 1.0),
    ('tanh', 'tanh', 'floating-point', lambda x: 1. - onp.tanh(x) ** 2, -1.0, 1.0),
    ('trunc', 'trunc', 'numeric', None, -5.0, 5.0),
])
@pytest.mark.parametrize('ndim', [2, 3, 4])
def test_np_standard_unary_funcs(func, func2, dtypes, ref_grad, low, high, ndim):
    class TestStandardUnary(HybridBlock):
        def __init__(self, func):
            super(TestStandardUnary, self).__init__()
            self._func = func

        def forward(self, a):
            return getattr(np, self._func)(a)

    type_mapping = {
        'floating-point': np.floating_dtypes,
        'numeric': np.numeric_dtypes,
        'integer or boolean': np.integer_dtypes + np.boolean_dtypes,
        'boolean': np.boolean_dtypes,
    }

    def array_values(low, high, shape):
        for d in np.integer_dtypes + np.boolean_dtypes + np.floating_dtypes:
            yield onp.random.uniform(low, high, shape).astype(d), d


    shapes = [i for i in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]]
    for shape in shapes:
        for (np_test_data, dtype) in array_values(low, high, shape):
            if dtype in type_mapping[dtypes]:
                rtol = 1e-2 if dtype == np.float16 else 1e-3
                atol = 1e-4 if dtype == np.float16 else 1e-5
                # get rid of warning: divide by zero
                if((func=='log' or func=='log10' or func=='log2') and
                    (dtype=='int8' or dtype=='uint8' or dtype=='int32' or
                    dtype=='int64')):
                    low = 1
                if (func=='arctanh' and dtype=='bool'):
                    continue
                np_func = getattr(onp, func2)
                mx_func = TestStandardUnary(func)
                mx_test_data = np.array(np_test_data, dtype=dtype)
                for hybridize in [True, False]:
                    if hybridize:
                        mx_func.hybridize()
                    if ref_grad:
                        mx_test_data.attach_grad()
                    np_out = np_func(np_test_data)
                    with mx.autograd.record():
                        y = mx_func(mx_test_data)
                    assert y.shape == np_out.shape
                    assert_almost_equal(y.asnumpy(), np_out, rtol=1e-3, atol=atol)
                    if np_out.dtype == np.bool_:
                        assert y.dtype == np.bool_

                    if ref_grad and (dtype == 'float16' or dtype == 'float32' or dtype == 'float64'):
                        y.backward()
                        assert_almost_equal(mx_test_data.grad.asnumpy(), ref_grad(np_test_data), rtol=1e-1, atol=1e-2, equal_nan=True)

                np_func = getattr(onp, func2)
                mx_out = getattr(mx.np, func)(mx_test_data)
                assert mx_out.shape == np_out.shape
                assert np.result_type(mx_out) == dtype
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=1e-5)

                assertRaises(NotImplementedError, getattr(np, func), mx_test_data, where=False)
                assertRaises(NotImplementedError, getattr(np, func), mx_test_data, subok=False)
                assertRaises(NotImplementedError, getattr(np, func), mx_test_data, dtype=onp.int8)
                assertRaises(TypeError, getattr(np, func), mx_test_data, dtype="abcdefg")
                assertRaises(NotImplementedError, getattr(np, func), mx_test_data, casting='safe')
                assertRaises(TypeError, getattr(np, func), mx_test_data, casting='mxnet')
                assertRaises(NotImplementedError, getattr(np, func), mx_test_data, order='C')
                assertRaises(NotImplementedError, getattr(np, func), mx_test_data, order='mxnet')


@use_np
@pytest.mark.flaky
@pytest.mark.parametrize('func,func2,promoted,dtypes,ref_grad_a,ref_grad_b,low,high', [
    ('add', 'add', True, 'numeric', lambda y, x1, x2: onp.ones(y.shape), None, -1.0, 1.0),
    ('atan2', 'arctan2', True, 'floating-point', lambda y, x1, x2: x2 / (onp.square(x1) + onp.square(x2)),
                                                 lambda y, x1, x2: -x1 / (onp.square(x1) + onp.square(x2)), -1, 1),
    ('bitwise_and', 'bitwise_and', True, 'integer or boolean', None, None, -100, 100),
    ('bitwise_or', 'bitwise_or', True, 'integer or boolean', None, None, -100, 100),
    ('bitwise_xor', 'bitwise_xor', True, 'integer or boolean', None, None, -100, 100),
    ('divide', 'divide', True, 'floating-point', lambda y, x1, x2: onp.ones(y.shape) / x2,
                                                 lambda y, x1, x2: -x1 / (x2 * x2), 0.1, 1.0),
    ('equal', 'equal', False, 'all', None, None, 0.0, 2.0),
    ('floor_divide', 'floor_divide', True, 'numeric', lambda y, x1, x2: onp.zeros(y.shape),
                                                      lambda y, x1, x2: onp.zeros(y.shape), 2.0, 10.0),
    ('greater', 'greater', False, 'numeric', None, None, 0.0, 2.0),
    ('greater_equal', 'greater_equal', False, 'numeric', None, None, 0.0, 2.0),
    ('less', 'less', False, 'numeric', None, None, 0.0, 2.0),
    ('less_equal', 'less_equal', False, 'numeric', None, None, 0.0, 2.0),
    ('logaddexp', 'logaddexp', True, 'floating-point', lambda y, x1, x2: onp.exp(x1) / (onp.exp(x1) + onp.exp(x2)),
                                                       lambda y, x1, x2: onp.exp(x2) / (onp.exp(x1) + onp.exp(x2)), -10, 10),
    ('logical_and', 'logical_and', False, 'boolean', None, None, -100, 100),
    ('logical_or', 'logical_or', False, 'boolean', None, None, -100, 100),
    ('logical_xor', 'logical_xor', False, 'boolean', None, None, -100, 100),
    ('multiply', 'multiply', True, 'numeric', lambda y, x1, x2: onp.broadcast_to(x2, y.shape),
                                              lambda y, x1, x2: onp.broadcast_to(x1, y.shape), -1.0, 1.0),
    ('not_equal', 'not_equal', False, 'all', None, None, 0.0, 2.0),
    ('pow', 'power', True, 'floating-point', lambda y, x1, x2: onp.power(x1, x2 - 1.0) * x2,
                                             lambda y, x1, x2: onp.power(x1, x2) * onp.log(x1), 1.0, 3.0),
    ('subtract', 'subtract', True, 'numeric', lambda y, x1, x2: onp.ones(y.shape),
                                              lambda y, x1, x2: -onp.ones(y.shape), -1.0, 1.0),
])
@pytest.mark.parametrize('lshape,rshape', [
    ((3, 2), (3, 2)),
    ((3, 2), (3, 1)),
    ((3, 1), (3, 0)),
    ((0, 2), (1, 2)),
    ((2, 3, 4), (3, 1)),
# MXNet numpy does not match original numpy behavior when broadcasting 0-dim arrays.
# See https://github.com/apache/incubator-mxnet/issues/20898.
#    ((2, 3), ()),
#    ((), (2, 3))
    ((2, 3), (1,)),
    ((1,), (2, 3))
])
def test_np_standard_binary_funcs(func, func2, promoted, dtypes, ref_grad_a, ref_grad_b, low, high, lshape, rshape):
    class TestStandardBinary(HybridBlock):
        def __init__(self, func):
            super(TestStandardBinary, self).__init__()
            self._func = func

        def forward(self, a, b,):
            return getattr(np, self._func)(a, b)

    type_mapping = {
        'floating-point': np.floating_dtypes,
        'numeric': np.numeric_dtypes,
        'integer or boolean': np.integer_dtypes + np.boolean_dtypes,
        'boolean': np.boolean_dtypes,
        'all': np.numeric_dtypes + np.boolean_dtypes,
    }

    def array_values(low, high, shape):
        for d in np.integer_dtypes + np.boolean_dtypes + np.floating_dtypes:
            yield onp.random.uniform(low, high, shape).astype(d), d


    for (left_value, ltype) in array_values(low, high, lshape):
        for (right_value, rtype) in array_values(low, high, rshape):
            if ltype in type_mapping[dtypes] and rtype in type_mapping[dtypes]:
                try:
                    promote_type = np.result_type(ltype, rtype)
                except Exception as e:
                    # Unkown type promotion between two types
                    continue
                rtol = 1e-2 if ltype == np.float16 or rtype == np.float16 else 1e-3
                atol = 1e-4 if ltype == np.float16 or rtype == np.float16 else 1e-5
                mx_left_value = np.array(left_value, dtype=ltype)
                mx_right_value = np.array(right_value, dtype=rtype)
                mx_func = TestStandardBinary(func)
                np_func = getattr(onp, func2)
                for hybridize in [True, False]:
                    if hybridize:
                        mx_func.hybridize()
                    if ref_grad_a:
                        mx_left_value.attach_grad()
                        mx_right_value.attach_grad()
                    np_out = np_func(left_value, right_value)
                    with mx.autograd.record():
                        y = mx_func(mx_left_value, mx_right_value)
                    assert y.shape == np_out.shape
                    assert_almost_equal(y.asnumpy(), np_out.astype(y.dtype), rtol=rtol, atol=atol,
                                        use_broadcast=False, equal_nan=True)

                    if ref_grad_a and ltype in np.floating_dtypes and rtype in np.floating_dtypes:
                        y.backward()
                        assert_almost_equal(mx_left_value.grad.asnumpy(),
                                            collapse_sum_like(ref_grad_a(y.asnumpy(), left_value, right_value), mx_left_value.shape),
                                            rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)
                        if ref_grad_b is None:
                            assert_almost_equal(mx_right_value.grad.asnumpy(),
                                                collapse_sum_like(ref_grad_a(y.asnumpy(), right_value, left_value), mx_right_value.shape),
                                                rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)
                        else:
                            assert_almost_equal(mx_right_value.grad.asnumpy(),
                                                collapse_sum_like(ref_grad_b(y.asnumpy(), left_value, right_value), mx_right_value.shape),
                                                rtol=1e-1, atol=1e-2, equal_nan=True, use_broadcast=False)

                np_out = getattr(onp, func2)(left_value, right_value)
                mx_out = getattr(np, func)(mx_left_value, mx_right_value)
                assert mx_out.shape == np_out.shape
                if promoted:
                    assert np.result_type(ltype, rtype) == mx_out.dtype
                else:
                    assert mx_out.dtype == np.bool_
                assert_almost_equal(mx_out.asnumpy(), np_out.astype(mx_out.dtype), rtol=rtol, atol=atol,
                                    use_broadcast=False, equal_nan=True)

