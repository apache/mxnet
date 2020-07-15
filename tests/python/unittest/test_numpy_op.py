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
import sys
import unittest
import itertools
import numpy as _np
import platform
import mxnet as mx
import scipy.stats as ss
from nose.tools import assert_raises
from mxnet import np, npx
from mxnet.gluon import HybridBlock
from mxnet.base import MXNetError
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient, use_np, collapse_sum_like
from common import assertRaises, with_seed
import random
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf
from mxnet.numpy_op_signature import _get_builtin_op
from mxnet.test_utils import is_op_runnable, has_tvm_ops
from mxnet.operator import get_all_registered_operators


@with_seed()
@use_np
def test_np_tensordot():
    class TestTensordot(HybridBlock):
        def __init__(self, axes):
            super(TestTensordot, self).__init__()
            self._axes = axes

        def hybrid_forward(self, F, a, b):
            return F.np.tensordot(a, b, self._axes)

    def tensordot_backward(a, b, axes=2):
        if (a.ndim < 1) or (b.ndim < 1):
            raise ValueError('An input is zero-dim')

        if _np.isscalar(axes):
            a_axes_summed = [i + a.ndim - axes for i in range(axes)]
            b_axes_summed = [i for i in range(axes)]
        else:
            if len(axes) != 2:
                raise ValueError('Axes must consist of two arrays.')
            a_axes_summed, b_axes_summed = axes
            if _np.isscalar(a_axes_summed):
                a_axes_summed = a_axes_summed,
            if _np.isscalar(b_axes_summed):
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

        ad1 = _np.prod([a.shape[i] for i in a_axes_remained]) if len(a_axes_remained) > 0 else 1
        ad2 = _np.prod([a.shape[i] for i in a_axes_summed]) if len(a_axes_summed) > 0 else 1
        bd1 = _np.prod([b.shape[i] for i in b_axes_summed]) if len(b_axes_summed) > 0 else 1
        bd2 = _np.prod([b.shape[i] for i in b_axes_remained]) if len(b_axes_remained) > 0 else 1

        out_grad = _np.ones((ad1, bd2))

        new_a = _np.transpose(a, a_axes)
        new_a_shape = new_a.shape[:]
        new_a = new_a.reshape((ad1, ad2))
        new_b = _np.transpose(b, b_axes)
        new_b_shape = new_b.shape[:]
        new_b = new_b.reshape((bd1, bd2))

        reverse_a_axes = [0 for i in a_axes]
        for i in range(len(a_axes)):
            reverse_a_axes[a_axes[i]] = i

        reverse_b_axes = [0 for i in b_axes]
        for i in range(len(b_axes)):
            reverse_b_axes[b_axes[i]] = i

        grad_b = _np.dot(new_a.T, out_grad).reshape(new_b_shape)
        grad_b = _np.transpose(grad_b, reverse_b_axes)
        grad_a = _np.dot(out_grad, new_b.T).reshape(new_a_shape)
        grad_a = _np.transpose(grad_a, reverse_a_axes)

        return [grad_a, grad_b]

    # test non zero size input
    tensor_shapes = [
        ((3, 5), (5, 4), 1),  # (a_shape, b_shape, axes)
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
    ]

    for hybridize in [True, False]:
        for a_shape, b_shape, axes in tensor_shapes:
            for dtype in [_np.float32, _np.float64]:
                test_tensordot = TestTensordot(axes)
                if hybridize:
                    test_tensordot.hybridize()
                a = rand_ndarray(shape = a_shape, dtype = dtype).as_np_ndarray()
                b = rand_ndarray(shape = b_shape, dtype = dtype).as_np_ndarray()
                a.attach_grad()
                b.attach_grad()

                np_out = _np.tensordot(a.asnumpy(), b.asnumpy(), axes)
                with mx.autograd.record():
                    mx_out = test_tensordot(a, b)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)
                mx_out.backward()
                np_backward = tensordot_backward(a.asnumpy(), b.asnumpy(), axes)
                assert_almost_equal(a.grad.asnumpy(), np_backward[0], rtol = 1e-3, atol=1e-5)
                assert_almost_equal(b.grad.asnumpy(), np_backward[1], rtol = 1e-3, atol=1e-5)

                # Test imperative once again
                mx_out = np.tensordot(a, b, axes)
                np_out = _np.tensordot(a.asnumpy(), b.asnumpy(), axes)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                # test numeric gradient
                if (_np.prod(a_shape) > 0 and _np.prod(b_shape) > 0):
                    a_sym = mx.sym.Variable("a").as_np_ndarray()
                    b_sym = mx.sym.Variable("b").as_np_ndarray()
                    mx_sym = mx.sym.np.tensordot(a_sym, b_sym, axes).as_nd_ndarray()
                    check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                      rtol=1e-1, atol=1e-1, dtype = dtype)


@with_seed()
@use_np
def test_np_dot():
    shapes = [
        ((3, 0), (0, 4)),
        ((3,), (3,)),        # Case 1
        ((3, 4), (4, 5)),    # Case 2
        ((), ()),            # Case 3
        ((3, 4, 5), ()),     # Case 3.5.1
        ((), (3, 4, 5)),     # Case 3.5.2
        ((3, 4, 5), (5, )),  # Case 4
        ((3, 4, 5), (5, 2)), # Case 5
        ((5,), (5, 2)),
        ((3, 5, 4), (5, 4, 3)),
        ((3, 4), (5, 4, 3)),
        ((4,), (5, 4, 3))
    ]

    eps = 1e-3

    for shape_a, shape_b in shapes:
        np_a = _np.random.uniform(-1.0, 1.0, shape_a)
        np_a[abs(np_a) < eps] = 2 * eps
        np_b = _np.random.uniform(-1.0, 1.0, shape_b)
        np_b[abs(np_b) < eps] = 2 * eps
        a = mx.nd.array(np_a)
        b = mx.nd.array(np_b)
        np_res = _np.dot(np_a, np_b)
        mx_res = np.dot(a.as_np_ndarray(), b.as_np_ndarray())
        assert mx_res.shape == np_res.shape
        assert_almost_equal(np_res, mx_res.asnumpy(), rtol=1e-5, atol=1e-5)
        mx_a = mx.sym.Variable("a")
        mx_b = mx.sym.Variable("b")
        mx_sym = mx.sym.np.dot(mx_a.as_np_ndarray(), mx_b.as_np_ndarray()).as_nd_ndarray()
        if (len(shape_a) > 0 and len(shape_b) > 0 and _np.prod(shape_a) > 0 and _np.prod(shape_b) > 0):
            check_numeric_gradient(mx_sym, {"a": a, "b": b}, numeric_eps=eps, rtol=1e-2, atol=1e-3)

    bad_shapes = [((4, 5), (2, 3)), ((3, 4, 5), (6, ))]

    for shape_a, shape_b in bad_shapes:
        a = mx.nd.array(random.random()) if len(shape_a) == 0 else rand_ndarray(shape_a)
        b = mx.nd.array(random.random()) if len(shape_b) == 0 else rand_ndarray(shape_b)
        try:
            mx_res = np.dot(a.as_np_ndarray(), b.as_np_ndarray())
        except mx.base.MXNetError:
            continue
        assert False


@with_seed()
@use_np
def test_np_vdot():
    class TestVdot(HybridBlock):
        def __init__(self):
            super(TestVdot, self).__init__()

        def hybrid_forward(self, F, a, b):
            return F.np.vdot(a, b)

    def vdot_backward(a, b):
        return [b, a]

    # test different size inputs
    tensor_shapes = [(), (5,), (3, 3)]

    for hybridize in [True, False]:
        for shape in tensor_shapes:
            for dtype in [_np.float32, _np.float64]:
                test_vdot = TestVdot()
                if hybridize:
                    test_vdot.hybridize()
                a = rand_ndarray(shape=shape, dtype=dtype).as_np_ndarray()
                b = rand_ndarray(shape=shape, dtype=dtype).as_np_ndarray()
                a.attach_grad()
                b.attach_grad()

                np_out = _np.vdot(a.asnumpy(), b.asnumpy())
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
                np_out = _np.vdot(a.asnumpy(), b.asnumpy())
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                # test numeric gradient
                if len(shape) > 0 and _np.prod(shape) > 0:
                    a_sym = mx.sym.Variable("a").as_np_ndarray()
                    b_sym = mx.sym.Variable("b").as_np_ndarray()
                    mx_sym = mx.sym.np.vdot(a_sym, b_sym).as_nd_ndarray()
                    check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                      rtol=1e-1, atol=1e-1, dtype=dtype)


@with_seed()
@use_np
def test_np_inner():
    class TestInner(HybridBlock):
        def __init__(self):
            super(TestInner, self).__init__()

        def hybrid_forward(self, F, a, b):
            return F.np.inner(a, b)

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

        ad1 = _np.prod([a.shape[i] for i in a_axes_remained]) if len(a_axes_remained) > 0 else 1
        ad2 = _np.prod([a.shape[i] for i in a_axes_summed]) if len(a_axes_summed) > 0 else 1
        bd1 = _np.prod([b.shape[i] for i in b_axes_summed]) if len(b_axes_summed) > 0 else 1
        bd2 = _np.prod([b.shape[i] for i in b_axes_remained]) if len(b_axes_remained) > 0 else 1

        out_grad = _np.ones((ad1, bd2))

        new_a = _np.transpose(a, a_axes)
        new_a_shape = new_a.shape[:]
        new_a = new_a.reshape((ad1, ad2))
        new_b = _np.transpose(b, b_axes)
        new_b_shape = new_b.shape[:]
        new_b = new_b.reshape((bd1, bd2))

        reverse_a_axes = [0 for i in a_axes]
        for i in range(len(a_axes)):
            reverse_a_axes[a_axes[i]] = i

        reverse_b_axes = [0 for i in b_axes]
        for i in range(len(b_axes)):
            reverse_b_axes[b_axes[i]] = i

        grad_b = _np.dot(new_a.T, out_grad).reshape(new_b_shape)
        grad_b = _np.transpose(grad_b, reverse_b_axes)
        grad_a = _np.dot(out_grad, new_b.T).reshape(new_a_shape)
        grad_a = _np.transpose(grad_a, reverse_a_axes)

        return [grad_a, grad_b]

    # test non zero size input
    tensor_shapes = [
        ((3,), (3,)),
        ((2, 3), (3,)),
        ((3,), (2, 3))
    ]

    for hybridize in [True, False]:
        for a_shape, b_shape in tensor_shapes:
            for dtype in [_np.float32, _np.float64]:
                test_inner = TestInner()
                if hybridize:
                    test_inner.hybridize()
                a = rand_ndarray(shape=a_shape, dtype=dtype).as_np_ndarray()
                b = rand_ndarray(shape=b_shape, dtype=dtype).as_np_ndarray()
                a.attach_grad()
                b.attach_grad()

                np_out = _np.inner(a.asnumpy(), b.asnumpy())
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
                np_out = _np.inner(a.asnumpy(), b.asnumpy())
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                # test numeric gradient
                a_sym = mx.sym.Variable("a").as_np_ndarray()
                b_sym = mx.sym.Variable("b").as_np_ndarray()
                mx_sym = mx.sym.np.inner(a_sym, b_sym).as_nd_ndarray()
                check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                  rtol=1e-1, atol=1e-1, dtype=dtype)


@with_seed()
@use_np
def test_np_outer():
    class TestOuter(HybridBlock):
        def __init__(self):
            super(TestOuter, self).__init__()

        def hybrid_forward(self, F, a, b):
            return F.np.outer(a, b)

    # test non zero size input
    tensor_shapes = [
        ((3,), (3,)),
        ((2, 3), (6,)),
        ((6,), (2, 3))
    ]

    for hybridize in [True, False]:
        for a_shape, b_shape in tensor_shapes:
            for dtype in [_np.float32, _np.float64]:
                test_outer = TestOuter()
                if hybridize:
                    test_outer.hybridize()
                a = rand_ndarray(shape=a_shape, dtype=dtype).as_np_ndarray()
                b = rand_ndarray(shape=b_shape, dtype=dtype).as_np_ndarray()
                a.attach_grad()
                b.attach_grad()

                np_out = _np.outer(a.asnumpy(), b.asnumpy())
                with mx.autograd.record():
                    mx_out = test_outer(a, b)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                mx_out.backward()

                # Test imperative once again
                mx_out = np.outer(a, b)
                np_out = _np.outer(a.asnumpy(), b.asnumpy())
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                # test numeric gradient
                a_sym = mx.sym.Variable("a").as_np_ndarray()
                b_sym = mx.sym.Variable("b").as_np_ndarray()
                mx_sym = mx.sym.np.outer(a_sym, b_sym).as_nd_ndarray()
                check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                                       rtol=1e-1, atol=1e-1, dtype=dtype)


@with_seed()
@use_np
def test_np_sum():
    class TestSum(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestSum, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.sum(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' in dtype

    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    acc_type = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64',
                'int8': 'int32', 'int32': 'int64', 'int64': 'int64', 'bool': 'int64'}
    is_windows = sys.platform.startswith('win')
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool']:
                    for dtype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                        if (is_int(dtype) and not is_int(itype)) or (is_windows and is_int(itype))\
                                or (itype == 'bool' and\
                                    (dtype not in ('float32', 'float64', 'int32', 'int64') or is_windows)):
                            continue
                        # test gluon
                        test_sum = TestSum(axis=axis, dtype=dtype, keepdims=keepdims)
                        if hybridize:
                            test_sum.hybridize()
                        if is_int(itype):
                            x = _np.random.randint(-128, 128, shape, dtype=itype)
                            x = np.array(x)
                        elif itype == 'bool':
                            x = _np.random.randint(0, 2, shape) < 1
                            x = np.array(x, dtype='bool')
                        else:
                            x = np.random.uniform(-1.0, 1.0, size=shape, dtype=itype)
                        expected_ret = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        if itype == 'bool':
                            if is_op_runnable() and (not is_windows):  # special handling of boolean ndarray
                                y = test_sum(x)
                                assert y.dtype == expected_ret.dtype
                                assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-4, atol=1e-5,
                                                    use_broadcast=False)
                            continue

                        x.attach_grad()
                        with mx.autograd.record():
                            y = test_sum(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                            atol=1e-5 if dtype == 'float16' else 1e-5, use_broadcast=False)

                        y.backward()
                        assert same(x.grad.asnumpy(), _np.ones(shape=x.shape, dtype=x.dtype))

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.sum(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-2, atol=1e-3, dtype=_np.float32)

                        # test imperative
                        mx_out = np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_max_min():
    class TestMax(HybridBlock):
        def __init__(self, axis=None, keepdims=False):
            super(TestMax, self).__init__()
            self._axis = axis
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return a.max(axis=self._axis, keepdims=self._keepdims)

    class TestMin(HybridBlock):
        def __init__(self, axis=None, keepdims=False):
            super(TestMin, self).__init__()
            self._axis = axis
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return a.min(axis=self._axis, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' == dtype

    def get_grad(axis, func_name):
        index = -1 if func_name == 'max' else 0
        if axis == ():
            return _np.ones((2,3,4,5))
        else:
            temp = _np.zeros((2,3,4,5))
            if axis == 0:
                temp[index,:,:,:] = 1
                return temp
            elif axis == 1:
                temp[:,index,:,:] = 1
                return temp
            elif axis == 2:
                temp[:,:,index,:] = 1
                return temp
            elif axis == 3:
                temp[:,:,:,index] = 1
                return temp
            elif not axis:
                temp[index,index,index,index] = 1
                return temp
            raise ValueError('axis should be int or None or ()')

    def _test_np_exception(func, shape, dim):
        x = np.random.uniform(-1.0, 1.0, shape)
        out = getattr(x, func)()
        assert out.ndim == dim, 'dimension mismatch, output.ndim={}, dim={}'.format(output.ndim, dim)

    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    for func in ['max', 'min']:
        for hybridize in [False, True]:
            for keepdims in [True, False]:
                for axis in ([i for i in range(in_data_dim)] + [(), None]):
                    for itype in ['float16', 'float32', 'float64', 'int']:
                        # test gluon
                        if func == 'max':
                            test_gluon = TestMax(axis=axis, keepdims=keepdims)
                        else:
                            test_gluon = TestMin(axis=axis, keepdims=keepdims)
                        if hybridize:
                            test_gluon.hybridize()
                        if is_int(itype):
                            x = np.arange(120).reshape((2, 3, 4, 5))
                        else:
                            x = np.random.uniform(-1.0, 1.0, size=shape, dtype=itype)
                        x.attach_grad()
                        if func == 'max':
                            expected_ret = _np.amax(x.asnumpy(), axis=axis, keepdims=keepdims)
                        else:
                            expected_ret = _np.amin(x.asnumpy(), axis=axis, keepdims=keepdims)
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
                        if func == 'max':
                            mx_out = np.max(x, axis=axis, keepdims=keepdims)
                            np_out = _np.amax(x.asnumpy(), axis=axis, keepdims=keepdims)
                        else:
                            mx_out = np.min(x, axis=axis, keepdims=keepdims)
                            np_out = _np.amin(x.asnumpy(), axis=axis, keepdims=keepdims)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    # test zero and zero dim
    shapes = [(), (0), (2, 0), (0, 2, 1)]
    exceptions = [False, True, True, True]
    dims = [0] * len(shapes)
    for func in ['max', 'min']:
        for shape, exception, dim in zip(shapes, exceptions, dims):
            if exception:
                assertRaises(MXNetError, _test_np_exception, func, shape, dim)
            else:
                _test_np_exception(func, shape, dim)


@with_seed()
@use_np
def test_np_mean():
    class TestMean(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestMean, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
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

                    expected_ret = _np.mean(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims)
                    expected_ret = expected_ret.astype(dtype)
                    with mx.autograd.record():
                        y = test_mean(x)
                    assert y.shape == expected_ret.shape
                    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                        atol=1e-5 if dtype == 'float16' else 1e-5)

                    y.backward()
                    N = x.size / y.size
                    assert same(x.grad.asnumpy(), _np.ones(shape=x.shape, dtype=x.dtype) / N)

                    # test numeric
                    if itype == 'float32' and dtype == 'float32':
                        x_sym = mx.sym.Variable("x").as_np_ndarray()
                        mx_sym = mx.sym.np.mean(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                        check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                               numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                    # test imperative
                    mx_out = np.mean(x, axis=axis, dtype=dtype, keepdims=keepdims)
                    np_out = _np.mean(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                for itype, dtype in itertools.product(it_types, [None] + ft_types + it_types):
                    if dtype == 'bool':
                        continue
                    # test gluon
                    test_mean = TestMean(axis=axis, dtype=dtype, keepdims=keepdims)
                    if hybridize:
                        test_mean.hybridize()

                    if itype == 'bool':
                        x = np.array(_np.random.uniform(size=shape) > 0.5)
                    else:
                        x = np.random.uniform(-128, 127, size=shape).astype(itype)

                    expected_ret = _np.mean(x.asnumpy(), axis=axis, dtype=dtype, keepdims=keepdims)

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
                    np_out = _np.mean(x.asnumpy(), axis=axis, dtype=dtype, keepdims=keepdims).astype(dtype)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_moment():
    class TestMoment(HybridBlock):
        def __init__(self, name, axis=None, dtype=None, keepdims=False, ddof=0):
            super(TestMoment, self).__init__()
            self._name = name
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims
            self._ddof = ddof

        def hybrid_forward(self, F, a, *args, **kwargs):
            return getattr(a, self._name)(axis=self._axis, dtype=self._dtype, keepdims=self._keepdims, ddof=self._ddof)

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
                                    x = _np.random.randint(-16, 16, shape, dtype=itype)
                                    x = mx.nd.array(x)
                                else:
                                    x = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype=itype)
                                x = x.as_np_ndarray()
                                x.attach_grad()
                                expected_ret = getattr(_np, name)(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims, ddof=ddof)
                                expected_ret = expected_ret.astype(dtype)
                                y = test_moment(x)
                                assert y.shape == expected_ret.shape
                                assert_almost_equal(y.asnumpy(), expected_ret, rtol=rtol, atol=atol, use_broadcast=False, equal_nan=True)

                                # test imperative
                                mx_out = getattr(np, name)(x, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
                                np_out = getattr(_np, name)(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims, ddof=ddof).astype(dtype)
                                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol, use_broadcast=False, equal_nan=True)

@with_seed()
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
        np_a = _np.random.uniform(size=shape)

        mx_shape = np.shape(mx_a)
        np_shape = _np.shape(np_a)

        assert mx_shape == np_shape


@with_seed()
@use_np
def test_np_linspace():
    configs = [
        (0.0, 1.0, 10),
        (-2, 4, 30),
        (5.234324, 8.98324, 324),
        (2, 10, 100)
    ]
    exception_configs = [
        (0, 10, -1),
        (0, 1, 2.5)
    ]
    dtypes = ['int32', 'float16', 'float32', 'float64', None]
    for config in configs:
        for dtype in dtypes:
            for endpoint in [False, True]:
                for retstep in [False, True]:
                    if isinstance(config, tuple):
                        mx_ret = np.linspace(*config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                        np_ret = _np.linspace(*config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                    else:
                        mx_ret = np.linspace(config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                        np_ret = _np.linspace(config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                    if retstep:
                        assert_almost_equal(mx_ret[0].asnumpy(), np_ret[0], atol=1e-3, rtol=1e-5)
                        same(mx_ret[1], np_ret[1])
                    else:
                        assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-3, rtol=1e-5)
    # check for exception input
    for config in exception_configs:
        assertRaises(MXNetError, np.linspace, *config)
    # check linspace equivalent to arange
    for test_index in range(1000):
        assert_almost_equal(mx.np.linspace(0, test_index, test_index + 1).asnumpy(), _np.arange(test_index + 1))

    class TestLinspace(HybridBlock):
        def __init__(self, start, stop, num=50, endpoint=None, retstep=False, dtype=None, axis=0):
            super(TestLinspace, self).__init__()
            self._start = start
            self._stop = stop
            self._num = num
            self._endpoint = endpoint
            self._retstep = retstep
            self._dtype = dtype

        def hybrid_forward(self, F, x):
            if self._retstep:
                raise ValueError("linspace didn't support retstep = True inside HybridBlock")
            else:
                return x + F.np.linspace(self._start, self._stop, self._num, \
                self._endpoint, self._retstep, self._dtype)

    for dtype in dtypes:
        x = np.zeros(shape=(), dtype=dtype)
        for config in configs:
            for hybridize in [False, True]:
                for endpoint in [False, True]:
                    if isinstance(config, tuple):
                        net = TestLinspace(*config, endpoint=endpoint, dtype=dtype)
                        np_out = _np.linspace(*config, endpoint=endpoint, dtype=dtype)
                    else:
                        net = TestLinspace(config, endpoint=endpoint, dtype=dtype)
                        np_out = _np.linspace(config, endpoint=endpoint, dtype=dtype)
                    if hybridize:
                        net.hybridize()
                    mx_out = net(x)
                    assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-3, rtol=1e-5)


@with_seed()
@use_np
def test_np_logspace():
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

        def hybrid_forward(self, F, x):
            return x + F.np.logspace(self._start, self._stop, self._num, self._endpoint, self._base, self._dtype, self.axis)

    configs = [
        (0.0, 1.0, 20),
        (2, 8, 0),
        (22, 11, 1),
        (2.22, 9.99, 11),
        (4.99999, 12.11111111, 111)
    ]
    base_configs = [0, 1, 5, 8, 10, 33]
    dtypes = ['float32', 'float64', None]

    for config in configs:
        for dtype in dtypes:
            for endpoint in [False, True]:
                for hybridize in [False, True]:
                    for base in base_configs:
                        x = np.zeros(shape=(), dtype=dtype)
                        net = TestLogspace(*config, endpoint=endpoint, base=base, dtype=dtype)
                        np_out = _np.logspace(*config, endpoint=endpoint, base=base, dtype=dtype)
                        if hybridize:
                            net.hybridize()
                        mx_out = net(x)
                        assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-3, rtol=1e-5)
                        if dtype is not None:
                            assert mx_out.dtype == np_out.dtype

                        # Test imperative once again
                        mx_ret = np.logspace(*config, endpoint=endpoint, base=base, dtype=dtype)
                        np_ret = _np.logspace(*config, endpoint=endpoint, base=base, dtype=dtype)
                        assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-3, rtol=1e-5)
                        if dtype is not None:
                            assert mx_out.dtype == np_out.dtype


@with_seed()
@use_np
def test_npx_slice():
    class TestSlice(HybridBlock):
        def __init__(self, begin, end, step):
            super(TestSlice, self).__init__()
            self._begin = begin
            self._end = end
            self._step = step

        def hybrid_forward(self, F, a):
            return F.npx.slice(a, begin=self._begin, end=self._end, step=self._step)

    shape = (8, 16, 9, 9)
    np_array = _np.arange(_np.prod(shape), dtype='int32').reshape(shape)
    configs = [
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
    ]

    for hybridize in [True, False]:
        for config in configs:
            start, end, step = config[0], config[1], config[2]
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
            expected_grad = _np.zeros(shape)
            expected_grad[basic_index] = 1
            assert same(a.grad.asnumpy(), expected_grad)

@with_seed()
@use_np
def test_npx_batch_dot():
    ctx = mx.context.current_context()
    dtypes = ['float32', 'float64']
    if ctx.device_type == 'gpu':
        dtypes += ['float16']
    eps_dict = {'float32': 1E-4, 'float64': 1E-4, 'float16': 1E-3}
    class TestBatchDot(HybridBlock):
        def __init__(self, transpose_a, transpose_b):
            super(TestBatchDot, self).__init__()
            self._transpose_a = transpose_a
            self._transpose_b = transpose_b

        def hybrid_forward(self, F, lhs, rhs):
            return F.npx.batch_dot(lhs, rhs,
                                   transpose_a=self._transpose_a,
                                   transpose_b=self._transpose_b)

    def batch_dot_numpy(lhs, rhs, transpose_a, transpose_b):
        assert lhs.ndim == rhs.ndim >= 3
        if transpose_a:
            lhs = lhs.swapaxes(-1, -2)
        if transpose_b:
            rhs = rhs.swapaxes(-1, -2)
        return _np.matmul(lhs, rhs)

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
                        lhs_val = mx.np.array(_np.random.uniform(-1.0, 1.0, lhs_shape), dtype=dtype)
                        rhs_val = mx.np.array(_np.random.uniform(-1.0, 1.0, rhs_shape), dtype=dtype)
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
            lhs_val = mx.np.array(_np.random.uniform(-1.0, 1.0, lhs_shape), dtype=dtype)
            rhs_val = mx.np.array(_np.random.uniform(-1.0, 1.0, rhs_shape), dtype=dtype)
            assert_raises(MXNetError, lambda: mx.npx.batch_dot(lhs_val, rhs_val,
                                                               transpose_a=transpose_a,
                                                               transpose_b=transpose_b))


@with_seed()
@use_np
def test_npi_boolean_assign():
    class TestBooleanAssignScalar(HybridBlock):
        def __init__(self, val):
            super(TestBooleanAssignScalar, self).__init__()
            self._val = val

        def hybrid_forward(self, F, a, mask):
            return F.np._internal.boolean_mask_assign_scalar(a, mask, self._val, out=a)

    class TestBooleanAssignTensor(HybridBlock):
        def __init__(self):
            super(TestBooleanAssignTensor, self).__init__()

        def hybrid_forward(self, F, a, mask, value):
            return F.np._internal.boolean_mask_assign_tensor(a, mask, value, out=a)

    shapes = [(3, 4), (3, 0), ()]
    for hybridize in [False]:
        for shape in shapes:
            test_data = np.random.uniform(size=shape)
            mx_mask = np.around(np.random.uniform(size=shape))
            valid_num = int(mx_mask.sum())
            np_mask = mx_mask.asnumpy().astype(_np.bool)
            for val in [42., np.array(42.), np.array([42.]), np.random.uniform(size=(valid_num,))]:
                test_block = TestBooleanAssignScalar(val) if isinstance(val, float) else TestBooleanAssignTensor()
                if hybridize:
                    test_block.hybridize()
                np_data = test_data.asnumpy()
                mx_data = test_data.copy()
                np_data[np_mask] = val
                mx_data = test_block(mx_data, mx_mask) if isinstance(val, float) else test_block(mx_data, mx_mask, val)
                assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_reshape():
    class TestReshape(HybridBlock):
        def __init__(self, newshape):
            super(TestReshape, self).__init__()
            self._newshape = newshape

        def hybrid_forward(self, F, a):
            return F.np.reshape(a, self._newshape)

    shape_pairs = [((2, 6), (6, 2)), ((2, 6), (3, 4)), ((1, 0), (0,)), ((0, 0), (0,)), ((), (1, 1, 1))]
    for hybridize in [True, False]:
        for shape_pair in shape_pairs:
            shape1, shape2 = shape_pair
            test_reshape = TestReshape(shape2)
            if hybridize:
                test_reshape.hybridize()
            x = rand_ndarray(shape1).as_np_ndarray()
            x.attach_grad()
            np_out = _np.reshape(x.asnumpy(), shape2)
            with mx.autograd.record():
                mx_out = test_reshape(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)
            mx_out.backward()
            np_backward = _np.ones(shape1)
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5, use_broadcast=False)

            mx_out = np.reshape(x, shape2)
            np_out = _np.reshape(x.asnumpy(), shape2)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
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

        def hybrid_forward(self, F, x):
            return F.np.squeeze(x, self._axis)

    for shape, axis in config:
        data_np = _np.random.uniform(size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = _np.squeeze(data_np, axis)
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
            assert_almost_equal(data_mx.grad.asnumpy(), _np.ones_like(data_np),
                                rtol=1e-5, atol=1e-6, use_broadcast=False)


@with_seed()
@use_np
def test_np_prod():
    class TestProd(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestProd, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.prod(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

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
                        x = np.array(_np.random.uniform(-2.0, 2.0, size=shape), dtype=itype)
                        x.attach_grad()
                        expected_ret = _np.prod(x.asnumpy(), axis=axis, keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_prod(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5, use_broadcast=False)
                        y.backward()
                        # use keepdims=True so that broadcast divide can be used to calculate
                        # grad of input
                        expected_ret = _np.prod(x.asnumpy(), axis=axis, keepdims=True)
                        assert_almost_equal(x.grad.asnumpy(), expected_ret / x.asnumpy(), rtol=1e-3, atol=1e-3,
                                            use_broadcast=False)

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.prod(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.prod(x.asnumpy(), axis=axis, keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_flatten():
    class TestFlatten(HybridBlock):
        def hybrid_forward(self, F, x):
            return x.flatten()

    shapes = [(), (2, 0, 1), (3, 4, 5), 6, (0,), (0, 0, 0)]
    for shape in shapes:
        for hybridize in [True, False]:
            test_flatten = TestFlatten()
            if hybridize:
                test_flatten.hybridize()
            a_np = _np.random.uniform(size=shape).astype('float32')
            a_mx = np.array(a_np, dtype=a_np.dtype)
            a_mx.attach_grad()
            with mx.autograd.record():
                ret = test_flatten(a_mx)
            expected_ret = a_np.flatten()
            assert_almost_equal(expected_ret, ret.asnumpy(), rtol=1e-5, atol=1e-6, use_broadcast=False)
            # check gradient
            ret.backward()
            assert_almost_equal(a_mx.grad.asnumpy(), _np.ones_like(a_np), rtol=1e-5, atol=1e-6, use_broadcast=False)


@with_seed()
@use_np
def test_np_broadcast_to():
    class TestBroadcastTo(HybridBlock):
        def __init__(self, dst_shape):
            super(TestBroadcastTo, self).__init__()
            self._dst_shape = dst_shape

        def hybrid_forward(self, F, x):
            return F.np.broadcast_to(x, self._dst_shape)

    shapes = [
        ((), (1, 2, 4, 5)),
        ((1,), (4, 5, 6)),
        ((1, 0), (2, 4, 0)),
        ((1, 1), (2, 4, 0)),
        ((4, 1), (1, 2, 3, 4, 5)),
        ((4, 1), (1, 0, 3, 4, 5))
    ]
    for src_shape, dst_shape in shapes:
        for hybridize in [True, False]:
            test_broadcast_to = TestBroadcastTo(dst_shape)
            if hybridize:
                test_broadcast_to.hybridize()

            a = _np.random.uniform(size=src_shape).astype(np.float32)
            expected_ret = _np.broadcast_to(a, dst_shape)
            a_mx = np.array(a, dtype=a.dtype)
            a_mx.attach_grad()
            with mx.autograd.record():
                ret = test_broadcast_to(a_mx)
            assert_almost_equal(ret.asnumpy(), expected_ret, rtol=1e-5, atol=1e-6, use_broadcast=False)
            ret.backward()
            expected_grad = collapse_sum_like(_np.ones_like(expected_ret), src_shape)
            assert_almost_equal(a_mx.grad.asnumpy(), expected_grad, rtol=1e-5, atol=1e-6, use_broadcast=False)


@with_seed()
@use_np
def test_np_transpose():
    def np_transpose_grad(out_shape, dtype, axes=None):
        ograd = _np.ones(out_shape, dtype=dtype)
        if axes is None or axes == ():
            return _np.transpose(ograd, axes)
        np_axes = _np.array(list(axes))
        transpose_axes = _np.zeros_like(np_axes)
        transpose_axes[np_axes] = _np.arange(len(np_axes))
        return _np.transpose(ograd, tuple(list(transpose_axes)))

    class TestTranspose(HybridBlock):
        def __init__(self, axes=None):
            super(TestTranspose, self).__init__()
            self.axes = axes

        def hybrid_forward(self, F, a):
            return F.np.transpose(a, self.axes)
    test_workloads = [[(), [(), None]],
                      [(2,), [(0,), None]],
                      [(0, 2), [(0, 1), (1, 0)]],
                      [(5, 10), [(0, 1), (1, 0), None]],
                      [(8, 2, 3), [(2, 0, 1), (0, 2, 1), (0, 1, 2), (2, 1, 0), (-1, 1, 0), None]],
                      [(8, 2, 16), [(0, 2, 1), (2, 0, 1), (0, 1, 2), (2, 1, 0), (-1, -2, -3)]],
                      [(8, 3, 4, 8), [(0, 2, 3, 1), (1, 2, 3, 0), (0, 3, 2, 1)]],
                      [(8, 3, 2, 3, 8), [(0, 1, 3, 2, 4), (0, 1, 2, 3, 4), (4, 0, 1, 2, 3)]],
                      [(3, 4, 3, 4, 3, 2), [(0, 1, 3, 2, 4, 5), (2, 3, 4, 1, 0, 5), None]]]

    for hybridize in [True, False]:
        for dtype in [_np.float32, _np.float16, _np.int32]:
            for data_shape, axes_workload in test_workloads:
                for axes in axes_workload:
                    for grad_req in ['write', 'add']:
                        test_trans = TestTranspose(axes)
                        if hybridize:
                            test_trans.hybridize()
                        x = np.random.normal(0, 1, data_shape).astype(dtype)
                        x = x.astype(dtype)
                        x.attach_grad(grad_req=grad_req)
                        if grad_req == 'add':
                            x.grad[()] = np.random.normal(0, 1, x.grad.shape).astype(x.grad.dtype)
                            x_grad_np = x.grad.asnumpy()
                        np_out = _np.transpose(x.asnumpy(), axes)
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
    # Test for error raising
    dat = np.random.normal(0, 1, (3, 4, 5), dtype=np.float32)
    assert_raises(MXNetError, lambda: dat.transpose((0, 0, 1)))
    assert_raises(MXNetError, lambda: dat.transpose((0, 1, 3)))



@with_seed()
@use_np
def test_np_meshgrid():
    nx, ny = (4, 5)
    x = np.array(_np.linspace(0, 1, nx), dtype=np.float32)
    y = np.array(_np.linspace(0, 1, ny), dtype=np.float32)
    z = np.ones(())
    xv, yv, zv = np.meshgrid(x, y, z)
    xv_expected, yv_expected, zv_expected = _np.meshgrid(x.asnumpy(), y.asnumpy(), z.asnumpy())
    assert same(xv.asnumpy(), xv_expected)
    assert same(yv.asnumpy(), yv_expected)
    assert same(zv.asnumpy(), zv_expected)


@with_seed()
@use_np
def test_np_broadcast_arrays():
    shape_config = [
        [(), (2, 1), (1, 3), (4, 1, 1), (5, 4, 2, 3)],
        [(0,), (), (2, 1), (1, 0), (3, 2, 1)]
    ]
    for shapes in shape_config:
        arrays_np = [_np.random.randint(low=0, high=1000, size=shape, dtype=_np.int32) for shape in shapes]
        arrays_mx = [np.array(arr, dtype=arr.dtype) for arr in arrays_np]
        expected_rets = _np.broadcast_arrays(*arrays_np)
        rets = np.broadcast_arrays(*arrays_mx)
        for expected_ret, ret in zip(expected_rets, rets):
            assert same(expected_ret, ret.asnumpy())


@with_seed()
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

        def hybrid_forward(self, F, x):
            return F.np.tile(x, reps=self._reps)

    for shape, reps in config:
        data_np = _np.random.randint(low=0, high=1000, size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = _np.tile(data_np, reps=reps)
        ret_mx = np.tile(data_mx, reps=reps)
        assert same(ret_mx.asnumpy(), ret_np)

        net = TestTile(reps)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            ret_mx = net(data_mx)
            assert same(ret_mx.asnumpy(), ret_np)


@with_seed()
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

        def hybrid_forward(self, F, x):
            return F.np.tril(x, k=self._k)

    for prefix in [1, -1]:
        for shape, k in config:
            data_np = _np.random.uniform(size=shape)
            data_mx = np.array(data_np, dtype=data_np.dtype)
            data_mx.attach_grad()
            ret_np = _np.tril(data_np, k*prefix)
            with mx.autograd.record():
                ret_mx = np.tril(data_mx, k*prefix)
            assert same(ret_mx.asnumpy(), ret_np)
            ret_mx.backward()
            if len(shape) == 2:
                grad_np = _np.tri(*shape, k=k*prefix)
                assert same(data_mx.grad.asnumpy(), grad_np)
            if len(shape) == 1:
                grad_np = _np.tri(*shape, k=k*prefix)
                grad_np = grad_np.sum(axis=0, keepdims=False)
                assert same(data_mx.grad.asnumpy(), grad_np)

            net = TestTril(k*prefix)
            for hybrid in [False, True]:
                if hybrid:
                    net.hybridize()
                ret_mx = net(data_mx)
                assert same(ret_mx.asnumpy(), ret_np)


@with_seed()
@use_np
def test_np_unary_funcs():
    def check_unary_func(func, ref_grad, shape, low, high):
        class TestUnary(HybridBlock):
            def __init__(self, func):
                super(TestUnary, self).__init__()
                self._func = func

            def hybrid_forward(self, F, a, *args, **kwargs):
                return getattr(F.np, self._func)(a)

        np_func = getattr(_np, func)
        mx_func = TestUnary(func)
        np_test_data = _np.random.uniform(low, high, shape).astype(_np.float32)
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

        np_out = getattr(_np, func)(np_test_data)
        mx_out = getattr(mx.np, func)(mx_test_data)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


        assertRaises(NotImplementedError, getattr(np, func), mx_test_data, where=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  subok=False)
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  dtype=_np.int8)
        assertRaises(TypeError, getattr(np, func), mx_test_data,  dtype="abcdefg")
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  casting='safe')
        assertRaises(TypeError, getattr(np, func), mx_test_data,  casting='mxnet')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  order='C')
        assertRaises(NotImplementedError, getattr(np, func), mx_test_data,  order='mxnet')

    funcs = {
        'absolute' : (lambda x: -1. * (x < 0) + (x > 0), -1.0, 1.0),
        'cbrt' : (lambda x: 1. / (3. * _np.cbrt(x) ** 2), -1.0, 1.0),
        'ceil' : (None, -10.0, 10.0),
        'exp' : (lambda x: _np.exp(x), -1.0, 1.0),
        'expm1' : (lambda x: _np.exp(x), -1.0, 1.0),
        'fix' : (None, -10.0, 10.0),
        'floor' : (None, -10.0, 10.0),
        'log' : (lambda x: 1.0 / x, 0.1, 5.0),
        'log10' : (lambda x: 1.0 / (x * _np.log(10)), 0.1, 10.0),
        'log1p' : (lambda x: 1.0 / (1.0 + x), -0.9, 5.0),
        'log2' : (lambda x: 1.0 / (x * _np.log(2)), 0.1, 2.0),
        'logical_not' : (None, -1.0, 1.0),
        'negative' : (lambda x: -1. * _np.ones(x.shape), -1.0, 1.0),
        'reciprocal' : (lambda x: -1. / (x ** 2), 0.01, 1.0),
        'rint' : (None, -5.0, 5.0),
        'sign' : (None, -1.0, 1.0),
        'sqrt' : (lambda x: 0.5 / _np.sqrt(x), 0.001, 10.0),
        'square' : (lambda x: 2.0 * x, -1.0, 1.0),
        'trunc' : (None, -5.0, 5.0),
        'sin' : (lambda x: _np.cos(x), -1.0, 1.0),
        'cos' : (lambda x: -_np.sin(x), -1.0, 1.0),
        'tan' : (lambda x: _np.tan(x) ** 2 + 1.0, -1.0, 1.0),
        'arcsin' : (lambda x: 1. / (1. - x ** 2) ** (1. / 2.), -1.0, 1.0),
        'arccos' : (lambda x: -1. / (1. - x ** 2.) ** (1. / 2.), -1.0, 1.0),
        'arctan' : (lambda x: 1. / (x ** 2. + 1.), -1.0, 1.0),
        'degrees' : (lambda x: 180. / _np.pi * _np.ones(x.shape), -1.0, 1.0),
        'radians' : (lambda x: _np.pi / 180. * _np.ones(x.shape), -1.0, 1.0),
        'sinh' : (lambda x: _np.cosh(x), -1.0, 1.0),
        'cosh' : (lambda x: _np.sinh(x), -1.0, 1.0),
        'tanh' : (lambda x: 1. - _np.tanh(x) ** 2, -1.0, 1.0),
        'arcsinh' : (lambda x: 1./(x**2 + 1.)**(1./2.), -1.0, 1.0),
        'arccosh' : (lambda x: 1./(x**2 - 1.)**(1./2.), 2.0, 5.0),
        'arctanh' : (lambda x: -1./(x**2 - 1.), -0.99, 0.99)
    }
    if has_tvm_ops():
        funcs['rad2deg'] = (lambda x: 180. / _np.pi * _np.ones(x.shape), -1.0, 1.0)
        funcs['deg2rad'] = (lambda x: _np.pi / 180. * _np.ones(x.shape), -1.0, 1.0)
    ndim = random.choice([2, 3, 4])
    shape = random.choice([rand_shape_nd(ndim, dim=3), (1, 0, 2)])
    for shape in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]:
        for func, func_data in funcs.items():
            ref_grad, low, high = func_data
            check_unary_func(func, ref_grad, shape, low, high)


@with_seed()
@use_np
def test_np_binary_funcs():
    def check_binary_func(func, lshape, rshape, low, high, lgrads, rgrads=None, alltypes=None):
        class TestBinary(HybridBlock):
            def __init__(self, func):
                super(TestBinary, self).__init__()
                self._func = func

            def hybrid_forward(self, F, a, b, *args, **kwargs):
                return getattr(F.np, self._func)(a, b)

        np_func = getattr(_np, func)
        mx_func = TestBinary(func)
        alltypes = alltypes if alltypes else [[_np.float16, _np.float32, _np.float64]]
        for dtypes, lgrad, rgrad in zip(alltypes, lgrads, rgrads if rgrads else lgrads):
            for dtype in dtypes:
                ldtype = rdtype = dtype
                if isinstance(dtype, tuple):
                    assert len(dtype) == 2
                    ldtype, rdtype = dtype
                np_test_x1 = _np.random.uniform(low, high, lshape).astype(ldtype)
                np_test_x2 = _np.random.uniform(low, high, rshape).astype(rdtype)
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

                np_out = getattr(_np, func)(np_test_x1, np_test_x2)
                mx_out = getattr(mx.np, func)(mx_test_x1, mx_test_x2)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out.astype(mx_out.dtype), rtol=1e-3, atol=1e-5,
                                    use_broadcast=False, equal_nan=True)

                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2, where=False)
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  subok=False)
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  dtype=_np.int8)
                assertRaises(TypeError, getattr(np, func), mx_test_x1, mx_test_x2,  dtype="abcdefg")
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  casting='safe')
                assertRaises(TypeError, getattr(np, func), mx_test_x1, mx_test_x2,  casting='mxnet')
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  order='C')
                assertRaises(NotImplementedError, getattr(np, func), mx_test_x1, mx_test_x2,  order='mxnet')

    funcs = {
        'add': (-1.0, 1.0, [lambda y, x1, x2: _np.ones(y.shape)], None),
        'subtract':
        (-1.0, 1.0, [lambda y, x1, x2: _np.ones(y.shape)],
                    [lambda y, x1, x2: -_np.ones(y.shape)]),
        'multiply': (-1.0, 1.0, [lambda y, x1, x2: _np.broadcast_to(x2, y.shape)],
                                [lambda y, x1, x2: _np.broadcast_to(x1, y.shape)]),
        'divide': (0.1, 1.0, [lambda y, x1, x2: _np.ones(y.shape) / x2],
                   [lambda y, x1, x2: -x1 / (x2 * x2)]),
        'mod': (1.0, 10.0,
                [lambda y, x1, x2: _np.ones(y.shape),
                 lambda y, x1, x2: _np.zeros(y.shape)],
                [lambda y, x1, x2: -_np.floor(x1 / x2),
                 lambda y, x1, x2: _np.zeros(y.shape)],
                [[_np.float16, _np.float32, _np.float64], [_np.int32]]),
        'remainder': (1.0, 10.0,
                      [lambda y, x1, x2: _np.ones(y.shape),
                       lambda y, x1, x2: _np.zeros(y.shape)],
                      [lambda y, x1, x2: -_np.floor(x1 / x2),
                       lambda y, x1, x2: _np.zeros(y.shape)],
                      [[_np.float16, _np.float32, _np.float64], [_np.int32]]),
        'power': (1.0, 2.0, [lambda y, x1, x2: _np.power(x1, x2 - 1.0) * x2],
                             [lambda y, x1, x2: _np.power(x1, x2) * _np.log(x1)]),
        'lcm': (-100, 100, [None], None, [[_np.int32]]),
        'bitwise_xor': (-100, 100, [None], None, [[_np.int32]]),
        'maximum': (-1, 1, [lambda y, x1, x2: _np.ones(y.shape) * (x1 >= x2)],
                           [lambda y, x1, x2: _np.ones(y.shape) * (x1 < x2)]),
        'minimum': (-1, 1, [lambda y, x1, x2: _np.ones(y.shape) * (x1 <= x2)],
                           [lambda y, x1, x2: _np.ones(y.shape) * (x1 > x2)]),
        'copysign': (-1, 1,
                     [lambda y, x1, x2: _np.ones(y.shape) * (((x1 * x2) >= 0).astype(_np.float32) - ((x1 * x2) < 0).astype(_np.float32))],
                     [lambda y, x1, x2: _np.zeros(y.shape)]),
        'arctan2': (-1, 1, [lambda y, x1, x2: x2 / (_np.square(x1) + _np.square(x2))],
                           [lambda y, x1, x2: -x1 / (_np.square(x1) + _np.square(x2))]),
        'hypot': (-1, 1, [lambda y, x1, x2: x1 / y],
                         [lambda y, x1, x2: x2 / y]),
        'ldexp': (-3, 3, [None], None, [[_np.int32]]),
    }
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


@with_seed()
@use_np
def test_np_mixed_precision_binary_funcs():
    itypes = [np.bool, np.int8, np.int32, np.int64]
    ftypes = [np.float16, np.float32, np.float64]
    def check_mixed_precision_binary_func(func, low, high, lshape, rshape, lgrad, rgrad, ltype, rtype):
        class TestMixedBinary(HybridBlock):
            def __init__(self, func):
                super(TestMixedBinary, self).__init__()
                self._func = func

            def hybrid_forward(self, F, a, b, *args, **kwargs):
                return getattr(F.np, self._func)(a, b)

        np_func = getattr(_np, func)
        mx_func = TestMixedBinary(func)
        np_test_x1 = _np.random.uniform(low, high, lshape).astype(ltype)
        np_test_x2 = _np.random.uniform(low, high, rshape).astype(rtype)
        mx_test_x1 = mx.numpy.array(np_test_x1, dtype=ltype)
        mx_test_x2 = mx.numpy.array(np_test_x2, dtype=rtype)
        rtol = 1e-2 if ltype is np.float16 or rtype is np.float16 else 1e-3
        atol = 1e-3 if ltype is np.float16 or rtype is np.float16 else 1e-5
        for hybridize in [True, False]:
            if hybridize:
                mx_func.hybridize()
            np_out = np_func(np_test_x1, np_test_x2)
            with mx.autograd.record():
                y = mx_func(mx_test_x1, mx_test_x2)
            assert y.shape == np_out.shape
            assert_almost_equal(y.asnumpy(), np_out.astype(y.dtype), rtol=rtol, atol=atol,
                                use_broadcast=False, equal_nan=True)

        np_out = getattr(_np, func)(np_test_x1, np_test_x2)
        mx_out = getattr(mx.np, func)(mx_test_x1, mx_test_x2)
        assert mx_out.shape == np_out.shape
        assert_almost_equal(mx_out.asnumpy(), np_out.astype(mx_out.dtype), rtol=rtol, atol=atol,
                            use_broadcast=False, equal_nan=True)

    funcs = {
        'add': (-1.0, 1.0, None, None),
        'subtract': (-1.0, 1.0, None, None),
        'multiply': (-1.0, 1.0, lambda y, x1, x2: _np.broadcast_to(x2, y.shape),
                                lambda y, x1, x2: _np.broadcast_to(x1, y.shape))
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


@with_seed()
@use_np
def test_np_boolean_binary_funcs():
    def check_boolean_binary_func(func, mx_x1, mx_x2):
        class TestBooleanBinary(HybridBlock):
            def __init__(self, func):
                super(TestBooleanBinary, self).__init__()
                self._func = func

            def hybrid_forward(self, F, a, b, *args, **kwargs):
                return getattr(F.np, self._func)(a, b)

        np_x1 = mx_x1.asnumpy()
        np_x2 = mx_x2.asnumpy()
        np_func = getattr(_np, func)
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

        np_out = getattr(_np, func)(np_x1, np_x2)
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
            x1 = np.array(_np.random.uniform(size=lshape) > 0.5)
            x2 = np.array(_np.random.uniform(size=rshape) > 0.5)
            check_boolean_binary_func(func, x1, x2)


@with_seed()
@use_np
def test_npx_relu():
    def np_relu(x):
        return _np.maximum(x, 0.0)
    def np_relu_grad(x):
        return 1.0 * (x > 0.0)

    class TestReLU(HybridBlock):
        def __init__(self):
            super(TestReLU, self).__init__()

        def hybrid_forward(self, F, a):
            return F.npx.relu(a)

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


@with_seed()
@use_np
def test_npx_sigmoid():
    def np_sigmoid(x):
        return _np.divide(1.0, (1.0 + _np.exp(-x)))
    def np_sigmoid_grad(ya):
        return ya * (1 - ya)

    class TestSigmoid(HybridBlock):
        def __init__(self):
            super(TestSigmoid, self).__init__()

        def hybrid_forward(self, F, a):
            return F.npx.sigmoid(a)

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


@with_seed()
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
                np_ret = _np.arange(*config, dtype=dtype)
            else:
                mx_ret = np.arange(config, dtype=dtype)
                np_ret = _np.arange(config, dtype=dtype)
            assert same(mx_ret.asnumpy(), np_ret)

    class TestRange(HybridBlock):
        def __init__(self, start, stop=None, step=None, dtype=None):
            super(TestRange, self).__init__()
            self._start = start
            self._stop = stop
            self._step = step
            self._dtype = dtype

        def hybrid_forward(self, F, x):
            return x + F.np.arange(self._start, self._stop, self._step, dtype=self._dtype)

    for dtype in dtypes:
        x = np.zeros(shape=(), dtype=dtype)
        for config in configs:
            for hybridize in [False, True]:
                if isinstance(config, tuple):
                    net = TestRange(*config, dtype=dtype)
                    np_out = _np.arange(*config, dtype=dtype)
                else:
                    net = TestRange(config, dtype=dtype)
                    np_out = _np.arange(config, dtype=dtype)
                if hybridize:
                    net.hybridize()
                mx_out = net(x)
                assert same(mx_out.asnumpy(), np_out)


@with_seed()
@use_np
def test_np_split():
    class TestSplit(HybridBlock):
        def __init__(self, indices_or_sections, axis=None):
            super(TestSplit, self).__init__()
            self._axis = axis
            self._indices_or_sections = indices_or_sections

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.split(a, indices_or_sections=self._indices_or_sections,
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
        for axis in range(len(shape)):
            indices = get_indices(shape[axis])
            sections = 7 if shape[axis] is 0 else shape[axis]
            for indices_or_sections in [indices, sections]:
                # test gluon
                test_split = TestSplit(axis=axis, indices_or_sections=indices_or_sections)
                if hybridize:
                    test_split.hybridize()

                a = mx.nd.random.uniform(-1.0, 1.0, shape=shape).as_np_ndarray()
                a.attach_grad()
                expected_ret = _np.split(a.asnumpy(), indices_or_sections=indices_or_sections, axis=axis)
                with mx.autograd.record():
                    y = test_split(a)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                mx.autograd.backward(y)

                assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_outs = np.split(a, indices_or_sections=indices_or_sections, axis=axis)
                np_outs = _np.split(a.asnumpy(), indices_or_sections=indices_or_sections, axis=axis)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_vsplit():
    class TestVsplit(HybridBlock):
        def __init__(self, indices_or_sections):
            super(TestVsplit, self).__init__()
            self._indices_or_sections = indices_or_sections

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.vsplit(a, indices_or_sections=self._indices_or_sections)

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
                expected_ret = _np.vsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                with mx.autograd.record():
                    y = test_vsplit(a)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                mx.autograd.backward(y)

                assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_outs = np.vsplit(a, indices_or_sections=indices_or_sections)
                np_outs = _np.vsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_concat():
    class TestConcat(HybridBlock):
        def __init__(self, axis=None):
            super(TestConcat, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, a, *args):
            return F.np.concatenate([a] + list(args), axis=self._axis)

    def get_new_shape(shape, axis):
        shape_lst = list(shape)
        if axis is not None:
            shape_lst[axis] = random.randint(0, 3)
        return tuple(shape_lst)

    for shape in [(0, 0), (2, 3), (2, 1, 3)]:
        for hybridize in [True, False]:
            for axis in [0, 1, None]:
                for grad_req in ['write', 'add', 'null']:
                    # test gluon
                    test_concat = TestConcat(axis=axis)
                    if hybridize:
                        test_concat.hybridize()

                    grad_req_c = grad_req
                    grad_req_d = grad_req
                    if grad_req == 'null':
                        ide = random.randint(0, 2)
                        grad_req_c = 'write' if ide == 0 else 'add'
                        grad_req_c = 'write' if ide == 1 else 'add'

                    a = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                    a.attach_grad(grad_req)
                    b = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                    b.attach_grad(grad_req)
                    c = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                    c.attach_grad(grad_req_c)
                    d = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                    d.attach_grad(grad_req_d)
                    expected_ret = _np.concatenate([a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()], axis=axis)

                    with mx.autograd.record():
                        y = test_concat(a, b, c, d)

                    assert y.shape == expected_ret.shape
                    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)

                    y.backward()
                    if grad_req != 'null':
                        assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)
                    if grad_req != 'null':
                        assert_almost_equal(b.grad.asnumpy(), _np.ones(b.shape), rtol=1e-3, atol=1e-5)
                    if grad_req_c != 'null':
                        assert_almost_equal(c.grad.asnumpy(), _np.ones(c.shape), rtol=1e-3, atol=1e-5)
                    if grad_req_d != 'null':
                        assert_almost_equal(d.grad.asnumpy(), _np.ones(d.shape), rtol=1e-3, atol=1e-5)

                    # test imperative
                    mx_out = np.concatenate([a, b, c, d], axis=axis)
                    np_out = _np.concatenate([a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()], axis=axis)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_append():
    class TestAppend(HybridBlock):
        def __init__(self, axis=None):
            super(TestAppend, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, a, b):
            return F.np.append(a, b, axis=self._axis)

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
                    expected_ret = _np.append(a.asnumpy(), b.asnumpy(), axis=axis)

                    with mx.autograd.record():
                        y = test_append(a, b)

                    assert y.shape == expected_ret.shape
                    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)
                    y.backward()

                    if grad_req_a != 'null':
                        assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)
                    assert_almost_equal(b.grad.asnumpy(), _np.ones(b.shape), rtol=1e-3, atol=1e-5)
                    #test imperative
                    mx_out = np.append(a, b, axis=axis)
                    np_out = _np.append(a.asnumpy(), b.asnumpy(), axis=axis)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_stack():
    class TestStack(HybridBlock):
        def __init__(self, axis=None):
            super(TestStack, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, a, *args):
            return F.np.stack([a] + list(args), axis=self._axis)

    a, b, c, d = mx.sym.Variable("a"), mx.sym.Variable("b"), mx.sym.Variable("c"), mx.sym.Variable("d")
    ret = mx.sym.np.stack([a.as_np_ndarray(), b.as_np_ndarray(), c.as_np_ndarray(), d.as_np_ndarray()])
    assert type(ret) == mx.sym.np._Symbol

    for shape in [(0, 0), (2, 3)]:
        for hybridize in [True, False]:
            for axis in range(2):
                test_stack = TestStack(axis=axis)
                if hybridize:
                    test_stack.hybridize()
                np_a = _np.random.uniform(-1.0, 1.0, shape).astype(_np.float32)
                np_b = _np.random.uniform(-1.0, 1.0, shape).astype(_np.float32)
                np_c = _np.random.uniform(-1.0, 1.0, shape).astype(_np.float32)
                np_d = _np.random.uniform(-1.0, 1.0, shape).astype(_np.float32)

                mx_a = np.array(np_a)
                mx_a.attach_grad()
                mx_b = np.array(np_b)
                mx_b.attach_grad()
                mx_c = np.array(np_c)
                mx_c.attach_grad()
                mx_d = np.array(np_d)
                mx_d.attach_grad()
                expected_ret = _np.stack([np_a, np_b, np_c, np_d], axis=axis)
                with mx.autograd.record():
                    y = test_stack(mx_a, mx_b, mx_c, mx_d)

                y.backward()

                assert_almost_equal(mx_a.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_b.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_c.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_d.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)

                np_out = _np.stack([np_a, np_b, np_c, np_d], axis=axis)
                mx_out = np.stack([mx_a, mx_b, mx_c, mx_d], axis=axis)
                assert same(mx_out.asnumpy(), np_out)


@with_seed()
@use_np
def test_np_dstack():
    class TestDStack(HybridBlock):
        def __init__(self):
            super(TestDStack, self).__init__()

        def hybrid_forward(self, F, a, *args):
            return F.np.dstack([a] + list(args))

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
            np_out = _np.dstack((a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

            # test symbolic backward
            mx_out.backward()
            assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(b.grad.asnumpy(), _np.ones(b.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(c.grad.asnumpy(), _np.ones(c.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(d.grad.asnumpy(), _np.ones(d.shape), rtol=1e-3, atol=1e-5)

            # test imperative
            mx_out = np.dstack((a, b, c, d))
            np_out = _np.dstack((a.asnumpy(),b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_ravel():
    class TestRavel(HybridBlock):
        def __init__(self):
            super(TestRavel, self).__init__()

        def hybrid_forward(self, F, a):
            return F.np.ravel(a)

    types = ['float64', 'float32', 'float16', 'int64', 'int32', 'int8']
    for oneType in types:
        for hybridize in [True, False]:
            for shape in [(), (2,), (2, 2), (1, 2, 3), (3, 0), (1, 0, 2)]:
                test_ravel = TestRavel()
                if hybridize:
                    test_ravel.hybridize()
                x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                x.attach_grad()
                np_out = _np.ravel(x.asnumpy())
                with mx.autograd.record():
                    mx_out = test_ravel(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                mx_out.backward()
                np_backward = _np.ones(shape)
                assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

                mx_out = np.ravel(x)
                np_out = _np.ravel(x.asnumpy())
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_randint():
    ctx = mx.context.current_context()
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
            buckets = _np.array(buckets, dtype=dtype).tolist()
            probs = [(buckets[i][1] - buckets[i][0]) / float(scale) for i in range(5)]
            generator_mx = lambda x: np.random.randint(low, high, size=x, dtype=dtype, ctx=ctx).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs, nrepeat=100)
            # Scipy uses alpha = 0.01 for testing discrete distribution generator but we are using default alpha=0.05 (higher threshold ensures robustness)
            # Refer - https://github.com/scipy/scipy/blob/9f12af697763fb5f9767d5cb1280ce62456a3974/scipy/stats/tests/test_discrete_basic.py#L45
            generator_mx_same_seed = \
                lambda x: _np.concatenate(
                    [np.random.randint(low, high, size=x // 10, dtype=dtype, ctx=ctx).asnumpy()
                        for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs, nrepeat=100)


@with_seed()
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

        def hybrid_forward(self, F, x):
            return F.np.swapaxes(x, self._axis1, self._axis2)

    for shape, axis1, axis2 in config:
        data_np = _np.random.uniform(size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = _np.swapaxes(data_np, axis1=axis1, axis2=axis2)
        ret_mx = np.swapaxes(data_mx, axis1=axis1, axis2=axis2)
        assert same(ret_mx.asnumpy(), ret_np)

        net = TestSwapaxes(axis1, axis2)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            ret_mx = net(data_mx)
            assert same(ret_mx.asnumpy(), ret_np)


@with_seed()
@use_np
def test_np_argmin_argmax():
    workloads = [
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
    ]
    dtypes = ['float16', 'float32', 'float64']
    ops = ['argmin', 'argmax']

    class TestArgExtreme(HybridBlock):
        def __init__(self, op_name, axis=None):
            super(TestArgExtreme, self).__init__()
            self._op_name = op_name
            self._axis = axis

        def hybrid_forward(self, F, x):
            return getattr(x, self._op_name)(self._axis)

    for op_name in ops:
        for shape, axis, throw_exception in workloads:
            for dtype in dtypes:
                a = np.random.uniform(size=shape, dtype=dtype)
                if throw_exception:
                    # Cannot use assert_exception because sometimes the main thread
                    # proceeds to `assert False` before the exception is thrown
                    # in the worker thread. Have to use mx.nd.waitall() here
                    # to block the main thread.
                    try:
                        getattr(np, op_name)(a, axis)
                        mx.nd.waitall()
                        assert False
                    except mx.MXNetError:
                        pass
                else:
                    mx_ret = getattr(np, op_name)(a, axis=axis)
                    np_ret = getattr(_np, op_name)(a.asnumpy(), axis=axis)
                    assert same(mx_ret.asnumpy(), np_ret)

                for hybridize in [False, True]:
                    net = TestArgExtreme(op_name, axis)
                    if hybridize:
                        net.hybridize()
                    if throw_exception:
                        try:
                            net(a)
                            mx.nd.waitall()
                            assert False
                        except mx.MXNetError:
                            pass
                    else:
                        mx_ret = net(a)
                        assert same(mx_ret.asnumpy(), np_ret)


@with_seed()
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

        def hybrid_forward(self, F, x):
            return x.clip(self._a_min, self._a_max)

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


@with_seed()
@use_np
def test_np_random():
    shapes = [(), (1,), (2, 3), (4, 0, 5), 6, (7, 8), None]
    dtypes = ['float16', 'float32', 'float64']
    op_names = ['uniform', 'normal']
    for shape in shapes:
        for dtype in dtypes:
            for op_name in op_names:
                op = getattr(np.random, op_name, None)
                assert op is not None
                out = op(size=shape, dtype=dtype)
                expected_shape = shape
                if not isinstance(shape, tuple):
                    expected_shape = () if shape is None else (shape,)
                assert out.shape == expected_shape

    class TestRandom(HybridBlock):
        def __init__(self, shape, op_name):
            super(TestRandom, self).__init__()
            self._shape = shape
            self._op_name = op_name

        def hybrid_forward(self, F, x):
            op = getattr(F.np.random, self._op_name, None)
            assert op is not None
            return x + op(size=shape)

    x = np.ones(())
    for op_name in op_names:
        for shape in shapes:
            for hybridize in [False, True]:
                net = TestRandom(shape, op_name)
                if hybridize:
                    net.hybridize()
                out = net(x)
                expected_shape = shape
                if not isinstance(shape, tuple):
                    expected_shape = () if shape is None else (shape,)
                assert out.shape == expected_shape


@with_seed()
@use_np
def test_random_seed():
    for seed in [234, 594, 7240, 20394]:
        ret = []
        for _ in range(2):
            npx.random.seed(seed=seed)
            ret.append(np.random.uniform(size=(2, 3)))
        assert_almost_equal(ret[0].asnumpy(), ret[1].asnumpy(), rtol=1e-4, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_cumsum():
    def np_cumsum_backward(ograd, axis=None, dtype=None):
        return _np.flip(_np.cumsum(_np.flip(ograd, axis=axis), axis=axis, dtype=dtype), axis=axis)

    class TestCumsum(HybridBlock):
        def __init__(self, axis=None, dtype=None):
            super(TestCumsum, self).__init__()
            self._axis = axis
            self._dtype = dtype

        def hybrid_forward(self, F, a):
            return a.cumsum(axis=self._axis, dtype=self._dtype)

    shapes = [(2, 3, 4), (2, 0, 3), ()]
    for hybridize in [True, False]:
        for shape in shapes:
            for axis in [None] + [i for i in range(0, len(shape))]:
                for otype in [None, _np.float32, _np.float64]:
                    test_cumsum = TestCumsum(axis=axis, dtype=otype)
                    if hybridize:
                        test_cumsum.hybridize()
                    for itype in [_np.float16, _np.float32, _np.float64]:
                        x = rand_ndarray(shape).astype(itype).as_np_ndarray()
                        x.attach_grad()
                        np_out = _np.cumsum(x.asnumpy(), axis=axis, dtype=otype)
                        with mx.autograd.record():
                            mx_out = test_cumsum(x)
                        assert mx_out.shape == np_out.shape
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                        mx_out.backward()
                        np_backward = np_cumsum_backward(_np.ones(np_out.shape, dtype=otype),
                                                         axis=axis, dtype=otype).reshape(x.shape)
                        assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5)

                        mx_out = np.cumsum(x, axis=axis, dtype=otype)
                        np_out = _np.cumsum(x.asnumpy(), axis=axis, dtype=otype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    for shape in shapes:
        for axis in [None] + [i for i in range(0, len(shape))]:
            for otype in [None, _np.int32, _np.int64]:
                for itype in [_np.bool, _np.int8, _np.int32, _np.int64]:
                    x = rand_ndarray(shape).astype(itype).as_np_ndarray()
                    np_out = _np.cumsum(x.asnumpy(), axis=axis, dtype=otype)
                    mx_out = np.cumsum(x, axis=axis, dtype=otype)
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_histogram():
    shapes = [(), (3, 4), (3, 0)]

    for shape in shapes:
        mx_a = np.random.uniform(0.0, 10.0, size=shape)
        np_a = mx_a.asnumpy()
        mx_bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5., 6., 7., 8., 9., 10.])
        np_bins = mx_bins.asnumpy()
        for bins, _range in [(20, (0.0, 10.0)), (mx_bins, None)]:
            mx_cnts, mx_bins = np.histogram(mx_a, bins=bins, range=_range)
            np_cnts, np_bins = _np.histogram(np_a, bins=bins if isinstance(bins, mx.base.numeric_types) else bins.asnumpy(), range=_range)
            assert_almost_equal(mx_cnts.asnumpy(), np_cnts, rtol=1e-3, atol=1e-5)
            assert_almost_equal(mx_bins.asnumpy(), np_bins, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_choice():
    class TestUniformChoice(HybridBlock):
        def __init__(self, sample_size, replace):
            super(TestUniformChoice, self).__init__()
            self.sample_size = sample_size
            self.replace = replace

        def hybrid_forward(self, F, a):
            return F.np.random.choice(a=a, size=self.sample_size, replace=self.replace, p=None)

    class TestWeightedChoice(HybridBlock):
        def __init__(self, sample_size, replace):
            super(TestWeightedChoice, self).__init__()
            self.sample_size = sample_size
            self.replace = replace

        def hybrid_forward(self, F, a, p):
            op = getattr(F.np.random, "choice", None)
            return F.np.random.choice(a, self.sample_size, self.replace, p)

    def test_sample_with_replacement(sampler, num_classes, shape, weight=None):
        samples = sampler(num_classes, shape, replace=True, p=weight).asnumpy()
        generated_density = _np.histogram(samples, _np.arange(num_classes + 1), density=True)[0]
        expected_density = (weight.asnumpy() if weight is not None else
                            _np.array([1 / num_classes] * num_classes))
        # test almost equal
        assert_almost_equal(generated_density, expected_density, rtol=1e-1, atol=1e-1)
        # test shape
        assert (samples.shape == shape)

    def test_sample_without_replacement(sampler, num_classes, shape, num_trials, weight=None):
        samples = sampler(num_classes, shape, replace=False, p=weight).asnumpy()
        # Check shape and uniqueness
        assert samples.shape == shape
        assert len(_np.unique(samples)) == samples.size
        # Check distribution
        bins = _np.zeros((num_classes))
        expected_freq = (weight.asnumpy() if weight is not None else
                         _np.array([1 / num_classes] * num_classes))
        for i in range(num_trials):
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
            assert len(_np.unique(samples.asnumpy())) == samples_size

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
    #     weight = np.array(_np.random.dirichlet([1.0] * num_classes))
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
    #     weight = np.array(_np.random.dirichlet([1.0] * num_classes))
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
                weight = np.array(_np.random.dirichlet([1.0] * num_classes)).astype(wtype)
                test_indexing_mode(test_choice, num_classes, num_classes // 2, replace, None)
                test_indexing_mode(test_choice_weighted, num_classes, num_classes // 2, replace, weight)


@with_seed()
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
                np_ret = _np.eye(*config, dtype=dtype)
            else:
                mx_ret = np.eye(config, dtype=dtype)
                np_ret = _np.eye(config, dtype=dtype)
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

        def hybrid_forward(self, F, x):
            return x + F.np.eye(self._N, self._M, self._k, dtype=self._dtype)

    for dtype in dtypes:
        x = np.zeros(shape=(), dtype=dtype)
        for config in configs:
            for hybridize in [False, True]:
                if isinstance(config, tuple):
                    net = TestEye(*config, dtype=dtype)
                    np_out = _np.eye(*config, dtype=dtype)
                else:
                    net = TestEye(config, dtype=dtype)
                    np_out = _np.eye(config, dtype=dtype)
                if hybridize:
                    net.hybridize()
                mx_out = net(x)
                assert same(mx_out.asnumpy(), np_out)


@with_seed()
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
            np_out = _np.indices(dimensions=shape, dtype=dtype)
            mx_out = np.indices(dimensions=shape, dtype=dtype)
            same(mx_out.asnumpy(), np_out)
            assert mx_out.shape == np_out.shape

    @use_np
    class TestIndices(HybridBlock):
        def __init__(self, dimensions=None, dtype=None):
            super(TestIndices, self).__init__()
            self._dimensions = dimensions
            self._dtype = dtype

        def hybrid_forward(self, F, x):
            return x + F.np.indices(dimensions=self._dimensions, dtype=self._dtype)

    for dtype in dtypes:
        for shape in shapes:
            x = np.zeros(shape=(), dtype=dtype)
            for hybridize in [False, True]:
                net = TestIndices(dimensions=shape, dtype=dtype)
                np_out = _np.indices(dimensions=shape, dtype=dtype)
                if hybridize:
                    net.hybridize()
                mx_out = net(x)
                same(mx_out.asnumpy(), np_out)
                assert mx_out.shape == np_out.shape


@with_seed()
@use_np
def test_np_repeat():
    config = [
        ((), 2, None),
        ((), 0, None),
        ((4, 2), 2, None),
        ((4, 2), 2, 0),
        ((4, 2), 2, 1),
        ((4, 2), 2, -1),
    ]

    class TestRepeat(HybridBlock):
        def __init__(self, repeats, axis=None):
            super(TestRepeat, self).__init__()
            self._repeats = repeats
            self._axis = axis

        def hybrid_forward(self, F, x):
            return x.repeat(self._repeats, self._axis)

    for shape, repeats, axis in config:
        data_np = _np.random.randint(low=0, high=1000, size=shape)
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


@with_seed()
@use_np
def test_np_linalg_norm():
    @use_np
    class TestLinalgNorm(HybridBlock):
        def __init__(self, ord=None, axis=None, keepdims=False):
            super(TestLinalgNorm, self).__init__()
            self._ord = ord
            self._axis = axis
            self._keepdims = keepdims

        def hybrid_forward(self, F, x):
            return F.np.linalg.norm(x, ord=self._ord, axis=self._axis, keepdims=self._keepdims)

    a = np.arange(5 * 6 * 7 * 8).reshape((5, 6, 7, 8))
    ords = [None, 'fro']
    axes = [None, (0, 2), (1, 0), (1, 2)]
    for ord in ords:
        for axis in axes:
            if ord == 'fro' and axis is None and a.ndim > 2:
                continue
            for keepdims in [False, True]:
                for hybridize in [False, True]:
                    net = TestLinalgNorm(ord, axis, keepdims)
                    if hybridize:
                        net.hybridize()
                    mx_ret = net(a)
                    np_ret = _np.linalg.norm(a.asnumpy(), ord=ord, axis=axis, keepdims=keepdims)
                    assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-5, rtol=1e-4)


@with_seed()
@use_np
def test_np_linalg_svd():
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

    def check_svd(UT, L, V, data_np):
        shape = data_np.shape
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
                rtol = atol = 0.01
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
                # check svd validity
                check_svd(UT, L, V, data_np)
                # check descending singular values
                s = [L[..., i] - L[..., i + 1] for i in range(L.shape[-1] - 1)]
                s = _np.array(s)
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


@with_seed()
@use_np
def test_np_vstack():
    class TestVstack(HybridBlock):
        def __init__(self):
            super(TestVstack, self).__init__()

        def hybrid_forward(self, F, a, *args):
            return F.np.vstack([a] + list(args))

    def g(data):
        return _np.ones_like(data)

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
                    v_np.append(_np.array(_np.random.uniform(-10.0, 10.0, config[i]), dtype=dtype))
                    v.append(mx.nd.array(v_np[i]).as_np_ndarray())
                    v[i].attach_grad()
                expected_np = _np.vstack(v_np)
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
                expected_np = _np.vstack(v_np)
                assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


@with_seed()
@use_np
def test_np_roll():
    class TestRoll(HybridBlock):
        def __init__(self, shift=None, axis=None):
            super(TestRoll, self).__init__()
            self._shift = shift
            self._axis = axis

        def hybrid_forward(self, F, x):
            return F.np.roll(x, shift=self._shift, axis=self._axis)

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
    i_dtype = {"float32" : _np.float32,
               "float64" : _np.float64
               }
    for dtype in dtypes:
        for config in configs:
            for hybridize in [False, True]:
                shape, shift, axis = config[0], config[1], config[2]
                x = rand_ndarray(shape=shape, dtype=dtype).as_np_ndarray()
                net = TestRoll(shift=shift, axis=axis)
                np_out = _np.roll(x.asnumpy(), shift=shift, axis=axis)
                if hybridize:
                    net.hybridize()
                x.attach_grad()
                with mx.autograd.record():
                    mx_out = net(x)
                assert mx_out.shape == np_out.shape
                mx_out.backward()
                assert same(mx_out.asnumpy(), np_out)
                assert same(x.grad.shape, x.shape)
                assert same(x.grad.asnumpy(), _np.ones(shape))

                # test imperativen
                np_out = _np.roll(x.asnumpy(), shift=shift, axis=axis)
                mx_out = np.roll(x, shift=shift, axis=axis)
                assert same(mx_out.asnumpy(), np_out)

                # test numeric
                if dtype in ['float32', 'float64'] and len(shape)> 0 and  _np.prod(shape) > 0:
                    x_sym = mx.sym.Variable("x").as_np_ndarray()
                    mx_sym = mx.sym.np.roll(x_sym, shift=shift, axis=axis).as_nd_ndarray()
                    check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                           numeric_eps=1e-3, rtol=1e-3, atol=1e-5, dtype=i_dtype[dtype])


@with_seed()
@use_np
def test_np_trace():
    class TestTrace(HybridBlock):
        def __init__(self, axis1, axis2, offset):
            super(TestTrace, self).__init__()
            self._axis1 = axis1
            self._axis2 = axis2
            self._offset = offset

        def hybrid_forward(self, F, data):
            return F.np.trace(data, axis1=self._axis1, axis2=self._axis2, offset=self._offset)

    def g(data, axis1, axis2, offset):
        idx = _np.indices(data.shape)
        ret = _np.zeros_like(data)
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
                                data_np = _np.random.uniform(-10.0, 10.0, shape)
                                data = mx.nd.array(data_np, dtype=dtype)
                                data_np = data.asnumpy()
                                data.attach_grad()
                                expected_np = _np.trace(data_np, axis1=axis1, axis2=axis2, offset=offset)
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
        data_np = _np.random.uniform(-1.0, 1.0, shape)
        data_mx = mx.nd.array(data_np)
        try:
            output = np.trace(data_mx.as_np_ndarray(), axis1=axis1, axis2=axis2, offset=offset)
        except mx.base.MXNetError:
            continue
        assert False


@with_seed()
@use_np
def test_np_windows():
    class TestWindows(HybridBlock):
        def __init__(self, func, M, dtype):
            super(TestWindows, self).__init__()
            self._func = func
            self._M = M
            self._dtype = dtype

        def hybrid_forward(self, F, x, *args, **kwargs):
            op = getattr(F.np, self._func)
            assert op is not None
            return x + op(M=self._M, dtype=self._dtype)

    configs = [-10, -3, -1, 0, 1, 6, 10, 20]
    dtypes = ['float32', 'float64']
    funcs = ['hanning', 'hamming', 'blackman']
    for config in configs:
        for dtype in dtypes:
            for func in funcs:
                x = np.zeros(shape=(), dtype=dtype)
                for hybridize in [False, True]:
                    np_func = getattr(_np, func)
                    mx_func = TestWindows(func, M=config, dtype=dtype)
                    np_out = np_func(M=config).astype(dtype)
                    if hybridize:
                        mx_func.hybridize()
                    mx_out = mx_func(x)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                    # test imperative
                    mx_out = getattr(np, func)(M=config, dtype=dtype)
                    np_out = np_func(M=config).astype(dtype)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_flip():
    class TestFlip(HybridBlock):
        def __init__(self, axis):
            super(TestFlip, self).__init__()
            self.axis = axis

        def hybrid_forward(self, F, x):
            return F.np.flip(x, self.axis)

    shapes = [(1, 2, 3), (1, 0), ()]
    types = ['int32', 'int64', 'float16', 'float32', 'float64']
    for hybridize in [True, False]:
        for oneType in types:
            rtol, atol=1e-3, 1e-5
            for shape in shapes:
                axis = random.randint(-1, len(shape) - 1)
                if axis is -1:
                    axis = None
                test_flip = TestFlip(axis)
                if hybridize:
                    test_flip.hybridize()
                x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                x.attach_grad()
                np_out = _np.flip(x.asnumpy(), axis)
                with mx.autograd.record():
                    mx_out = test_flip(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                mx_out.backward()
                np_backward = _np.ones(np_out.shape)
                assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

                # Test imperative once again
                mx_out = np.flip(x, axis)
                np_out = _np.flip(x.asnumpy(), axis)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@with_seed()
@use_np
def test_np_around():
    class TestAround(HybridBlock):
        def __init__(self, decimals):
            super(TestAround, self).__init__()
            self.decimals = decimals

        def hybrid_forward(self, F, x):
            return F.np.around(x, self.decimals)

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
                    np_out = _np.around(x.asnumpy(), d)
                    mx_out = test_around(x)
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)

                    mx_out = np.around(x, d)
                    np_out = _np.around(x.asnumpy(), d)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@with_seed()
@use_np
def test_np_nonzero():
    class TestNonzero(HybridBlock):
        def __init__(self):
            super(TestNonzero, self).__init__()

        def hybrid_forward(self, F, x):
            return F.npx.nonzero(x)

    types = ['int32', 'int64', 'float64', 'float32', 'float16']
    for hybridize in [True, False]:
        for shape in [(), (1, 2, 3), (1, 0)]:
            for oneType in types:
                rtol, atol = 1e-3, 1e-5
                test_nonzero = TestNonzero()
                if hybridize:
                    test_nonzero.hybridize()
                x = rand_ndarray(shape, dtype=oneType).as_np_ndarray()
                np_out = _np.nonzero(x.asnumpy())
                np_out = _np.transpose(np_out)
                mx_out = test_nonzero(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol, atol)

                # Test imperative once again
                mx_out = npx.nonzero(x)
                np_out = _np.nonzero(x.asnumpy())
                np_out = _np.transpose(np_out)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol, atol)


@with_seed()
@use_np
def test_np_unique():
    class TestUnique(HybridBlock):
        def __init__(self, return_index=False, return_inverse=False, return_counts=False, axis=None):
            super(TestUnique, self).__init__()
            self._return_index = return_index
            self._return_inverse = return_inverse
            self._return_counts = return_counts
            self._axis = axis

        def hybrid_forward(self, F, a):
            return F.np.unique(a, self._return_index, self._return_inverse, self._return_counts, self._axis)

    configs = [
        ((), True, True, True, None),
        ((1, ), True, True, True, -1),
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
        for hybridize in [False]:
            for config in configs:
                test_unique = TestUnique(*config[1:])
                if hybridize:
                    test_unique.hybridize()
                x = _np.random.uniform(-8.0, 8.0, size=config[0])
                x = np.array(x, dtype=dtype)
                np_out = _np.unique(x.asnumpy(), *config[1:])
                mx_out = test_unique(x)
                assert mx_out[0].shape == np_out[0].shape
                for i in range(4):
                    assert_almost_equal(mx_out[i].asnumpy(), np_out[i], rtol=1e-3, atol=1e-5)

                # Test imperative once again
                mx_out = np.unique(x, *config[1:])
                np_out = _np.unique(x.asnumpy(), *config[1:])
                for i in range(4):
                    assert_almost_equal(mx_out[i].asnumpy(), np_out[i], rtol=1e-3, atol=1e-5)


@with_seed()
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

        def hybrid_forward(self, F, a, indices):
            return F.np.take(a, indices, axis=self._axis, mode=self._mode)

    def grad_helper(grad_in, axis, idx, mode):
        k = grad_in.shape[axis]
        if mode == 'clip':
            idx = 0 if idx < 0 else idx
            idx = k - 1 if idx >= k else idx
        else:
            idx = idx % k
        if axis == None:
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
            raise ValueError("axis %d is not supported..." % axis)

    def check_output_n_grad(data_shape, idx_shape, axis, mode):
        data_real = _np.random.normal(size=data_shape).astype('float32')
        idx_real = _np.random.randint(low=-100, high=100, size=idx_shape)
        same(np.take(np.array(data_real), np.array(idx_real), axis=axis, mode=mode).asnumpy(),
             _np.take(data_real, idx_real, axis=axis, mode=mode))

        grad_in = _np.zeros(data_shape, dtype='float32')

        test_take = TestTake(axis=axis, mode=mode)
        if hybridize:
            test_take.hybridize()
        x = np.array(data_real)
        x.attach_grad()
        with mx.autograd.record():
            mx_out = test_take(x, np.array(idx_real))
        same(mx_out.asnumpy(), _np.take(data_real, idx_real, axis=axis, mode=mode))

        if axis and axis < 0:
            axis += len(data_shape)
        try:
            for i in _np.nditer(idx_real):
                grad_helper(grad_in, axis, i, mode)
        except:
            pass

        mx_out.backward()
        same(x.grad.asnumpy(), grad_in)

    for hybridize in [True, False]:
        for mode in ['clip', 'wrap']:
            for data_ndim in range(1, 5):
                for idx_ndim in range(1, 4):
                    for axis in range(-data_ndim, data_ndim):
                        data_shape = ()
                        for _ in range(data_ndim):
                            data_shape += (_np.random.randint(low=1, high=5), )
                        idx_shape = ()
                        for _ in range(idx_ndim):
                            idx_shape += (_np.random.randint(low=1, high=5), )
                        check_output_n_grad(data_shape, idx_shape, axis, mode)

            for config in configs:
                check_output_n_grad(config[0], config[1], config[2], mode)


@unittest.skipUnless(sys.version_info.major >= 3 and sys.version_info.minor >= 5,
                     'inspect package requires Python >= 3.5 to work properly')
@with_seed()
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


@with_seed()
@use_np
def test_np_moveaxis():
    class TestMoveaxis(HybridBlock):
        def __init__(self, source=None, destination=None):
            super(TestMoveaxis, self).__init__()
            self._source = source
            self._destination= destination

        def hybrid_forward(self, F, x):
            return F.np.moveaxis(x, source=self._source, destination=self._destination)

    dtypes = ['int32', 'int64', 'float16', 'float32', 'float64']
    for hybridize in [False, True]:
        for dtype in dtypes:
            for ndim in [0, 1, 2, 3, 4, 5, 6]:
                shape = rand_shape_nd(ndim, dim=5, allow_zero_size=True)
                np_data = _np.random.uniform(low=-100, high=100, size=shape).astype(dtype)
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
                    np_out = _np.moveaxis(np_data, source=source, destination=destination)
                    mx_data.attach_grad()
                    with mx.autograd.record():
                        mx_out = test_moveaxis(mx_data)
                    assert mx_out.shape == np_out.shape
                    mx_out.backward()
                    assert same(mx_data.grad.shape, mx_data.shape)
                    assert same(mx_data.grad.asnumpy(), _np.ones(shape))
                    # test imperative
                    np_out = _np.moveaxis(np_data, source=source, destination=destination)
                    mx_out = np.moveaxis(mx_data, source=source, destination= destination)
                    assert np_out.dtype == mx_out.dtype
                    assert same(mx_out.asnumpy(), np_out)


@with_seed()
@use_np
def test_np_rot90():
    class TestTRot90(HybridBlock):
        def __init__(self, k=1, axes=(0, 1)):
            super(TestTRot90, self).__init__()
            self._k = k
            self._axes = axes

        def hybrid_forward(self, F, a, *args):
            return F.np.rot90(a, self._k, self._axes)

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
                np_out = _np.rot90(x.asnumpy(), k=k, axes=axes)
                with mx.autograd.record():
                    mx_out = net(x)
                assert mx_out.shape == np_out.shape
                assert same(mx_out.asnumpy(), np_out)
                mx_out.backward()
                np_backward = _np.ones(shape, dtype)

                assert same(x.grad.asnumpy().shape, np_backward.shape)
                assert same(x.grad.asnumpy(), np_backward)

                np_out = _np.rot90(x.asnumpy(), k=k, axes=axes)
                mx_out = np.rot90(x, k=k, axes=axes)
                assert same(mx_out.asnumpy(), np_out)


@with_seed()
@use_np
def test_np_hsplit():
    class TestHSplit(HybridBlock):
        def __init__(self, indices_or_sections):
            super(TestHSplit, self).__init__()
            self._indices_or_sections = indices_or_sections

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.hsplit(a, indices_or_sections=self._indices_or_sections)

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
                expected_ret = _np.hsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                with mx.autograd.record():
                    y = test_hsplit(a)
                assert len(y) == len(expected_ret)
                for mx_out, np_out in zip(y, expected_ret):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)
                mx.autograd.backward(y)
                assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_outs = np.hsplit(a, indices_or_sections=indices_or_sections)
                np_outs = _np.hsplit(a.asnumpy(), indices_or_sections=indices_or_sections)
                for mx_out, np_out in zip(mx_outs, np_outs):
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@use_np
def test_np_einsum():
    class TestEinsum(HybridBlock):
        def __init__(self, subscripts, optimize):
            super(TestEinsum, self).__init__()
            self.subscripts = subscripts
            self.optimize = optimize

        def hybrid_forward(self, F, *operands):
            return F.np.einsum(self.subscripts, *operands, optimize=self.optimize)

    def dbg(name, data):
        print('type of {} = {}'.format(name, type(data)))
        print('shape of {} = {}'.format(name, data.shape))
        print('{} = {}'.format(name, data))

    configs = [
        ('ii', [(5, 5)], lambda *args: (_np.eye(5),)),
        ('ii->i', [(5, 5)], lambda *args: (_np.eye(5),)),
        ('ij->i', [(5, 5)], lambda *args: (_np.ones((5, 5)),)),
        ('...j->...', [(5, 5)], lambda *args: (_np.ones((5, 5)),)),
        ('ji', [(2, 3)], lambda *args: (_np.ones((2, 3)),)),
        ('ij->ji', [(2, 3)], lambda *args: (_np.ones((2, 3)),)),
        ('i, i', [(5,), (5,)], lambda *args: (args[1], args[0])),
        ('ij, j', [(5, 5), (5,)], lambda *args: (_np.tile(args[1][None, :], [5, 1]),
                                                 args[0].sum(axis=0))),
        ('...j, j', [(5, 5), (5,)], lambda *args: (_np.tile(args[1][None, :], [5, 1]),
                                                   _np.sum(args[0], axis=0))),
        ('..., ...', [(), (2, 3)], lambda *args: (_np.sum(args[1], axis=None),
                                                  args[0] * _np.ones((2, 3)))),
        (', ij', [(), (2, 3)], lambda *args: (_np.sum(args[1], axis=None),
                                              args[0] * _np.ones((2, 3)))),
        ('i, j', [(2,), (5, )], lambda *args: (_np.sum(args[1], axis=None) * _np.ones(2),
                                               _np.sum(args[0], axis=None) * _np.ones(5))),
        ('ijk, jil->kl', [(3, 4, 5), (4, 3, 2)], lambda *args: (_np.tile(_np.transpose(_np.sum(args[1],
                                                                                               axis=-1))[:, :, None],
                                                                         [1, 1, 5]),
                                                                _np.tile(_np.transpose(_np.sum(args[0],
                                                                                               axis=-1))[:, :, None],
                                                                         [1, 1, 2]))),
        ('ii->i', [(3, 3)], lambda *args: (_np.eye(3),)),
        ('ki, jk->ij', [(3, 2), (4, 3)], lambda *args: (_np.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                        _np.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('ki, ...k->i...', [(3, 2), (4, 3)], lambda *args: (_np.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                            _np.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('k..., jk', [(3, 2), (4, 3)], lambda *args: (_np.tile(args[1].sum(axis=0)[:, None], [1, 2]),
                                                      _np.tile(args[0].sum(axis=1)[None, :], [4, 1]))),
        ('ij, jk', [(5, 0), (0, 4)], lambda *args: (_np.empty((5, 0)), _np.empty((0, 4)))),
        (('ij,jk,kl->il'), [(2, 2), (2, 5), (5, 2)], lambda *args: (_np.dot(_np.ones((2, 2)), _np.dot(args[1], args[2]).T),
                                                                    _np.dot(args[0].T, _np.dot(_np.ones((2, 2)), args[2].T)),
                                                                    _np.dot(_np.dot(args[0], args[1]).T, _np.ones((2, 2))))),
        # broadcast bug
        ('ij, ij -> i', [(1, 4), (2, 4)], lambda *args: (_np.sum(args[1], axis=0)[None, :],
                                                         _np.tile(args[0], [2, 1]))),
        # issue #16576
        # commented due to long running time
        # ('abiz,abjz->abij', [(64, 8, 128, 512), (64, 8, 128, 512)], lambda *args: (_np.matmul(_np.ones((64, 8, 128, 128)), args[1]),
        #                                                                            _np.matmul(_np.ones((64, 8, 128, 128)), args[0]))),
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
                        tmp = _np.array(_np.random.uniform(-1.0, 1.0, shape), dtype=dtype)
                        x_np.append(tmp.astype(acc_type[dtype]))
                        x.append(np.array(tmp, dtype=dtype))
                        x[-1].attach_grad()
                    expected_np = _np.einsum(subscripts, *x_np, optimize=optimize).astype(dtype)
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
                    expected_np = _np.einsum(subscripts, *x_np, optimize=optimize)
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
                atol = 1e-4 if dtype == 'float16' else 1e-5
                grad = []
                x_np = []
                for shape in operands:
                    x_np.append(_np.array(_np.random.uniform(-2.0, 2.0, shape),
                                          dtype=dtype))
                for optimize in [False, True]:
                    x = []
                    for (iop, op) in enumerate(operands):
                        x.append(np.array(x_np[iop], dtype=dtype))
                        x[-1].attach_grad()
                    test_einsum = TestEinsum(subscripts, optimize)
                    if hybridize:
                        test_einsum.hybridize()
                    expected_np = _np.einsum(subscripts, *[op.astype(acc_type[dtype]) for op in x_np],
                                             optimize=optimize).astype(dtype)
                    with mx.autograd.record():
                        out_mx = test_einsum(*x)
                    assert out_mx.shape == expected_np.shape
                    assert_almost_equal(out_mx.asnumpy(), expected_np, rtol=rtol, atol=atol)
                    out_mx.backward()
                    cur_grad = []
                    for (iop, op) in enumerate(x):
                        cur_grad.append(op.grad.asnumpy())
                    grad.append(cur_grad)
                for (iop, op) in enumerate(grad[0]):
                    assert_almost_equal(grad[0][iop], grad[1][iop], rtol=rtol, atol=atol)


@with_seed()
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
    ctx = mx.context.current_context()
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
            samples, ctx=ctx, dtype=dtype).asnumpy()
        verify_generator(generator=generator_mx, buckets=buckets,
                         probs=probs, nsamples=samples, nrepeat=trials)
        generator_mx_same_seed =\
            lambda x: _np.concatenate(
                [np.random.rand(x // 10, ctx=ctx, dtype=dtype).asnumpy()
                    for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets,
                         probs=probs, nsamples=samples, nrepeat=trials)


@with_seed()
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
        if _np.issubdtype(dtype, _np.integer) or (dtype is np.bool):
            assert out_mx.dtype == np.float32
        else:
            assert out_mx.dtype == dtype
        out_np = _np.true_divide(a.asnumpy(), b.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

        val = _np.random.randint(3, 50)
        out_mx = a / val
        out_np = _np.true_divide(a.asnumpy(), val)
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

        out_mx = val / a
        out_np = _np.true_divide(val, a.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

    for shape_pair, itype, ftype in itertools.product(shapes, itypes, ftypes):
        i_ = np.random.uniform(3, 50, size=shape_pair[0]).astype(itype)
        f_ = np.random.uniform(3, 50, size=shape_pair[-1]).astype(ftype)

        out_mx = i_ / f_
        assert out_mx.dtype == ftype
        out_np = _np.true_divide(i_.asnumpy(), f_.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)

        out_mx = f_ / i_
        assert out_mx.dtype == ftype
        out_np = _np.true_divide(f_.asnumpy(), i_.asnumpy())
        assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-3, atol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_np_column_stack():
    class TestColumnStack(HybridBlock):
        def __init__(self):
            super(TestColumnStack, self).__init__()

        def hybrid_forward(self, F, a, *args):
            return F.np.column_stack([a] + list(args))

    def g(data):
        return _np.ones_like(data)

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
            v_np.append(_np.array(_np.random.uniform(-10.0, 10.0, config[i]), dtype=dtype))
            v.append(mx.nd.array(v_np[i]).as_np_ndarray())
            v[i].attach_grad()
        expected_np = _np.column_stack(v_np)
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
        expected_np = _np.column_stack(v_np)
        assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


def test_npx_reshape():
    class TestNumpyXReshape(HybridBlock):
        def __init__(self, newshape, reverse):
            super(TestNumpyXReshape, self).__init__()
            self._newshape = newshape
            self._reverse = reverse

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.npx.reshape(a, self._newshape, reverse=self._reverse)

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
                expected_grad = _np.ones(shape)
                if grad_req == 'add':
                    expected_grad += init_a_grad.asnumpy()
                assert_almost_equal(a.grad.asnumpy(), expected_grad, rtol=1e-3, atol=1e-5)

                # test imperative
                npx_out = npx.reshape(a, newshape, reverse=reverse)
                expected_out = _np.reshape(a.asnumpy(), expected_ret_shape)
                assert_almost_equal(npx_out.asnumpy(), expected_out, rtol=1e-3, atol=1e-5)


@with_seed()
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


@with_seed()
@use_np
def test_np_diff():
    def np_diff_backward(ograd, n, axis):
        res = ograd
        for i in range(n):
            res = _np.negative(_np.diff(res, n=1, axis=axis, prepend=0, append=0))
        return res

    class TestDiff(HybridBlock):
        def __init__(self, n=1, axis=-1):
            super(TestDiff, self).__init__()
            self._n = n
            self._axis = axis

        def hybrid_forward(self, F, a):
            return F.np.diff(a, n=self._n, axis=self._axis)

    shapes = [tuple(random.randrange(10) for i in range(random.randrange(6))) for j in range(5)]
    for hybridize in [True, False]:
        for shape in shapes:
            for axis in [i for i in range(-len(shape), len(shape))]:
                for n in [i for i in range(0, shape[axis]+1)]:
                    test_np_diff = TestDiff(n=n, axis=axis)
                    if hybridize:
                        test_np_diff.hybridize()
                    for itype in [_np.float16, _np.float32, _np.float64]:
                        # note the tolerance shall be scaled by the input n
                        if itype == _np.float16:
                            rtol = atol = 1e-2*len(shape)*n
                        else:
                            rtol = atol = 1e-5*len(shape)*n
                        x = rand_ndarray(shape).astype(itype).as_np_ndarray()
                        x.attach_grad()
                        np_out = _np.diff(x.asnumpy(), n=n, axis=axis)
                        with mx.autograd.record():
                            mx_out = test_np_diff(x)
                        assert mx_out.shape == np_out.shape
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)
                        mx_out.backward()
                        if (np_out.size == 0):
                            np_backward = _np.zeros(shape)
                        else:
                            np_backward = np_diff_backward(_np.ones(np_out.shape, dtype=itype), n=n, axis=axis)
                        assert x.grad.shape == np_backward.shape
                        assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=rtol, atol=atol)

                        mx_out = np.diff(x, n=n, axis=axis)
                        np_out = _np.diff(x.asnumpy(), n=n, axis=axis)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=rtol, atol=atol)


@with_seed()
@use_np
def test_np_column_stack():
    class TestColumnStack(HybridBlock):
        def __init__(self):
            super(TestColumnStack, self).__init__()

        def hybrid_forward(self, F, a, *args):
            return F.np.column_stack([a] + list(args))

    def g(data):
        return _np.ones_like(data)

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
            v_np.append(_np.array(_np.random.uniform(-10.0, 10.0, config[i]), dtype=dtype))
            v.append(mx.nd.array(v_np[i]).as_np_ndarray())
            v[i].attach_grad()
        expected_np = _np.column_stack(v_np)
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
        expected_np = _np.column_stack(v_np)
        assert_almost_equal(mx_out.asnumpy(), expected_np, rtol=rtol, atol=atol)


@with_seed()
@use_np
def test_np_expand_dims():
    class TestExpandDims(HybridBlock):
        def __init__(self, axis):
            super(TestExpandDims, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, x):
            return F.np.expand_dims(x, self._axis)

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
            x_np = _np.random.uniform(0, 100, size=shape).astype(dtype)
            expected = _np.expand_dims(x_np, axis)
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
                    assert same(x.grad.asnumpy(), _np.ones_like(x.asnumpy()))
                else:
                    assert_almost_equal(x.grad.asnumpy(), initial_grad.asnumpy() + _np.ones_like(initial_grad.asnumpy()),
                                        atol=1e-2 if dtype is np.float16 else 1e-4,
                                        rtol=1e-2 if dtype is np.float16 else 1e-4,
                                        use_broadcast=False)

                # check imperative again
                y = np.expand_dims(x, axis)
                assert_almost_equal(y.asnumpy(), expected, use_broadcast=False)


if __name__ == '__main__':
    import nose
    nose.runmodule()
