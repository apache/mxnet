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
from mxnet.test_utils import check_numeric_gradient
from common import assertRaises, with_seed
import random
import collections


@with_seed()
@npx.use_np_shape
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
        ((3, 4, 5, 6, 7), (5, 6, 7, 1, 2), 3),
        ((3, 5, 4, 6, 7), (7, 6, 5, 1, 2), [[1, 3, 4], [2, 1, 0]]),
        ((3, 5, 4), (5, 4, 3), [[1, 0, 2], [0, 2, 1]]),
        ((3, 5, 4), (5, 3, 4), [[2, 0], [2, 1]]),
        ((2, 2), (2, 2), 2),
        ((3, 5, 4), (5, ), [[1], [0]]),
        ((2,), (2, 3), 1),
        ((3,), (3,), 0),
        ((2,), (2, 3), 0),
        ((3, 5, 4), (5, ), 0),
        ((2, 3, 4), (4, 3, 2), [[], []])
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
                a_sym = mx.sym.Variable("a").as_np_ndarray()
                b_sym = mx.sym.Variable("b").as_np_ndarray()
                mx_sym = mx.sym.np.tensordot(a_sym, b_sym, axes).as_nd_ndarray()
                check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                  rtol=1e-1, atol=1e-1, dtype = dtype)

    # test zero size input
    zero_shapes = [
        ((3, 0), (0, 5), 1),
        ((3, 0), (0, 4), [1, 0]),
        ((0, 3), (3, 5), 1)
    ]

    for hybridize in [True, False]:
        for a_shape, b_shape, axes in zero_shapes:
            for dtype in [_np.float32, _np.float64]:
                test_tensordot = TestTensordot(axes)
                if hybridize:
                    test_tensordot.hybridize()
                a = rand_ndarray(shape = a_shape, dtype = dtype).as_np_ndarray()
                b = rand_ndarray(shape = b_shape, dtype = dtype).as_np_ndarray()

                np_out = _np.tensordot(a.asnumpy(), b.asnumpy(), axes)
                with mx.autograd.record():
                    mx_out = test_tensordot(a, b)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)

                # Test imperative once again
                mx_out = np.tensordot(a, b, axes)
                np_out = _np.tensordot(a.asnumpy(), b.asnumpy(), axes)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@npx.use_np_shape
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
                'int8': 'int32', 'int32': 'int64', 'int64': 'int64'}
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                    for dtype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                        if is_int(dtype) and not is_int(itype):
                            continue
                        # test gluon
                        test_sum = TestSum(axis=axis, dtype=dtype, keepdims=keepdims)
                        if hybridize:
                            test_sum.hybridize()
                        if is_int(itype):
                            x = _np.random.randint(-128, 128, shape, dtype=itype)
                            x = mx.nd.array(x)
                        else:
                            x = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype=itype)
                        x = x.as_np_ndarray()
                        x.attach_grad()
                        expected_ret = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_sum(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                            atol=1e-5 if dtype == 'float16' else 1e-5)

                        y.backward()
                        assert same(x.grad.asnumpy(), _np.ones(shape=x.shape, dtype=x.dtype))

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.sum(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@npx.use_np_shape
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
@npx.use_np_shape
def test_np_mean():
    @npx.use_np_shape
    class TestMean(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestMean, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.mean(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' in dtype

    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    acc_type = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64',
                'int8': 'int32', 'int32': 'int64', 'int64': 'int64'}
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float16', 'float32', 'float64']:
                    for dtype in ['float16', 'float32', 'float64']:
                        print(itype, dtype)
                        if is_int(dtype) and not is_int(itype):
                            continue
                        # test gluon
                        test_mean = TestMean(axis=axis, dtype=dtype, keepdims=keepdims)
                        if hybridize:
                            test_mean.hybridize()
                        if is_int(itype):
                            x = _np.random.randint(-128, 128, shape, dtype=itype)
                            x = mx.nd.array(x, dtype=itype)
                        else:
                            x = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype=itype)
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


@with_seed()
@npx.use_np_shape
def test_np_max():
    @npx.use_np_shape
    class TestMax(HybridBlock):
        def __init__(self, axis=None, keepdims=False):
            super(TestMax, self).__init__()
            self._axis = axis
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.max(a, axis=self._axis, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' == dtype

    def get_grad(axis):
        if axis == ():
            return _np.ones((2,3,4,5))
        else:
            temp = _np.zeros((2,3,4,5))
            if axis == 0:
                temp[-1,:,:,:] = 1
                return temp
            elif axis == 1:
                temp[:,-1,:,:] = 1
                return temp
            elif axis == 2:
                temp[:,:,-1,:] = 1
                return temp
            elif axis == 3:
                temp[:,:,:,-1] = 1
                return temp
            elif not axis:
                temp[-1,-1,-1,-1] = 1
                return temp
            raise ValueError('axis should be int or None or ()')

    def _test_np_max_exception(shape, dim):
        x = _np.random.uniform(-1.0, 1.0, shape)
        x = mx.nd.array(x).as_np_ndarray()
        out = mx.np.max(x)
        assert out.ndim == dim, 'dimension mismatch, output.ndim={}, dim={}'.format(output.ndim, dim)

    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float16', 'float32', 'float64', 'int']:
                    # test gluon
                    test_max = TestMax(axis=axis, keepdims=keepdims)
                    if hybridize:
                        test_max.hybridize()
                    if is_int(itype):
                        x = mx.nd.arange(120).reshape((2, 3, 4, 5))
                        x = mx.nd.array(x)
                    else:
                        x = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype=itype)
                    x = x.as_np_ndarray()
                    x.attach_grad()
                    expected_ret = _np.amax(x.asnumpy(), axis=axis, keepdims=keepdims)
                    with mx.autograd.record():
                        y = test_max(x)
                    assert y.shape == expected_ret.shape
                    assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if itype == 'float16' else 1e-3,
                                        atol=1e-5 if itype == 'float16' else 1e-5)
                    y.backward()
                    # only check the gradient with hardcoded input
                    if is_int(itype):
                        assert same(x.grad.asnumpy(), get_grad(axis)), \
                            'x={}\ny={}\nx.grad={}\nnumpy={}'.format(x.asnumpy(), y.asnumpy(), x.grad.asnumpy(), get_grad(axis))

                    # test imperative
                    mx_out = np.max(x, axis=axis, keepdims=keepdims)
                    np_out = _np.amax(x.asnumpy(), axis=axis, keepdims=keepdims)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

    # test zero and zero dim
    shapes = [(), (0), (2, 0), (0, 2, 1)]
    exceptions = [False, True, True, True]
    dims = [0] * len(shapes)
    for shape, exception, dim in zip(shapes, exceptions, dims):
        if exception:
            assertRaises(MXNetError, _test_np_max_exception, shape, dim)
        else:
            _test_np_max_exception(shape, dim)


@with_seed()
@npx.use_np_shape
def test_np_transpose():
    # TODO(junwu): Add more test cases
    data = mx.sym.var('a').as_np_ndarray()
    ret = data.transpose()
    assert type(ret) == mx.sym.np._Symbol

    dtypes = ['float32', 'int32']
    for dtype in dtypes:
        for ndim in [0, 1, 2, 3, 4, 5, 6]:
            shape = rand_shape_nd(ndim, dim=5, allow_zero_size=True)
            np_data = _np.random.uniform(low=-100, high=100, size=shape).astype(dtype)
            mx_data = np.array(np_data, dtype=dtype)
            axes = [None]
            if ndim == 0:
                axes += [()]
            else:
                axis = [i for i in range(ndim)]
                axes.append(tuple(axis))
                random.shuffle(axis)
                axes.append(tuple(axis))
            for axis in axes:
                np_out = _np.transpose(np_data, axes=axis)
                mx_out = np.transpose(mx_data, axes=axis)
                assert np_out.dtype == mx_out.dtype
                assert same(mx_out.asnumpy(), np_out)
    # TODO(junwu): Add numerical gradient test and Gluon API test.


@with_seed()
@npx.use_np_shape
def test_npx_relu():
    # TODO(junwu): Add more test cases
    data = mx.sym.var('data').as_np_ndarray()
    ret = mx.sym.npx.relu(data)
    assert type(ret) == mx.sym.np._Symbol

    shapes = [(), (0, 2, 0)]
    shapes.extend([rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)])
    for shape in shapes:
        data = np.array(_np.random.uniform(size=shape).astype('float32'))
        ret = npx.relu(data)
        assert type(ret) == np.ndarray


@with_seed()
@npx.use_np_shape
def test_npx_sigmoid():
    # TODO(junwu): Add more test cases
    data = mx.sym.var('data').as_np_ndarray()
    ret = mx.sym.npx.sigmoid(data)
    assert type(ret) == mx.sym.np._Symbol

    shapes = [(), (0, 2, 0)]
    shapes.extend([rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)])
    for shape in shapes:
        data = np.array(_np.random.uniform(size=shape).astype('float32'))
        ret = npx.sigmoid(data)
        assert type(ret) == np.ndarray


@with_seed()
@npx.use_np_shape
def test_np_reshape():
    # TODO(junwu): Add more test cases
    data = mx.sym.var('a').as_np_ndarray()
    ret = data.reshape(())
    assert type(ret) == mx.sym.np._Symbol

    data = np.ones((1, 1, 1))
    ret = np.reshape(data, ())
    assert ret.shape == ()
    ret = np.reshape(ret, (1, 1, 1, 1))
    assert ret.shape == (1, 1, 1, 1)
    assert type(ret) == np.ndarray
    ret2 = ret.reshape(1, 1, -1)
    assert ret2.shape == (1, 1, 1)


@with_seed()
@npx.use_np_shape
def test_np_maximum():
    # TODO(junwu): Add more test cases
    x1, x2 = mx.sym.var('x1').as_np_ndarray(), mx.sym.var('x2').as_np_ndarray()
    ret = mx.sym.np.maximum(x1, x2)
    assert type(ret) == mx.sym.np._Symbol

    def check_maximum(x1, x2):
        mx_out = np.maximum(x1, x2)
        if isinstance(x1, np.ndarray) or isinstance(x2, np.ndarray):
            assert type(mx_out) == np.ndarray
        np_out = _np.maximum(x1.asnumpy() if isinstance(x1, np.ndarray) else x1,
                             x2.asnumpy() if isinstance(x2, np.ndarray) else x2)
        assert same(mx_out.asnumpy() if isinstance(mx_out, np.ndarray) else mx_out, np_out)

    check_maximum(np.zeros((2, 1)), np.ones((5, 1, 4)))
    check_maximum(np.zeros((2, 0)), np.ones((5, 1, 1)))
    check_maximum(np.zeros(()), np.ones((5, 1, 4)))


@with_seed()
@npx.use_np_shape
def test_np_minimum():
    # TODO(junwu): Add more test cases
    x1, x2 = mx.sym.var('x1').as_np_ndarray(), mx.sym.var('x2').as_np_ndarray()
    ret = mx.sym.np.minimum(x1, x2)
    assert type(ret) == mx.sym.np._Symbol

    def check_minimum(x1, x2):
        mx_out = np.minimum(x1, x2)
        if isinstance(x1, np.ndarray) or isinstance(x2, np.ndarray):
            assert type(mx_out) == np.ndarray
        np_out = _np.minimum(x1.asnumpy() if isinstance(x1, np.ndarray) else x1,
                             x2.asnumpy() if isinstance(x2, np.ndarray) else x2)
        assert same(mx_out.asnumpy() if isinstance(mx_out, np.ndarray) else mx_out, np_out)

    check_minimum(np.zeros((2, 1)), np.ones((5, 1, 4)))
    check_minimum(np.zeros((2, 0)), np.ones((5, 1, 1)))
    check_minimum(np.zeros(()), np.ones((5, 1, 4)))


@with_seed()
@npx.use_np_shape
def test_np_unary_funcs():
    def check_unary_func(func, ref_grad, shape, low, high):
        @npx.use_np_shape
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

            if ref_grad:
                y.backward()
                assert_almost_equal(mx_test_data.grad.asnumpy(), ref_grad(np_test_data), rtol=1e-5, atol=1e-6, equal_nan=True)

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
    ndim = random.choice([2, 3, 4])
    shape = random.choice([rand_shape_nd(ndim, dim=3), (1, 0, 2)])
    for shape in [rand_shape_nd(ndim, dim=3), (1, 0, 2)]:
        for func, func_data in funcs.items():
            ref_grad, low, high = func_data
            check_unary_func(func, ref_grad, shape, low, high)


@with_seed()
@npx.use_np_shape
def test_np_stack():
    @npx.use_np_shape
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
                assert y.shape == expected_ret.shape
                assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)

                y.backward()

                assert_almost_equal(mx_a.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_b.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_c.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(mx_d.grad.asnumpy(), _np.ones(shape), rtol=1e-3, atol=1e-5)

                np_out = _np.stack([np_a, np_b, np_c, np_d], axis=axis)
                mx_out = np.stack([mx_a, mx_b, mx_c, mx_d], axis=axis)
                assert same(mx_out.asnumpy(), np_out)


@with_seed()
@npx.use_np_shape
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

    @npx.use_np
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
@npx.use_np_shape
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

    @npx.use_np
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
@npx.use_np_shape
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
        assert_almost_equal(mx.np.linspace(0, test_index, test_index + 1).asnumpy(), mx.np.arange(test_index + 1).asnumpy())
    @npx.use_np
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
@npx.use_np_shape
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
    @npx.use_np
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
@npx.use_np_shape
def test_np_argmax():
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

    @npx.use_np
    class TestArgMax(HybridBlock):
        def __init__(self, axis=None):
            super(TestArgMax, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, x):
            return F.np.argmax(x, self._axis)

    for shape, axis, throw_exception in workloads:
        for dtype in dtypes:
            a = np.random.uniform(size=shape, dtype=dtype)
            if throw_exception:
                # Cannot use assert_exception because sometimes the main thread
                # proceeds to `assert False` before the exception is thrown
                # in the worker thread. Have to use mx.nd.waitall() here
                # to block the main thread.
                try:
                    np.argmax(a, axis)
                    mx.nd.waitall()
                    assert False
                except mx.MXNetError:
                    pass
            else:
                mx_ret = np.argmax(a, axis=axis)
                np_ret = _np.argmax(a.asnumpy(), axis=axis)
                assert same(mx_ret.asnumpy(), np_ret)

            for hybridize in [False, True]:
                net = TestArgMax(axis)
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
@npx.use_np_shape
def test_np_argsort():
    @npx.use_np_shape
    class TestArgsort(HybridBlock):
        def __init__(self, axis=-1):
            super(TestArgsort, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, a):
            return F.np.argsort(a, self._axis)

    shapes = [
        (), 
        (1,), 
        (5,4),
        (5,0,4),
        (5,0,0),
        (0,0,5),
        (0,0,0),
        (5,3,4)
    ] 
    for hybridize in [True, False]:
        for shape in shapes:
            for ax in list(range(len(shape))) + [-1, None]:
                test_argsort = TestArgsort(ax)
                if hybridize:
                    test_argsort.hybridize()

                x = np.random.uniform(size=shape)
                np_out = _np.argsort(x.asnumpy(), axis=ax)
                mx_out = test_argsort(x)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                # Test imperative once again
                mx_out = np.argsort(x, axis=ax)
                np_out = _np.argsort(x.asnumpy(), axis=ax)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@npx.use_np_shape
def test_np_linalg_norm():
    @npx.use_np
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
@npx.use_np_shape
def test_np_concat():
    class TestConcat(HybridBlock):
        def __init__(self, axis=None):
            super(TestConcat, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, a, *args):
            return F.np.concatenate([a] + list(args), axis=self._axis)

    def get_new_shape(shape, axis):
        shape_lst = list(shape)
        shape_lst[axis] = random.randint(0, 3)
        return tuple(shape_lst)

    for shape in [(0, 0), (2, 3)]:
        for hybridize in [True, False]:
            for axis in range(2):
                # test gluon
                test_concat = TestConcat(axis=axis)
                if hybridize:
                    test_concat.hybridize()

                a = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                a.attach_grad()
                b = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                b.attach_grad()
                c = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                c.attach_grad()
                d = mx.nd.random.uniform(-1.0, 1.0, shape=get_new_shape(shape, axis)).as_np_ndarray()
                d.attach_grad()
                expected_ret = _np.concatenate([a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()], axis=axis)
                with mx.autograd.record():
                    y = test_concat(a, b, c, d)
                assert y.shape == expected_ret.shape
                assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)

                y.backward()

                assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(b.grad.asnumpy(), _np.ones(b.shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(c.grad.asnumpy(), _np.ones(c.shape), rtol=1e-3, atol=1e-5)
                assert_almost_equal(d.grad.asnumpy(), _np.ones(d.shape), rtol=1e-3, atol=1e-5)

                # test imperative
                mx_out = np.concatenate([a, b, c, d], axis=axis)
                np_out = _np.concatenate([a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()], axis=axis)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@npx.use_np_shape
def test_np_hstack():
    class TestHStack(HybridBlock):
        def __init__(self):
            super(TestHStack, self).__init__()
        
        def hybrid_forward(self, F, a, *args):
            return F.np.hstack([a] + list(args))

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
            np_out = _np.hstack((a.asnumpy(), b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

            # test symbolic backward
            mx_out.backward()
            assert_almost_equal(a.grad.asnumpy(), _np.ones(a.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(b.grad.asnumpy(), _np.ones(b.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(c.grad.asnumpy(), _np.ones(c.shape), rtol=1e-3, atol=1e-5)
            assert_almost_equal(d.grad.asnumpy(), _np.ones(d.shape), rtol=1e-3, atol=1e-5)

            # test imperative
            mx_out = np.hstack((a, b, c, d))
            np_out = _np.hstack((a.asnumpy(),b.asnumpy(), c.asnumpy(), d.asnumpy()))
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@npx.use_np_shape
def test_np_swapaxes():
    config = [((0, 1, 2), 0, 1),
              ((0, 1, 2), -1, -2),
              ((4, 5, 6, 7), 2, 3),
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
@npx.use_np_shape
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
            return F.np.squeeze(x, axis=self._axis)

    for shape, axis in config:
        data_np = _np.random.uniform(size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = _np.squeeze(data_np, axis=axis)
        ret_mx = np.squeeze(data_mx, axis=axis)
        assert same(ret_mx.asnumpy(), ret_np)

        net = TestSqueeze(axis)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            ret_mx = net(data_mx)
            assert same(ret_mx.asnumpy(), ret_np)


@with_seed()
@npx.use_np_shape
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
@npx.use_np_shape
def test_np_cumsum():
    def np_cumsum_backward(ograd, axis=None, dtype=None):
        return _np.flip(_np.cumsum(_np.flip(ograd, axis=axis), axis=axis, dtype=dtype), axis=axis)

    @npx.use_np_shape
    class TestCumsum(HybridBlock):
        def __init__(self, axis=None, dtype=None):
            super(TestCumsum, self).__init__()
            self._axis = axis
            self._dtype = dtype

        def hybrid_forward(self, F, a):
            return F.np.cumsum(a, axis=self._axis, dtype=self._dtype)

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


@with_seed()
@npx.use_np_shape
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
        data_np = _np.random.uniform(size=shape)
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
@npx.use_np_shape
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
                        x = np.random.uniform(-2.0, 2.0, size=shape, dtype=itype)
                        x.attach_grad()
                        print(x.grad.dtype)
                        expected_ret = _np.prod(x.asnumpy(), axis=axis, keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_prod(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5)
                        y.backward()
                        # use keepdims=True so that broadcast divide can be used to calculate
                        # grad of input
                        expected_ret = _np.prod(x.asnumpy(), axis=axis, keepdims=True)
                        assert_almost_equal(x.grad.asnumpy(), expected_ret / x.asnumpy(), rtol=1e-3, atol=1e-3)

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.prod(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.prod(x.asnumpy(), axis=axis, keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@npx.use_np
def test_np_flatten():
    # TODO(junwu): Add more test cases
    shapes = [(), (2, 0, 1), (3, 4, 5), 6]
    for shape in shapes:
        a = _np.random.uniform(size=shape).astype('float32')
        a_mx = np.array(a, dtype=a.dtype)
        expected_ret = a.flatten()
        ret_mx = a_mx.flatten()
        assert same(expected_ret, ret_mx.asnumpy())


@with_seed()
@npx.use_np
def test_np_broadcast_to():
    # TODO(junwu): Add more test cases and backward test
    shapes = [(1, 2, 3, 4, 5), (1, 0, 3, 4, 5)]
    for shape in shapes:
        a = _np.random.uniform(size=(4, 1)).astype('float32')
        a_mx = np.array(a, dtype=a.dtype)
        expected_ret = _np.broadcast_to(a, shape)
        ret_mx = np.broadcast_to(a_mx, shape)
        assert same(expected_ret, ret_mx.asnumpy())


@with_seed()
@npx.use_np
def test_np_meshgrid():
    nx, ny = (4, 5)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.ones(())
    xv, yv, zv = np.meshgrid(x, y, z)
    xv_expected, yv_expected, zv_expected = _np.meshgrid(x.asnumpy(), y.asnumpy(), z.asnumpy())
    assert same(xv.asnumpy(), xv_expected)
    assert same(yv.asnumpy(), yv_expected)
    assert same(zv.asnumpy(), zv_expected)
    # TODO(junwu): Add more test


@with_seed()
@npx.use_np
def test_np_broadcast_arrays():
    # TODO(junwu): Add test
    pass


@with_seed()
@npx.use_np
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


if __name__ == '__main__':
    import nose
    nose.runmodule()
