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
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient
from common import with_seed
import random


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
    ]

    eps = 1e-3

    for shape_a, shape_b in shapes:
        np_a = _np.random.uniform(-1.0, 1.0, shape_a)
        np_a[abs(np_a) < eps] = 2 * eps;
        np_b = _np.random.uniform(-1.0, 1.0, shape_b)
        np_b[abs(np_b) < eps] = 2 * eps;
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
    ret = data.reshape(shape=())
    assert type(ret) == mx.sym.np._Symbol

    data = np.ones((1, 1, 1))
    ret = np.reshape(data, ())
    assert ret.shape == ()
    ret = np.reshape(ret, (1, 1, 1, 1))
    assert ret.shape == (1, 1, 1, 1)
    assert type(ret) == np.ndarray


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

        print(func)
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
                print(mx_test_data.grad.asnumpy())
                print(ref_grad(np_test_data))
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


if __name__ == '__main__':
    import nose
    nose.runmodule()
