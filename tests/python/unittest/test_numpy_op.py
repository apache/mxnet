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
from mxnet import np, npe
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient
from common import with_seed
import random


@with_seed()
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
                            mx_sym = mx.sym.np.sum(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_classic_ndarray()
                            check_numeric_gradient(mx_sym, [x], numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@np.use_np_compat
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
        mx_sym = mx.sym.np.dot(mx_a.as_np_ndarray(), mx_b.as_np_ndarray()).as_classic_ndarray()
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
def test_np_mean():
    @np.use_np_compat
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
                            mx_sym = mx.sym.np.mean(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_classic_ndarray()
                            check_numeric_gradient(mx_sym, [x], numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.mean(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.mean(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@with_seed()
@np.use_np_compat
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
@np.use_np_compat
def test_relu():
    # TODO(junwu): Add more test cases
    data = mx.sym.var('data').as_np_ndarray()
    ret = mx.sym.npe.relu(data)
    assert type(ret) == mx.sym.np._Symbol

    shapes = [(), (0, 2, 0)]
    shapes.extend([rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)])
    for shape in shapes:
        data = np.array(_np.random.uniform(size=shape).astype('float32'))
        ret = npe.relu(data)
        assert type(ret) == np.ndarray


@with_seed()
@np.use_np_compat
def test_sigmoid():
    # TODO(junwu): Add more test cases
    data = mx.sym.var('data').as_np_ndarray()
    ret = mx.sym.npe.sigmoid(data)
    assert type(ret) == mx.sym.np._Symbol

    shapes = [(), (0, 2, 0)]
    shapes.extend([rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)])
    for shape in shapes:
        data = np.array(_np.random.uniform(size=shape).astype('float32'))
        ret = npe.sigmoid(data)
        assert type(ret) == np.ndarray


@with_seed()
@np.use_np_compat
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
@np.use_np_compat
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
@np.use_np_compat
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


if __name__ == '__main__':
    import nose
    nose.runmodule()
