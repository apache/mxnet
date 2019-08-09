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


if __name__ == '__main__':
    import nose
    nose.runmodule()
