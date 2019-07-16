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


if __name__ == '__main__':
    import nose
    nose.runmodule()
