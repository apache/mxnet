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

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.npx.slice(a, begin=self._begin, end=self._end, step=self._step)

    def get_start_end_step(shape):
        start = []
        end = []
        step_switch = random.randint(-1,1)
        step = None if step_switch == 0 else []
        for i in range(len(shape)):
            s = random.randint(0, shape[i]-1)
            e = random.randint(s+1, shape[i])
            if step_switch == 1:
                step.append(1)
                start.append(s)
                end.append(e)
            elif step_switch == -1:
                step.append(-1)
                if e == shape[i]:
                    e -= 1
                    s -= 1
                    if s == -1:
                        s = None
                start.append(e)
                end.append(s)
            else:
                start.append(s)
                end.append(e)
        return start, end, step

    for hybridize in [True, False]:
        for i in range(10):
            dim = random.randint(1,4)
            shape = [random.randint(1,5) for i in range(dim)]

            # test gluon
            start, end, step = get_start_end_step(shape)
            test_slice = TestSlice(begin=start, end=end, step=step)
            if hybridize:
                test_slice.hybridize()

            a = mx.nd.random.uniform(shape=shape).as_np_ndarray()
            a.attach_grad()
            if step is not None:
                expected_ret = a.as_nd_ndarray().slice(start, end, step)
            else:
                expected_ret = a.as_nd_ndarray().slice(start, end)
            with mx.autograd.record():
                y = test_slice(a)

            assert_almost_equal(y.asnumpy(), expected_ret.asnumpy(), rtol=1e-3, atol=1e-5)

            # test backward
            mx.autograd.backward(y)
            expected_grad = _np.zeros(shape)
            basic_index = tuple([ 
                slice(start[i], end[i], step[i]) if step is not None else slice(start[i], end[i])
                for i in range(len(start))
                ])
            expected_grad[basic_index] = 1
            assert_almost_equal(a.grad.asnumpy(), expected_grad, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    import nose
    nose.runmodule()
