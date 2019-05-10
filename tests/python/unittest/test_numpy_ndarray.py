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
from __future__ import division
import numpy as _np
import mxnet as mx
from mxnet import numpy as np
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray, assert_exception
from common import with_seed


@with_seed()
def test_array_creation():
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, None]
    objects = [[], (), [[1, 2], [3, 4]],
               _np.random.uniform(size=rand_shape_nd(3, allow_zero_size=True)),
               mx.nd.array(_np.random.uniform(size=rand_shape_nd(3, allow_zero_size=True)))]
    for dtype in dtypes:
        for src in objects:
            mx_arr = np.array(src, dtype=dtype)
            assert mx_arr.context == mx.current_context()
            if isinstance(src, mx.nd.NDArray):
                np_arr = _np.array(src.asnumpy(), dtype=dtype)
            else:
                np_arr = _np.array(src, dtype=dtype)
            assert same(mx_arr.asnumpy(), np_arr)
            assert mx_arr.dtype == np_arr.dtype


@with_seed()
@mx.use_np_compat
def test_zeros():
    # test np.zeros in Gluon
    class TestZeros(HybridBlock):
        def __init__(self, shape, dtype=None):
            super(TestZeros, self).__init__()
            self._shape = shape
            self._dtype = dtype

        def hybrid_forward(self, F, x, *args, **kwargs):
            return x + F.np.zeros(shape, dtype)

    class TestZerosOutputType(HybridBlock):
        def hybrid_forward(self, F, x, *args, **kwargs):
            return x, F.np.zeros(shape=())

    # test np.zeros in imperative
    def check_zero_array_creation(shape, dtype):
        np_out = _np.zeros(shape=shape, dtype=dtype)
        mx_out = np.zeros(shape=shape, dtype=dtype)
        assert same(mx_out.asnumpy(), np_out)
        if dtype is None:
            assert mx_out.dtype == _np.float32
            assert np_out.dtype == _np.float64

    shapes = [(0,), (2, 0, 2), (0, 0, 0, 0), ()]
    shapes += [rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)]
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, None]
    for shape in shapes:
        for dtype in dtypes:
            check_zero_array_creation(shape, dtype)
            x = mx.nd.array(_np.random.uniform(size=shape), dtype=dtype)
            if dtype is None:
                x = x.astype('float32')
            for hybridize in [True, False]:
                test_zeros = TestZeros(shape, dtype)
                test_zeros_output_type = TestZerosOutputType()
                if hybridize:
                    test_zeros.hybridize()
                    test_zeros_output_type.hybridize()
                y = test_zeros(x)
                assert type(y) == np.ndarray
                assert same(x.asnumpy(), y.asnumpy())
                y = test_zeros_output_type(x)
                assert type(y[1]) == np.ndarray


@with_seed()
@mx.use_np_compat
def test_ones():
    # test np.ones in Gluon
    class TestOnes(HybridBlock):
        def __init__(self, shape, dtype=None):
            super(TestOnes, self).__init__()
            self._shape = shape
            self._dtype = dtype

        def hybrid_forward(self, F, x, *args, **kwargs):
            return x * F.np.ones(shape, dtype)

    class TestOnesOutputType(HybridBlock):
        def hybrid_forward(self, F, x, *args, **kwargs):
            return x, F.np.ones(shape=())

    # test np.ones in imperative
    def check_ones_array_creation(shape, dtype):
        np_out = _np.ones(shape=shape, dtype=dtype)
        mx_out = np.ones(shape=shape, dtype=dtype)
        assert same(mx_out.asnumpy(), np_out)
        if dtype is None:
            assert mx_out.dtype == _np.float32
            assert np_out.dtype == _np.float64

    shapes = [(0,), (2, 0, 2), (0, 0, 0, 0), ()]
    shapes += [rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)]
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, None]
    for shape in shapes:
        for dtype in dtypes:
            check_ones_array_creation(shape, dtype)
            x = mx.nd.array(_np.random.uniform(size=shape), dtype=dtype).as_np_ndarray()
            if dtype is None:
                x = x.astype('float32')
            for hybridize in [True, False]:
                test_ones = TestOnes(shape, dtype)
                test_ones_output_type = TestOnesOutputType()
                if hybridize:
                    test_ones.hybridize()
                    test_ones_output_type.hybridize()
                y = test_ones(x)
                assert type(y) == np.ndarray
                assert same(x.asnumpy(), y.asnumpy())
                y = test_ones_output_type(x)
                assert type(y[1]) == np.ndarray


@with_seed()
@mx.use_np_compat
def test_ndarray_binary_element_wise_ops():
    # Cannot test operators like >, because boolean arrays are not supported yet.
    np_op_map = {'+': _np.add, '*': _np.multiply, '-': _np.subtract, '/': _np.divide,
                 'mod': _np.mod, 'pow': _np.power,
                 # '>': _np.greater, '>=': _np.greater_equal,
                 # '<': _np.less, '<=': _np.less_equal
                 }

    def get_np_ret(x1, x2, op):
        return np_op_map[op](x1, x2)

    class TestBinaryElementWiseOp(HybridBlock):
        def __init__(self, op, scalar=None, reverse=False):
            super(TestBinaryElementWiseOp, self).__init__()
            self._op = op
            self._scalar = scalar
            self._reverse = reverse  # if false, scalar is the right operand.

        def hybrid_forward(self, F, x, *args):
            if self._op == '+':
                if self._scalar is not None:
                    return x + self._scalar if not self._reverse else self._scalar + x
                else:
                    return x + args[0] if not self._reverse else args[0] + x
            elif self._op == '*':
                if self._scalar is not None:
                    return x * self._scalar if not self._reverse else self._scalar * x
                else:
                    return x * args[0] if not self._reverse else args[0] * x
            elif self._op == '-':
                if self._scalar is not None:
                    return x - self._scalar if not self._reverse else self._scalar - x
                else:
                    return x - args[0] if not self._reverse else args[0] - x
            elif self._op == '/':
                if self._scalar is not None:
                    return x / self._scalar if not self._reverse else self._scalar / x
                else:
                    return x / args[0] if not self._reverse else args[0] / x
            elif self._op == 'mod':
                if self._scalar is not None:
                    return x % self._scalar if not self._reverse else self._scalar % x
                else:
                    return x % args[0] if not self._reverse else args[0] % x
            elif self._op == 'pow':
                if self._scalar is not None:
                    return x ** self._scalar if not self._reverse else self._scalar ** x
                else:
                    return x ** args[0] if not self._reverse else args[0] ** x
            elif self._op == '>':
                if self._scalar is not None:
                    return x > self._scalar
                else:
                    return x > args[0]
            elif self._op == '>=':
                if self._scalar is not None:
                    return x >= self._scalar
                else:
                    return x >= args[0]
            elif self._op == '<':
                if self._scalar is not None:
                    return x < self._scalar
                else:
                    return x < args[0]
            elif self._op == '<=':
                if self._scalar is not None:
                    return x <= self._scalar
                else:
                    return x <= args[0]
            else:
                print(self._op)
                assert False

    def check_binary_op_result(shape1, shape2, op, dtype=None):
        if shape1 is None:
            mx_input1 = abs(_np.random.uniform()) + 1
            np_input1 = mx_input1
        else:
            mx_input1 = rand_ndarray(shape1, dtype=dtype).abs() + 1
            np_input1 = mx_input1.asnumpy()
        if shape2 is None:
            mx_input2 = abs(_np.random.uniform()) + 1
            np_input2 = mx_input2
        else:
            mx_input2 = rand_ndarray(shape2, dtype=dtype).abs() + 1
            np_input2 = mx_input2.asnumpy()

        scalar = None
        reverse = False
        if isinstance(mx_input1, mx.nd.NDArray) and not isinstance(mx_input2, mx.nd.NDArray):
            scalar = mx_input2
            reverse = False
        elif isinstance(mx_input2, mx.nd.NDArray) and not isinstance(mx_input1, mx.nd.NDArray):
            scalar = mx_input1
            reverse = True

        np_out = get_np_ret(np_input1, np_input2, op)
        for hybridize in [True, False]:
            if scalar is None:
                get_mx_ret = TestBinaryElementWiseOp(op)
                if hybridize:
                    get_mx_ret.hybridize()
                mx_out = get_mx_ret(mx_input1.as_np_ndarray(), mx_input2.as_np_ndarray())
                assert type(mx_out) == np.ndarray
                assert np_out.shape == mx_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-6, rtol=1e-5)

                mx_out = get_mx_ret(mx_input1, mx_input2.as_np_ndarray())
                assert type(mx_out) == np.ndarray
                assert np_out.shape == mx_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-6, rtol=1e-5)

                mx_out = get_mx_ret(mx_input1.as_np_ndarray(), mx_input2)
                assert type(mx_out) == np.ndarray
                assert np_out.shape == mx_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-6, rtol=1e-5)
            else:
                get_mx_ret = TestBinaryElementWiseOp(op, scalar=scalar, reverse=reverse)
                if hybridize:
                    get_mx_ret.hybridize()
                if reverse:
                    mx_out = get_mx_ret(mx_input2.as_np_ndarray())
                    assert type(mx_out) == np.ndarray
                else:
                    mx_out = get_mx_ret(mx_input1.as_np_ndarray())
                    assert type(mx_out) == np.ndarray
                assert np_out.shape == mx_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-6, rtol=1e-5)

    dtypes = [_np.float32, _np.float64, None]
    ops = np_op_map.keys()
    for dtype in dtypes:
        for op in ops:
            check_binary_op_result((3, 4), (3, 4), op, dtype)
            check_binary_op_result(None, (3, 4), op, dtype)
            check_binary_op_result((3, 4), None, op, dtype)
            check_binary_op_result((1, 4), (3, 1), op, dtype)
            check_binary_op_result(None, (3, 1), op, dtype)
            check_binary_op_result((1, 4), None, op, dtype)
            check_binary_op_result((1, 4), (3, 5, 4), op, dtype)
            check_binary_op_result((), (3, 5, 4), op, dtype)
            check_binary_op_result((), None, op, dtype)
            check_binary_op_result(None, (), op, dtype)
            check_binary_op_result((0, 2), (1, 1), op, dtype)
            check_binary_op_result((0, 2), None, op, dtype)
            check_binary_op_result(None, (0, 2), op, dtype)


@with_seed()
def test_np_op_output_type():
    # test imperative invoke
    data = np.array([1., 3.], dtype='float32')
    ret = np.sum(data)
    assert type(ret) == np.ndarray
    ret = mx.nd.sin(data)
    assert type(ret) == mx.nd.NDArray

    # test cached op
    class TestCachedOpOutputType(HybridBlock):
        @mx.use_np_compat
        def hybrid_forward(self, F, x, *args, **kwargs):
            ret1 = F.sin(x)
            ret2 = F.np.sum(x)
            return ret1, ret2

    net = TestCachedOpOutputType()
    for hybridize in [True, False]:
        if hybridize:
            net.hybridize()
        ret1, ret2 = net(data)
        assert type(ret1) == mx.nd.NDArray
        assert type(ret2) == np.ndarray


@with_seed()
def test_grad_ndarray_type():
    data = np.array(2, dtype=_np.float32)
    data.attach_grad()
    assert type(data.grad) == np.ndarray
    assert type(data.detach()) == np.ndarray


@with_seed()
def test_np_ndarray_astype():
    mx_data = np.array([2, 3, 4, 5], dtype=_np.int32)
    np_data = mx_data.asnumpy()

    def check_astype_equal(dtype, copy, expect_zero_copy=False):
        mx_ret = mx_data.astype(dtype=dtype, copy=copy)
        np_ret = np_data.astype(dtype=dtype, copy=copy)
        assert mx_ret.dtype == np_ret.dtype
        assert same(mx_ret.asnumpy(), np_ret)
        if expect_zero_copy:
            assert id(mx_ret) == id(mx_data)
            assert id(np_ret) == id(np_data)

    for dtype in [_np.int8, _np.uint8, _np.int32, _np.float16, _np.float32, _np.float64]:
        for copy in [True, False]:
            check_astype_equal(dtype, copy, copy is False and mx_data.dtype == dtype)


@with_seed()
def test_np_ndarray_copy():
    mx_data = np.array([2, 3, 4, 5], dtype=_np.int32)
    assert_exception(mx_data.copy, NotImplementedError, order='F')
    mx_ret = mx_data.copy()
    np_ret = mx_data.asnumpy().copy()
    assert same(mx_ret.asnumpy(), np_ret)


if __name__ == '__main__':
    import nose
    nose.runmodule()
