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
from __future__ import print_function
import numpy as np
import mxnet as mx
import math
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
import unittest


def np_softmax(x, axis=-1):
    # fix for old numpy on Travis not supporting keepdims
    # x = x - np.max(x, axis=-1, keepdims=True)
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    # x /= np.sum(x, axis=-1, keepdims=True)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


def check_elementwise_sum_with_shape(shape, n):
    # forward
    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.ElementWiseSum(*inputs, name='esum')
    arr = [mx.nd.empty(shape) for i in range(n)]
    arr_grad = [mx.nd.empty(shape) for i in range(n)]
    for i in range(n):
        arr[i][:] = np.random.uniform(-10, 10, shape)
    exec1 = out.bind(default_context(),
                     args=arr,
                     args_grad=arr_grad)
    out1 = exec1.outputs[0].asnumpy()
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0].asnumpy()
    out = sum(a.asnumpy() for a  in arr)
    assert_almost_equal(out, out1)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = np.random.uniform(-10, 10, shape)
    # backward
    exec1.backward([out_grad])
    for a in arr_grad:
        assert_almost_equal(a.asnumpy(), out_grad.asnumpy())


def test_elementwise_sum():
    np.random.seed(0)
    nrepeat = 2
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
            check_elementwise_sum_with_shape(shape, np.random.randint(1, 8))


def check_concat_with_shape(shapes, dimension, skip_second):
    # if skip_second is True, second argument will not have gradient.
    # it is to test #1130
    n = len(shapes)
    # forward
    target_dim = 0
    for shape in shapes:
        target_dim += shape[dimension]

    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.Concat(*inputs, name='conc',dim=dimension)
    arr = [mx.nd.empty(shape) for shape in shapes]
    for i in range(n):
        arr[i][:] = shapes[i][dimension]
    arr_np = [np.copy(narray.asnumpy()) for narray in arr]
    arr_grad = [mx.nd.empty(shape) for shape in shapes]
    dict_grad = {}
    arg_names = out.list_arguments()

    for name, g in zip(arg_names, arr_grad):
        if not skip_second or name != 'arg1':
            dict_grad[name] = g

    args = out.list_arguments()
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(**dict(zip(args, shapes)))
    out_grad = mx.nd.empty(out_shapes[0])
    exec1 = out.bind(default_context(),
                     args=arr,
                     args_grad=dict_grad)
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0]
    ret = np.concatenate([narray.asnumpy() for narray in arr], axis=dimension)
    assert_almost_equal(out1.asnumpy(), ret)
    # backward
    out1.copyto(out_grad)
    out_grad[:] += 1
    exec1.backward([out_grad])

    for i, name in enumerate(arg_names):
        if not skip_second or name != 'arg1':
            grad = dict_grad[name]
            np_grad = arr_np[i]
            assert_almost_equal(grad.asnumpy(), np_grad + 1)


def test_concat():
    for dimension in range(4):
        n = 2
        merge = [2, 3, 4, 5, 6]
        a = 2
        b = 3
        c = 4
        # test  2D
        if dimension<2:
            for dim in range(2, 6):
                shapes = []
                for i in range(dim):
                    if dimension == 0:
                        shapes.append((merge[i], a))
                    elif dimension == 1:
                        shapes.append((a, merge[i]))
                    check_concat_with_shape(shapes,dimension,True)
                    check_concat_with_shape(shapes,dimension,False)
                    # Test negative dim
                    check_concat_with_shape(shapes, dimension - 2, True)
                    check_concat_with_shape(shapes, dimension - 2, False)

        #test 3D
        if dimension<3:
            for dim in range(2, 6):
                shapes = []
                for i in range(dim):
                    if dimension == 0:
                        shapes.append((merge[i], a,b))
                    elif dimension ==1:
                        shapes.append((a,merge[i],b))
                    elif dimension ==2:
                        shapes.append((a,b,merge[i]))
                check_concat_with_shape(shapes,dimension,True)
                check_concat_with_shape(shapes,dimension,False)
                # Test negative dim
                check_concat_with_shape(shapes, dimension - 3, True)
                check_concat_with_shape(shapes, dimension - 3, False)
        # test 4D
        for dim in range(2, 6):
            shapes = []
            for i in range(dim):
                if dimension == 0:
                    shapes.append((merge[i],a,b,c))
                elif dimension == 1:
                    shapes.append((a,merge[i],b,c))
                elif dimension ==2:
                    shapes.append((a,b,merge[i],c))
                elif dimension ==3:
                    shapes.append((a,b,c,merge[i]))
            check_concat_with_shape(shapes,dimension,True)
            check_concat_with_shape(shapes,dimension,False)
            # Test negative dim
            check_concat_with_shape(shapes, dimension - 4, True)
            check_concat_with_shape(shapes, dimension - 4, False)

def test_slice_channel():
    def check_slice_channel(data_ndim, axis, num_outputs, squeeze_axis):
        ins = []
        if squeeze_axis:
            shape = np.random.randint(2, 5, data_ndim).tolist()
            shape[axis] = num_outputs
            out_ele_shape = [ele for ele in shape]
            del out_ele_shape[axis]
        else:
            shape = np.random.randint(1, 5, data_ndim).tolist()
            shape[axis] *= num_outputs
            out_ele_shape = [ele for ele in shape]
            out_ele_shape[axis] //= num_outputs
        data_npy = np.random.normal(size=shape)
        out_grads_npy = [np.random.normal(size=out_ele_shape) for i in range(num_outputs)]
        data = mx.sym.Variable('data')
        sym = mx.sym.SliceChannel(data=data, num_outputs=num_outputs, axis=axis, squeeze_axis=squeeze_axis)
        exe = sym.simple_bind(ctx=default_context(), data=data_npy.shape)
        assert len(exe.outputs) == num_outputs
        outputs = exe.forward(is_train=True, data=data_npy)
        for i in range(num_outputs):
            gt = data_npy.take(np.arange(i * shape[axis]/num_outputs,
                                         (i+1) * shape[axis]/num_outputs).astype(np.int), axis=axis)
            if squeeze_axis:

                assert_almost_equal(outputs[i].asnumpy(), gt.reshape(outputs[i].shape))
            else:
                assert_almost_equal(outputs[i].asnumpy(), gt)
        # test backward
        exe.backward(out_grads=[mx.nd.array(ele, ctx=default_context()) for ele in out_grads_npy])
        if squeeze_axis:
            assert_almost_equal(exe.grad_arrays[0].asnumpy(),
                                np.concatenate([np.expand_dims(ele, axis=axis) for ele in out_grads_npy],
                                               axis=axis))
        else:
            assert_almost_equal(exe.grad_arrays[0].asnumpy(),
                                np.concatenate(out_grads_npy, axis=axis))
    check_slice_channel(data_ndim=2, axis=1, num_outputs=3, squeeze_axis=True)
    check_slice_channel(data_ndim=4, axis=2, num_outputs=3, squeeze_axis=False)
    check_slice_channel(data_ndim=3, axis=-1, num_outputs=2, squeeze_axis=False)
    check_slice_channel(data_ndim=5, axis=-2, num_outputs=3, squeeze_axis=True)


def check_regression(symbol, forward, backward):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    out = symbol(data, label)
    shape = (3, 1)
    arr_data = mx.random.uniform(-1, 1, shape, ctx=mx.cpu()).copyto(default_context())
    arr_label = mx.random.uniform(0, 1, shape[0], ctx=mx.cpu()).copyto(default_context())
    arr_grad = mx.nd.empty(shape)
    exec1 = out.bind(default_context(),
                     args=[arr_data, arr_label],
                     args_grad={"data" : arr_grad})
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0].asnumpy()
    npout = forward(arr_data.asnumpy())
    assert_almost_equal(npout, out1)

    exec1.backward()
    npout = backward(npout,  arr_label.asnumpy().reshape(npout.shape))
    assert_almost_equal(npout, arr_grad.asnumpy())


def test_regression():
    check_regression(mx.symbol.LogisticRegressionOutput,
                     lambda x: 1.0 / (1.0 + np.exp(-x)),
                     lambda x, y : x - y)
    check_regression(mx.symbol.LinearRegressionOutput,
                     lambda x: x,
                     lambda x, y : x - y)
    check_regression(mx.symbol.MAERegressionOutput,
                     lambda x: x,
                     lambda x, y : np.where(x > y, np.ones(x.shape), -np.ones(x.shape)))

def check_softmax_grad(xpu):
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((1,4), ctx=xpu)
    label_nd = mx.nd.array([1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, use_ignore=False)
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((1,4))
    expected_grad_out[0, k] = -1
    assert np.isclose(grad_out - softmax_out, expected_grad_out).all()


def check_smoothed_softmax_grad(xpu):
    alpha = 0.2
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((1,4), ctx=xpu)
    label_nd = mx.nd.array([1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, use_ignore=False, smooth_alpha=alpha)
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.full((1,4), fill_value=-alpha/float(4-1))
    expected_grad_out[0, k] = - (1 - alpha)
    assert np.isclose(grad_out - softmax_out, expected_grad_out).all()


def check_softmax_with_ignore_label(xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SoftmaxOutput(data=X, label=L, ignore_label=0, use_ignore=True)

    shape = (20, 10)
    x = mx.nd.empty(shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    x_np = np.random.rand(*shape)
    l_np = np.random.randint(0, shape[1]-1, (shape[0],))
    x[:] = x_np
    l[:] = l_np

    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)
    exec1.backward()

    grad0 = grad.asnumpy()

    for i in range(int(shape[0]/2)):
        l_np[i] = 0
    l[:] = l_np

    exec1.forward(is_train=True)
    exec1.backward()
    grad1 = grad.asnumpy()

    assert abs(np.sum(grad1[:int(shape[0]/2)])) < 1e-5
    assert_almost_equal(grad0[int(shape[0]/2):], grad1[int(shape[0]/2):])


def check_softmax_with_shape(shape, xpu, preserve_shape=False):
    # bind with label
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SoftmaxOutput(data=X, label=L, preserve_shape=preserve_shape)
    x = mx.random.uniform(-1, 1, shape, ctx=xpu)
    l = mx.random.uniform(-1, 1, shape, ctx=xpu)
    l[:] = np_softmax(l.asnumpy())
    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)
    out = exec1.outputs[0].asnumpy()
    assert_almost_equal(out, np_softmax(x.asnumpy()), rtol=1e-4)
    exec1.backward()
    assert_almost_equal(grad.asnumpy(), np_softmax(x.asnumpy()) - l.asnumpy(), rtol=1e-4)


def test_python_op():
    X = mx.symbol.Variable('X')
    op = mx.operator.NumpyOp()
    s = op.get_symbol(X, name='numpy_op')

    x = mx.ndarray.ones((10))*10
    dx = mx.ndarray.zeros((10))
    dy = mx.ndarray.ones((10))
    exec1 = s.bind(default_context(), args=[x], args_grad = {'X': dx})
    exec1.forward(is_train=True)
    assert_almost_equal(x.asnumpy(), exec1.outputs[0].asnumpy())
    exec1.backward(dy)
    assert_almost_equal(dy.asnumpy(), dx.asnumpy())


def test_swapaxes():
    data = mx.symbol.Variable('data')
    shape = (2, 3, 4)
    data_tmp = np.ones(shape)
    data_tmp[0] = 1
    data_tmp[1] = 2
    arr_data = mx.nd.array(data_tmp)
    swap0 = mx.symbol.SwapAxis(data=data, dim1=0, dim2=2)
    swap = mx.symbol.SwapAxis(data=swap0, dim1=1, dim2=2)
    exe_c = swap.bind(default_context(), args=[arr_data])
    exe_c.forward(is_train=True)
    out = exe_c.outputs[0].asnumpy()

    swap0_ = np.swapaxes(data_tmp, 0, 2)
    swap_ = np.swapaxes(swap0_, 1, 2)

    assert_almost_equal(out, swap_)


def test_scalarop():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)*5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = 2 / (4-((1+data+1)*2/5)-0.8-(data!=0))

    npout_1 = (4-((1+data_tmp+1)*2/5)-0.8-(data_tmp!=0))
    npout = 2/npout_1

    check_symbolic_forward(test, [data_tmp], [npout])

    npout_grad = 2.*2/5
    npout_grad = 2*npout_grad /(npout_1 *npout_1 )

    check_symbolic_backward(test, [data_tmp], [np.ones(shape)*2], [npout_grad])


def test_scalar_pow():
    data = mx.symbol.Variable('data')
    shape = (1, 1)
    data_tmp = np.ones(shape)
    test = data ** 2
    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [data_tmp ** 2])
    check_symbolic_backward(test, [data_tmp], [np.ones(shape)], [2 * data_tmp])


def test_symbol_pow():
    shape = (1, 1)

    data = mx.symbol.Variable('data')
    data_tmp = np.ones(shape)*2

    exp = mx.symbol.Variable('exp')
    exp_tmp = np.ones(shape)*3

    test = data**exp

    check_numeric_gradient(test, [data_tmp, exp_tmp])
    check_symbolic_forward(test, [data_tmp, exp_tmp], [data_tmp**exp_tmp])

    data_dir = data_tmp**(exp_tmp - 1) * exp_tmp
    exp_dir = data_tmp**(exp_tmp) * np.log(data_tmp)
    check_symbolic_backward(test, [data_tmp, exp_tmp], [np.ones(shape)], [data_dir, exp_dir])


def test_pow_fn():
    shape = (3, 4)
    exp = mx.symbol.Variable("exp")
    y = mx.sym.pow(2, exp)
    x = np.ones(shape)*3
    check_numeric_gradient(y, [x], numeric_eps=1E-3)
    check_symbolic_forward(y, [x], [2**x])
    check_symbolic_backward(y, [x], [np.ones(shape)], [np.log(2) * 2**x])


def test_relu():
    def frelu(x):
        return np.maximum(x, 0.0)
    def frelu_grad(x):
        return 1.0 * (x > 0.0)
    shape = (3, 4)
    x = mx.symbol.Variable("x")
    y = mx.sym.relu(x)
    xa = np.random.uniform(low=-1.0,high=1.0,size=shape)
    ya = frelu(xa)
    ga = frelu_grad(xa)
    check_numeric_gradient(y, [xa], numeric_eps=1E-3)
    check_symbolic_forward(y, [xa], [ya])
    check_symbolic_backward(y, [xa], [np.ones(shape)], [ga])


def test_sigmoid():
    def fsigmoid(a):
        return np.divide(1.0, (1.0 + np.exp(-a)))
    shape = (3, 4)
    x = mx.symbol.Variable("x")
    y = mx.sym.sigmoid(x)
    xa = np.random.uniform(low=-1.0,high=1.0,size=shape)
    ya = fsigmoid(xa)
    check_numeric_gradient(y, [xa], numeric_eps=1E-3)
    check_symbolic_forward(y, [xa], [ya])
    check_symbolic_backward(y, [xa], [np.ones(shape)], [ya * (1 - ya)])


def test_binary_logic():
    def _inner_test(forward_gt, logic_sym, x_shape, y_shape, test_scalar=True):
        x = mx.symbol.Variable("x")
        y = mx.symbol.Variable("y")
        z = logic_sym(x, y)
        x_npy = np.random.randint(0, 4, size=x_shape).astype(np.float32)
        y_npy = np.random.randint(0, 4, size=y_shape).astype(np.float32)
        exe = z.simple_bind(ctx=default_context(), x=x_shape, y=y_shape)
        mx_out = exe.forward(is_train=True, x=x_npy, y=y_npy)[0].asnumpy()
        assert_almost_equal(mx_out, forward_gt(x_npy, y_npy))
        exe.backward()
        if test_scalar:
            z_lscalar = logic_sym(1, y)
            z_rscalar = logic_sym(x, 1)
            exe_lscalar = z_lscalar.simple_bind(ctx=default_context(), y=y_shape)
            exe_rscalar = z_rscalar.simple_bind(ctx=default_context(), x=x_shape)
            mx_lscalar_out = exe_lscalar.forward(is_train=True, y=y_npy)[0].asnumpy()
            mx_rscalar_out = exe_rscalar.forward(is_train=True, x=x_npy)[0].asnumpy()
            assert_almost_equal(mx_lscalar_out, forward_gt(1, y_npy))
            assert_almost_equal(mx_rscalar_out, forward_gt(x_npy, 1))
            exe_lscalar.backward()
            exe_rscalar.backward()
    # Test the no-broadcasting binary logic ops + scalar logic ops
    _inner_test(forward_gt=lambda x, y: x == y,
                logic_sym=lambda x, y: x == y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x > y,
                logic_sym=lambda x, y: x > y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x >= y,
                logic_sym=lambda x, y: x >= y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x < y,
                logic_sym=lambda x, y: x < y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x <= y,
                logic_sym=lambda x, y: x <= y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x != y,
                logic_sym=lambda x, y: x != y, x_shape=(10, 10), y_shape=(10, 10))
    # Test the broadcasting binary logic ops
    _inner_test(forward_gt=lambda x, y: x == y,
                logic_sym=lambda x, y: mx.sym.broadcast_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x > y,
                logic_sym=lambda x, y: mx.sym.broadcast_greater(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x >= y,
                logic_sym=lambda x, y: mx.sym.broadcast_greater_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x < y,
                logic_sym=lambda x, y: mx.sym.broadcast_lesser(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x <= y,
                logic_sym=lambda x, y: mx.sym.broadcast_lesser_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x != y,
                logic_sym=lambda x, y: mx.sym.broadcast_not_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)


def test_embedding():
    in_dim = 10
    out_dim = 4
    batch = 24

    data = mx.sym.Variable("data")
    embed = mx.sym.Embedding(data=data, input_dim=in_dim, output_dim=out_dim, name="embed")
    exe_test = embed.simple_bind(default_context(), grad_req={'data': 'null', 'embed_weight': 'write'}, data=(batch,))
    arg_map = dict(zip(embed.list_arguments(), exe_test.arg_arrays))
    grad_map = dict(zip(embed.list_arguments(), exe_test.grad_arrays))
    np_data = np.random.randint(low=0, high=in_dim, size=batch)
    np_weight = np.random.uniform(-0.01, 0.01, arg_map["embed_weight"].shape)
    np_onehot = np.zeros((batch, in_dim))
    np_onehot[np.arange(batch), np_data] = 1.0
    # forward
    arg_map["data"][:] = np_data
    arg_map["embed_weight"][:] = np_weight
    exe_test.forward(is_train=True)
    assert_almost_equal(exe_test.outputs[0].asnumpy(), np.dot(np_onehot, np_weight))
    # backward
    np_grad = np.random.uniform(-1, 1, exe_test.outputs[0].shape)
    grad = mx.nd.zeros(np_grad.shape)
    grad[:] = np_grad
    exe_test.backward([grad])
    assert_almost_equal(grad_map["embed_weight"].asnumpy(), np.dot(np_onehot.T, np_grad))


# check ops handle duplicate input correctly.
def test_binary_op_duplicate_input():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:] = 5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:] = 3
    out_grad = mx.nd.empty(shape)
    out_grad[:] = 1
    square = data * data
    exe_square = square.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_square.forward(is_train=True)
    assert_almost_equal(exe_square.outputs[0].asnumpy(), data_tmp * data_tmp)
    exe_square.backward(out_grad)
    assert_almost_equal(arr_grad.asnumpy(), 2.0 * data_tmp)


def test_sign():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = mx.sym.sign(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout = np.sign(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = 0;
    exe_test.backward(out_grad)
    assert_almost_equal(arr_grad.asnumpy(), npout_grad)


def test_round_ceil_floor():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5.543
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]= 2

    test = mx.sym.round(data) + mx.sym.ceil(data) +  mx.sym.floor(data)
    exe_test = test.bind(default_context(), args=[arr_data])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout = np.round(data_tmp) + np.ceil(data_tmp) + np.floor(data_tmp)
    assert_almost_equal(out, npout)


def test_trunc():
    data_tmp = np.random.rand(3, 4) * 10 - 5
    arr_data = mx.nd.array(data_tmp)
    data = mx.symbol.Variable('data')
    test = mx.sym.trunc(data)

    exe_test = test.bind(default_context(), args=[arr_data])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout = np.trunc(data_tmp)

    assert_almost_equal(out, npout)


def test_rsqrt_cos_sin():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test =  mx.sym.rsqrt(data) + mx.sym.cos(data) + mx.sym.sin(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout =  1/ np.sqrt(data_tmp) + np.cos(data_tmp) + np.sin(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = npout_grad * -(1.0 / (2.0 * data_tmp * np.sqrt(data_tmp))) + npout_grad * -1 * np.sin(data_tmp) + npout_grad * np.cos(data_tmp)
    exe_test.backward(out_grad)
    assert_almost_equal(arr_grad.asnumpy(), npout_grad)


def test_maximum_minimum():
    data1 = mx.symbol.Variable('data')
    data2 = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp1 = np.random.rand(3,4)
    data_tmp2 = np.random.rand(3,4)
    data_tmp1[:] = 2
    data_tmp2[:] = 3

    arr_data1 = mx.nd.array(data_tmp1)
    arr_data2 = mx.nd.array(data_tmp2)

    arr_grad1 = mx.nd.empty(shape)
    arr_grad2 = mx.nd.empty(shape)

    test =  mx.sym.maximum(data1,data2) + mx.sym.minimum(data1,data2);
    exe_test = test.bind(default_context(), args=[arr_data1,arr_data2], args_grad=[arr_grad1,arr_grad2])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout =  np.maximum(data_tmp1,data_tmp2) + np.minimum(data_tmp1,data_tmp2)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2
    exe_test.backward(out_grad)

    npout_grad = np.ones(shape)
    npout_grad[:] = 2
    mask1 = (data_tmp1 > data_tmp2).astype('float')
    mask2 = (data_tmp1 < data_tmp2).astype('float')
    npout_grad1 = npout_grad * mask1 + npout_grad * mask2
    npout_grad2 = (npout_grad - npout_grad * mask1) + (npout_grad - npout_grad * mask2)

    assert_almost_equal(arr_grad1.asnumpy(), npout_grad1)
    assert_almost_equal(arr_grad2.asnumpy(), npout_grad2)


def test_maximum_minimum_scalar():
    data1 = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp1 = np.random.rand(3,4)
    data_tmp1[:] = 2

    arr_data1 = mx.nd.array(data_tmp1)
    arr_grad1 = mx.nd.empty(shape)

    test =  mx.sym.maximum(data1,3) + mx.sym.maximum(9,data1) + mx.sym.minimum(5,data1) + mx.sym.minimum(data1,4)
    exe_test = test.bind(default_context(), args=[arr_data1], args_grad=[arr_grad1])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout =  np.maximum(data_tmp1,3) + np.maximum(9,data_tmp1) + np.minimum(5,data_tmp1) + np.minimum(data_tmp1,4)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2
    exe_test.backward(out_grad)

    npout_grad = np.ones(shape)
    npout_grad[:] = 2
    mask1 = (data_tmp1 > 3).astype('float')
    mask2 = (9 > data_tmp1).astype('float')
    mask3 = (5 < data_tmp1).astype('float')
    mask4 = (data_tmp1 < 4).astype('float')
    npout_grad1 = npout_grad * mask1 + (npout_grad - npout_grad * mask2) + (npout_grad - npout_grad * mask3) + npout_grad * mask4

    assert_almost_equal(arr_grad1.asnumpy(), npout_grad1)


def test_abs():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = mx.sym.abs(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout = abs(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = npout_grad * np.sign(data_tmp)
    exe_test.backward(out_grad)
    assert_almost_equal(arr_grad.asnumpy(), npout_grad)


def check_deconvolution_forward_backward(input_shape, num_filter, kernel, stride, pad):
    """configure A: input --> conv --> deconv --> output.
       the convolution and deconvoluiton has similar parameter which ensure
       the input shape is the same as output, and the same weights between conv
       and deconv;
       If the input value of forward() and backwrad() is the same, then
       the output value of them should also the same;
    """
    assert input_shape[1] == num_filter
    data = mx.sym.Variable(name="data")
    conv = mx.sym.Convolution(
        data=data, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "conv")
    deconv = mx.sym.Deconvolution(
        data=conv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "deconv")

    arg_names = deconv.list_arguments()
    arg_shapes, out_shapes, _ = deconv.infer_shape(data=input_shape)
    input_data = mx.random.uniform(-5, 5, input_shape, ctx=mx.cpu()).copyto(default_context())
    out_grad = input_data
    args = {}
    args["data"] = input_data
    args['conv_weight'] = args['deconv_weight'] = mx.random.normal(0, 1,
        (num_filter, input_shape[1]) + kernel, ctx=mx.cpu()).copyto(default_context())
    args_grad = [mx.nd.empty(s) for s in arg_shapes]

    exe = deconv.bind(default_context(), args=args, args_grad=args_grad)
    exe.forward(is_train=True)
    out = exe.outputs[0].asnumpy()
    exe.backward(out_grad)
    assert_almost_equal(out, args_grad[0].asnumpy(), rtol=1E-3, atol=1e-3)

    args_grad_addto_npy = [np.random.normal(size=s) for s in arg_shapes]
    args_grad_addto = [mx.nd.array(ele) for ele in args_grad_addto_npy]
    exe = deconv.bind(default_context(), args=args, args_grad=args_grad_addto, grad_req="add")
    exe.forward(is_train=True)
    out = exe.outputs[0].asnumpy()
    exe.backward(out_grad)
    assert_almost_equal(out + args_grad_addto_npy[0], args_grad_addto[0].asnumpy(), rtol=1e-4, atol=1e-3)


def check_deconvolution_gradient(input_shape, num_filter, pad):
    """configure A: input --> conv --> output.
       configure B: input --> deconv --> output
       the convolution and deconvoluiton has similar parameter which ensure
       the input shape is the same as output;
       During backward(), if the input of A equals output of B, and the output
       of A equals input of B, then the grad of weight should be the same;
    """
    ndim = len(pad)
    stride = (1,) * ndim
    kernel = tuple(2 * np.array(pad) + 1)
    data_conv = mx.sym.Variable(name="data_conv")
    conv = mx.sym.Convolution(
        data=data_conv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "conv")
    data_deconv = mx.sym.Variable(name="data_deconv")
    deconv = mx.sym.Deconvolution(
        data=data_deconv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "deconv")

    conv_data = mx.random.uniform(-5, 5, input_shape, ctx=mx.cpu()).copyto(default_context())
    conv_args = {}
    conv_args["data_conv"] = conv_data
    conv_args['conv_weight'] = \
        mx.random.normal(0, 1,(num_filter, input_shape[1]) + kernel, ctx=mx.cpu()).copyto(default_context())
    conv_args_grad = [mx.nd.zeros(conv_data.shape),
        mx.nd.zeros((num_filter, input_shape[1]) + kernel)]
    exe_conv = conv.bind(default_context(), args=conv_args, args_grad=conv_args_grad)
    exe_conv.forward(is_train=True)
    conv_out_grad = mx.random.normal(0, 2, exe_conv.outputs[0].shape, ctx=mx.cpu()).copyto(default_context())
    exe_conv.backward(conv_out_grad)

    deconv_data = conv_out_grad
    deconv_args = {}
    deconv_args['data_deconv'] = deconv_data
    deconv_args['deconv_weight'] = conv_args['conv_weight']
    deconv_args_grad = [mx.nd.zeros(deconv_data.shape),
        mx.nd.zeros((num_filter, input_shape[1]) + kernel)]
    deconv_addto_args_grad_npy = [np.random.normal(size=deconv_data.shape),
                                  np.random.normal(size=(num_filter, input_shape[1]) + kernel)]
    deconv_addto_args_grad = [mx.nd.array(deconv_addto_args_grad_npy[0]),
                              mx.nd.array(deconv_addto_args_grad_npy[1])]
    exe_deconv = deconv.bind(default_context(), args=deconv_args, args_grad=deconv_args_grad)
    exe_deconv.forward(is_train=True)
    deconv_out_grad = conv_data[:]
    exe_deconv.backward(deconv_out_grad)
    assert_almost_equal(conv_args_grad[1].asnumpy(), deconv_args_grad[1].asnumpy(), rtol=1e-3, atol=1e-2)
    # Test AddTo
    exe_deconv_addto = deconv.bind(default_context(), args=deconv_args,
                                   args_grad=deconv_addto_args_grad,
                                   grad_req="add")
    exe_deconv_addto.forward(is_train=True)
    deconv_out_grad = conv_data[:]
    exe_deconv_addto.backward(deconv_out_grad)
    assert_almost_equal(conv_args_grad[1].asnumpy() + deconv_addto_args_grad_npy[1],
                        deconv_addto_args_grad[1].asnumpy(), rtol=1e-3, atol=1e-2)


def check_deconvolution_target_shape(input_shape, kernel, stride, pad, adj, target_shape=None):
    data = mx.sym.Variable(name="data")
    if target_shape:
        deconv = mx.sym.Deconvolution(
            data=data, kernel=kernel, stride=stride, pad=pad, adj=adj, num_filter=5,
            target_shape = target_shape)
    else:
        deconv = mx.sym.Deconvolution(
            data=data, kernel=kernel, stride=stride, pad=pad, adj=adj, num_filter=5)
    arg_names = deconv.list_arguments()
    arg_shapes, out_shapes, _ = deconv.infer_shape(data=input_shape)
    default_target_size = 8
    if target_shape is None:
        target_shape = (default_target_size,) * len(kernel)
    assert out_shapes[0] == (input_shape[0], 5) + target_shape


def test_deconvolution():
    # 2D
    check_deconvolution_target_shape(
        input_shape         = (2,3,4,4),
        kernel              = (3,3),
        stride              = (2,2),
        target_shape        = (8,8),
        pad                 = (99,99),  # will be ignored
        adj                 = (101,101),  # will be ignored
    )
    check_deconvolution_target_shape(
        input_shape         = (2,3,4,4),
        kernel              = (3,3),
        stride              = (2,2),
        pad                 = (1,1),
        adj                 = (1,1),
    )
    check_deconvolution_forward_backward(
        input_shape         = (1,1,5,5),
        num_filter          = 1,
        kernel              = (3,3),
        stride              = (1,1),
        pad                 = (1,1)
    )
    check_deconvolution_forward_backward(
        input_shape         = (32,3,28,28),
        num_filter          = 3,
        kernel              = (3,3),
        stride              = (1,1),
        pad                 = (1,1)
    )
    check_deconvolution_forward_backward(
        input_shape         = (10, 3, 403, 403),
        num_filter          = 3,
        kernel              = (7,7),
        stride              = (5,5),
        pad                 = (2,2)
    )
    check_deconvolution_gradient(
        input_shape = (1,3,5,5),
        num_filter = 3,
        pad = (1,1)
    )
    check_deconvolution_gradient(
        input_shape = (5,3,100,100),
        num_filter = 3,
        pad = (3,3)
    )
    # 1D
    check_deconvolution_target_shape(
        input_shape         = (2,3,4),
        kernel              = (3,),
        stride              = (2,),
        target_shape        = (8,),
        pad                 = (99,),  # will be ignored
        adj                 = (101,),  # will be ignored
    )
    check_deconvolution_target_shape(
        input_shape         = (2,3,4),
        kernel              = (3,),
        stride              = (2,),
        pad                 = (1,),
        adj                 = (1,),
    )
    check_deconvolution_forward_backward(
        input_shape         = (1,1,5),
        num_filter          = 1,
        kernel              = (3,),
        stride              = (1,),
        pad                 = (1,)
    )
    check_deconvolution_forward_backward(
        input_shape         = (32,3,28),
        num_filter          = 3,
        kernel              = (3,),
        stride              = (1,),
        pad                 = (1,)
    )
    check_deconvolution_forward_backward(
        input_shape         = (10, 3, 403),
        num_filter          = 3,
        kernel              = (7,),
        stride              = (5,),
        pad                 = (2,)
    )
    check_deconvolution_gradient(
        input_shape = (1,3,5),
        num_filter = 3,
        pad = (1,)
    )
    check_deconvolution_gradient(
        input_shape = (5,3,100),
        num_filter = 3,
        pad = (3,)
    )


def check_nearest_upsampling_with_shape(shapes, scale, root_scale):
    arr = {'arg_%d'%i: mx.random.uniform(-10.0, 10.0, shape, ctx=mx.cpu()).copyto(default_context()) for i, shape in zip(range(len(shapes)), shapes)}
    arr_grad = {'arg_%d'%i: mx.nd.zeros(shape) for i, shape in zip(range(len(shapes)), shapes)}

    up = mx.sym.UpSampling(*[mx.sym.Variable('arg_%d'%i) for i in range(len(shapes))], sample_type='nearest', scale=root_scale)
    exe = up.bind(default_context(), args=arr, args_grad=arr_grad)
    exe.forward(is_train=True)
    exe.backward(exe.outputs)
    for k in range(len(shapes)):
        name = 'arg_%d'%k
        assert_allclose(arr[name].asnumpy()*root_scale**2*scale**(2*k), arr_grad[name].asnumpy(), rtol=1e-4)


def check_bilinear_upsampling_with_shape(shapes, scale, root_scale):
    arr = {'arg_%d'%i: mx.random.uniform(-10.0, 10.0, shape, ctx=mx.cpu()).copyto(default_context()) for i, shape in zip(range(len(shapes)), shapes)}
    arr_grad = {'arg_%d'%i: mx.nd.zeros(shape) for i, shape in zip(range(len(shapes)), shapes)}

    up = mx.sym.UpSampling(*[mx.sym.Variable('arg_%d'%i) for i in range(len(shapes))], sample_type='bilinear', scale=root_scale)
    exe = up.bind(default_context(), args=arr, args_grad=arr_grad)
    exe.forward(is_train=True)
    exe.backward(exe.outputs)
    for k in range(len(shapes)):
        name = 'arg_%d'%k
        assert_allclose(arr[name].asnumpy()*root_scale**2*scale**(2*k), arr_grad[name].asnumpy(), rtol=1e-4)


def test_nearest_upsampling():
    for root_scale in [1,2,3]:
        for scale in [1,2,3]:
            for num_shape in [1,2,3]:
                for base in [1,2,3]:
                    shapes = [(1,3,base*root_scale*scale**(num_shape-1-i),base*root_scale*scale**(num_shape-1-i)) for i in range(num_shape)]
                    check_nearest_upsampling_with_shape(shapes, scale, root_scale)


@unittest.skip("test fails intermittently. temporarily disabled till it gets fixed. tracked at https://github.com/apache/incubator-mxnet/issues/8044")
def test_batchnorm_training():
    def check_batchnorm_training(stype):
        for shape in [(2, 3), (2, 3, 2, 2)]:
            data_tmp = np.random.normal(-0.1, 0.1, size=shape)
            s = shape[1],
            gamma = np.ones(s)
            beta = np.ones(s)
            gamma[1] = 3
            beta[0] = 3

            rolling_mean = np.random.uniform(size=s)
            rolling_std = np.random.uniform(size=s)

            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(gamma).tostype(stype),
                           mx.nd.array(beta).tostype(stype)]
            mean_std = [mx.nd.array(rolling_mean).tostype(stype), mx.nd.array(rolling_std).tostype(stype)]

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            test = mx.symbol.BatchNorm(data, fix_gamma=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=True, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            test = mx.symbol.BatchNorm(data, fix_gamma=True, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=False)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            test = mx.symbol.BatchNorm(data, fix_gamma=False)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=False, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            test = mx.symbol.BatchNorm(data, fix_gamma=False, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

            # Test varying channel axis
            dim = len(shape)
            for chaxis in range(-dim, dim):
                chaxis_true = chaxis
                if chaxis < 0:
                    chaxis_true = dim + chaxis

                shapex = shape

                channel_count = shapex[chaxis_true]
                data_tmp = np.random.normal(-0.1, 0.1, size=shapex)

                gamma = np.ones(channel_count)
                beta = np.ones(channel_count)
                if channel_count > 1:
                    gamma[1] = 3
                beta[0] = 3

                in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(gamma).tostype(stype),
                               mx.nd.array(beta).tostype(stype)]

                xrolling_mean = np.random.uniform(size=channel_count)
                xrolling_std = np.random.uniform(size=channel_count)
                xmean_std = [mx.nd.array(xrolling_mean).tostype(stype),
                             mx.nd.array(xrolling_std).tostype(stype)]

                test = mx.symbol.BatchNorm(data, fix_gamma=True, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

                test = mx.symbol.BatchNorm(data, fix_gamma=True, use_global_stats=True, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

                test = mx.symbol.BatchNorm(data, fix_gamma=False, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

                test = mx.symbol.BatchNorm(data, fix_gamma=False, use_global_stats=True, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_batchnorm_training(stype)


def test_convolution_grouping():
    for dim in [1, 2, 3]:
        num_filter = 4
        num_group = 2
        kernel = (3,) * dim
        shape = (1, 4) + (9,) * dim

        x = mx.sym.Variable('x')
        w = mx.sym.Variable('w')
        b = mx.sym.Variable('b')
        y1 = mx.sym.Convolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group, kernel=kernel)
        xslice = mx.sym.SliceChannel(data=x, num_outputs=num_group, axis=1)
        wslice = mx.sym.SliceChannel(data=w, num_outputs=num_group, axis=0)
        bslice = mx.sym.SliceChannel(data=b, num_outputs=num_group, axis=0)
        y2 = mx.sym.Concat(*[mx.sym.Convolution(data=xslice[i], weight=wslice[i], bias=bslice[i],
                                                num_filter=num_filter//num_group, kernel=kernel)
                           for i in range(num_group)])

        exe1 = y1.simple_bind(default_context(), x=shape)
        exe2 = y2.simple_bind(default_context(), x=shape, w=(num_filter, shape[1]//num_group) + kernel, b=(num_filter,))
        for arr1, arr2 in zip(exe1.arg_arrays, exe2.arg_arrays):
            arr1[:] = np.random.normal(size=arr1.shape)
            arr2[:] = arr1
        exe1.forward(is_train=True)
        exe1.backward(exe1.outputs[0])
        exe2.forward(is_train=True)
        exe2.backward(exe2.outputs[0])

        for arr1, arr2 in zip(exe1.outputs + exe1.grad_arrays, exe2.outputs + exe2.grad_arrays):
            np.testing.assert_allclose(arr1.asnumpy(), arr2.asnumpy(), rtol=1e-3, atol=1e-4)


@unittest.skip("test fails intermittently. temporarily disabled till it gets fixed. tracked at https://github.com/apache/incubator-mxnet/issues/8712")
def test_depthwise_convolution():
    for dim in [1,2]:
        for num_base in [1, 4, 16, 32, 64]:
            for kernel_x in [3, 5]:
                for stride_x in [1, 2]:
                    for pad_x in [0, 1]:
                        for in_size in [7, 32]:
                            kernel = (kernel_x,) * dim
                            stride = (stride_x,) * dim
                            pad = (pad_x,) * dim
                            num_filter = num_base
                            num_group = num_base
                            shape = (2, num_base) + (in_size,) * dim

                            x = mx.sym.Variable('x')
                            w = mx.sym.Variable('w')
                            b = mx.sym.Variable('b')
                            y1 = mx.sym.Convolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group,
                                    kernel=kernel, stride=stride, pad=pad)
                            xslice = mx.sym.SliceChannel(data=x, num_outputs=num_group, axis=1)
                            wslice = mx.sym.SliceChannel(data=w, num_outputs=num_group, axis=0)
                            bslice = mx.sym.SliceChannel(data=b, num_outputs=num_group, axis=0)
                            y2 = mx.sym.Concat(*[mx.sym.Convolution(data=xslice[i], weight=wslice[i], bias=bslice[i],
                                                                    num_filter=num_filter//num_group, kernel=kernel,
                                                                    stride=stride, pad=pad)
                                                for i in range(num_group)])

                            dev = default_context()
                            exe1 = y1.simple_bind(dev, x=shape)
                            exe2 = y2.simple_bind(mx.cpu(), x=shape, w=(num_filter, shape[1]//num_group)+kernel,
                                    b=(num_filter,))
                            for arr1, arr2 in zip(exe1.arg_arrays, exe2.arg_arrays):
                                arr1[:] = np.random.normal(size=arr1.shape)
                                arr2[:] = arr1
                            exe1.forward(is_train=True)
                            exe1.backward(exe1.outputs[0])
                            exe2.forward(is_train=True)
                            exe2.backward(exe2.outputs[0])

                            for arr1, arr2 in zip(exe1.outputs + exe1.grad_arrays, exe2.outputs + exe2.grad_arrays):
                                np.testing.assert_allclose(arr1.asnumpy(), arr2.asnumpy(), rtol=1e-3, atol=1e-3)


def gen_broadcast_data(idx):
    # Manually set test cases
    binary_op_data_shape = np.array(
        [[[2, 5, 1, 30, 7], [1, 5, 448, 30, 1]],
        [[10, 49, 1, 77, 17], [10, 1, 2, 1, 17]],
        [[13, 2, 65, 2,  1], [13, 1, 65, 1, 225]],
        [[9, 434, 4, 2, 37], [9, 1, 4, 1, 37]],
        [[2, 52, 1, 4, 1], [1, 52, 60, 1, 37]],
        [[1, 23, 7, 122, 50], [2, 1, 7, 1, 50]],
        [[1, 17, 1, 5, 1], [22, 1, 2, 1, 28]],
        [[29, 1, 2, 1, 8], [29, 22, 1, 130, 1]],
        [[2, 36, 1, 427, 3], [1, 36, 11, 427, 1]],
        [[1, 2, 1, 100, 7], [1, 2, 448, 100, 1]],
        [[1, 2, 495, 77, 7], [1, 2, 1, 1, 7]],
        [[1, 43, 65, 2, 1], [1, 43, 65, 1, 225]],
        [[1, 92, 434, 2, 2], [1, 92, 1, 2, 2]],
        [[1, 92, 1, 4, 1], [1, 92, 134, 1, 17]],
        [[1, 53, 2, 122, 143], [1, 1, 2, 1, 143]],
        [[1, 179, 1, 87, 17], [1, 179, 1, 1, 17]],
        [[1, 1, 17, 5, 1], [1, 22, 1, 1, 28]],
        [[1, 2, 1, 1, 8], [1, 2, 52, 430, 1]],
        [[1, 163, 1, 22, 3], [1, 163, 116, 22, 1]],
        [[1, 1, 44, 30, 7], [1, 1, 44, 30, 1]],
        [[1, 1, 1, 1, 28], [1, 127, 1, 5, 28]],
        [[1, 2, 394, 38, 1], [1, 2, 394, 38, 16]],
        [[1, 10, 49, 77, 17], [1, 1, 1, 1, 17]],
        [[1, 431, 6, 2, 225], [1, 1, 6, 2, 225]],
        [[1, 15, 1, 28, 1], [1, 15, 1, 28, 463]],
        [[1, 129, 2, 48, 96], [1, 129, 2, 1, 1]],
        [[1, 1, 403, 17, 2], [1, 44, 403, 17, 2]],
        [[1, 1, 65, 2, 22], [1, 1, 65, 1, 1]],
        [[1, 24, 103, 17, 18], [1, 24, 1, 1, 1]],
        [[1, 1, 1, 1, 2], [1, 24, 194, 50, 1]],
        [[1, 1, 107, 84, 9], [1, 1, 1, 1, 1]]])
    if idx < binary_op_data_shape.shape[0]:
        l_shape = binary_op_data_shape[idx][0]
        r_shape = binary_op_data_shape[idx][1]
    else:
        # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
        ndim = np.random.randint(1, 6)
        shape = np.random.randint(1, 6, size=(ndim,))
        l_same_dim = np.random.randint(0, 5)
        r_same_dim = np.random.randint(0, 5)
        l_axis_flags = np.random.randint(0, 2, size=ndim)
        r_axis_flags = np.random.randint(0, 2, size=ndim)
        if l_same_dim == 4:
            l_axis_flags = np.ones(ndim)
        if r_same_dim == 4:
            r_axis_flags = np.ones(ndim)
        l_shape = shape.copy()
        r_shape = shape.copy()
        l_shape[np.where(l_axis_flags == 0)] = 1
        r_shape[np.where(r_axis_flags == 0)] = 1
    return [np.random.random(l_shape), np.random.random(r_shape)]


def gen_broadcast_data_int(idx):
    d = gen_broadcast_data(idx);
    return [np.round(d[0]*100).astype(int), np.round(d[1]*100).astype(int)]


def gen_binary_data(dummy):
    ndim = np.random.randint(1, 6)
    shape = np.random.randint(1, 6, size=(ndim,))
    return [np.random.random(shape), np.random.random(shape)]


def gen_binary_data_int(dummy):
    d = gen_binary_data(dummy);
    return [np.round(d[0]*100).astype(int), np.round(d[1]*100).astype(int)]


def check_binary_op_forward(symbol, baseline, gen_data, rtol=1e-3, atol=1e-5, mx_nd_func=None):
    sample_num = 200
    for i in range(sample_num):
        d = gen_data(i)
        x = baseline(d[0], d[1])
        y = symbol.bind(default_context(), args={'a': mx.nd.array(d[0]), 'b': mx.nd.array(d[1])})
        y.forward(is_train=True)
        y = y.outputs[0].asnumpy()
        if mx_nd_func is not None:
            d0 = mx.nd.array(d[0], dtype=d[0].dtype)
            d1 = mx.nd.array(d[1], dtype=d[1].dtype)
            assert_almost_equal(y, mx_nd_func(d0, d1).asnumpy(), rtol=rtol, atol=atol)
        idx = np.abs(x-y) > atol+rtol*np.abs(x)
        if idx.any():
            print('found precision problem')
            d[0] = np.broadcast_to(d[0], x.shape)
            d[1] = np.broadcast_to(d[1], x.shape)
            print('a: {}'.format(d[0][idx]))
            print('b: {}'.format(d[1][idx]))
            import struct
            print('a hex: {}'.format(struct.pack('d', d[0][idx]).encode('hex')))
            print('b hex: {}'.format(struct.pack('d', np.broadcast_to(d[1], x.shape)[idx]).encode('hex')))
            print('in baseline(a, b): {}'.format(x[idx]))
            print('in symbol(a, b): {}'.format(y[idx]))
            print('diff: {}'.format(np.abs(x-y)[idx] - atol-rtol*np.abs(x)[idx]))
        assert_allclose(y, x, rtol=rtol, atol=atol)


def check_binary_op_backward(symbol, baseline, gen_data, rtol=1e-3, atol=1e-5):
    sample_num = 200
    for i in range(sample_num):
        d = gen_data(i)
        out = np.random.random((d[0] + d[1]).shape)

        def reduce_op(shape, x):
            if shape == x.shape:
                return x
            keepdims_shape = list(x.shape)
            for i in range(len(shape)):
                if x.shape[i] != shape[i]:
                    keepdims_shape[i] = 1
                    x = np.sum(x, axis=i).reshape(keepdims_shape)
            return x

        baseline_grad1, baseline_grad2 = baseline(out, d[0], d[1])
        x_1 = reduce_op(d[0].shape, baseline_grad1)
        x_2 = reduce_op(d[1].shape, baseline_grad2)
        y_1 = mx.nd.empty(d[0].shape)
        y_2 = mx.nd.empty(d[1].shape)
        y = symbol.bind(default_context(), args={'a': mx.nd.array(d[0]), 'b': mx.nd.array(d[1])},
                        args_grad=[y_1, y_2])
        y.forward(is_train=True)
        y.backward([mx.nd.array(out)])
        assert_allclose(y_1.asnumpy(), x_1, rtol=rtol, atol=atol)
        assert_allclose(y_2.asnumpy(), x_2, rtol=rtol, atol=atol)


def test_binary_op():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')

    def test_bplus(a, b):
        c = a + b
        check_binary_op_forward(c, lambda a, b: a + b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, g_out), gen_binary_data)

    def test_bminus(a, b):
        c = a - b
        check_binary_op_forward(c, lambda a, b: a - b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, - g_out), gen_binary_data)

    def test_bmul(a, b):
        c = a * b
        check_binary_op_forward(c, lambda a, b: a * b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * b, g_out * a), gen_binary_data)

    def test_bdiv(a, b):
        c = a / b
        check_binary_op_forward(c, lambda a, b: a / b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out / b, - g_out * a / (b * b)), gen_binary_data)

    def test_bmod(a, b):
        c = a % b
        check_binary_op_forward(c, lambda a, b: a % b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, - g_out * (a // b)), gen_binary_data)

    def test_bmod_int(a, b):
        c = mx.sym.cast(a, dtype='int32') % mx.sym.cast(b, dtype='int32')
        check_binary_op_forward(c, lambda a, b: a % b, gen_binary_data_int)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_binary_data_int)

    def test_bpow(a, b):
        c = a ** b
        check_binary_op_forward(c, lambda a, b: a ** b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * a **(b - 1) * b,
                                        g_out * a ** b * np.log(a)), gen_binary_data)

    def test_bneq(a, b):
        c = a != b
        check_binary_op_forward(c, lambda a, b: (a != b).astype(a.dtype), gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_binary_data)

    test_bplus(a, b)
    test_bminus(a, b)
    test_bmul(a, b)
    test_bdiv(a, b)
    test_bmod(a, b)
    test_bmod_int(a, b)
    test_bpow(a, b)
    test_bneq(a, b)


def test_broadcast_binary_op():
    def check_bmaxmin_gradient(test_sym, x, y, delta, rtol, atol):
        """This function ensures that checking the numerical gradient of
        broadcast_max/min is not crossing the boundary y=x where there
        is no gradient definition at those sigularities."""
        x_max = np.max(x)
        y = x_max + 2 * delta + np.random.random(y.shape)
        check_numeric_gradient(test_sym, [x, y], numeric_eps=delta, rtol=rtol, atol=atol)

        x_min = np.min(x)
        y = x_min - 2 * delta - np.random.random(y.shape)
        check_numeric_gradient(test_sym, [x, y], numeric_eps=delta, rtol=rtol, atol=atol)

    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')

    def test_bplus(a, b):
        c = mx.sym.broadcast_plus(a, b)
        check_binary_op_forward(c, lambda a, b: a + b, gen_broadcast_data, mx_nd_func=mx.nd.add)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, g_out), gen_broadcast_data)

    def test_bminus(a, b):
        c = mx.sym.broadcast_minus(a, b)
        check_binary_op_forward(c, lambda a, b: a - b, gen_broadcast_data, mx_nd_func=mx.nd.subtract)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, - g_out), gen_broadcast_data)

    def test_bmul(a, b):
        c = mx.sym.broadcast_mul(a, b)
        check_binary_op_forward(c, lambda a, b: a * b, gen_broadcast_data, mx_nd_func=mx.nd.multiply)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * b, g_out * a), gen_broadcast_data)

    def test_bdiv(a, b):
        c = mx.sym.broadcast_div(a, b)
        check_binary_op_forward(c, lambda a, b: a / b, gen_broadcast_data, mx_nd_func=mx.nd.divide)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out / b, - g_out * a / (b * b)), gen_broadcast_data)

    def test_bmod(a, b):
        c = mx.sym.broadcast_mod(a, b)
        check_binary_op_forward(c, lambda a, b: a % b, gen_broadcast_data, atol=1, mx_nd_func=mx.nd.modulo)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, - g_out * (a // b)), gen_broadcast_data, atol=1)

    def test_bmod_int(a, b):
        c = mx.sym.broadcast_mod(mx.sym.cast(a, dtype='int32'), mx.sym.cast(b, dtype='int32'))
        check_binary_op_forward(c, lambda a, b: a % b, gen_broadcast_data_int, mx_nd_func=mx.nd.modulo)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_broadcast_data_int)

    def test_bpow(a, b):
        c = mx.sym.broadcast_power(a, b)
        check_binary_op_forward(c, lambda a, b: a ** b, gen_broadcast_data, mx_nd_func=mx.nd.power)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * a **(b - 1) * b,
                                                         g_out * a ** b * np.log(a)), gen_broadcast_data)

    def test_bequal(a, b):
        c = mx.sym.broadcast_equal(a, b)
        check_binary_op_forward(c, lambda a, b: (a == b).astype(a.dtype), gen_broadcast_data_int,
                                mx_nd_func=mx.nd.equal)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_broadcast_data_int)

    def test_bmax(a, b):
        c = mx.sym.broadcast_maximum(a, b)
        check_binary_op_forward(c, lambda x, y: np.maximum(x, y), gen_broadcast_data, mx_nd_func=mx.nd.maximum)
        # pass idx=200 to gen_broadcast_data so that generated ndarrays' sizes are not too big
        data = gen_broadcast_data(idx=200)
        check_bmaxmin_gradient(c, data[0], data[1], 0.001, 1e-2, 1e-3)

    def test_bmin(a, b):
        c = mx.sym.broadcast_minimum(a, b)
        check_binary_op_forward(c, lambda x, y: np.minimum(x, y), gen_broadcast_data, mx_nd_func=mx.nd.minimum)
        # pass idx=200 to gen_broadcast_data so that generated ndarrays' sizes are not too big
        data = gen_broadcast_data(idx=200)
        check_bmaxmin_gradient(c, data[0], data[1], 0.001, 1e-2, 1e-3)

    test_bplus(a, b)
    test_bminus(a, b)
    test_bmul(a, b)
    test_bdiv(a, b)
    test_bmod(a, b)
    test_bmod_int(a, b)
    test_bpow(a, b)
    test_bequal(a, b)
    test_bmax(a, b)
    test_bmin(a, b)


def test_run_convolution_dilated_impulse_response(dil=(1,1), kernel_shape=(3,3), verbose=False):
    dim = len(dil)
    assert(len(kernel_shape) == dim)
    # Input for spike response
    data_size = 33
    data_shape = (1, 1) + (data_size,) * dim
    center = (0,0) + (data_size // 2,) * dim
    spike_imgs = np.zeros(shape=data_shape, dtype=np.float32)
    spike_imgs[center] = 1.0
    spike_img = mx.nd.array(spike_imgs)
    spike_img2 = mx.nd.array(spike_imgs)

    kernel_weights = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)
    kernel_weights2 = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)

    kernel = mx.symbol.Variable('kernel')
    in_img = mx.symbol.Variable('input')
    net = mx.symbol.Convolution(in_img, num_filter=1,kernel=kernel_shape, dilate=dil, no_bias="true", name='test_convolution')
    net.list_arguments()
    be = net.bind(default_context(), args={ 'input' : spike_img, 'test_convolution_weight' : kernel_weights},
                args_grad={'input' : spike_img2, 'test_convolution_weight' : kernel_weights2 } )
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    ndo = be.outputs[0]

    out_grads = np.zeros(shape=be.outputs[0].shape, dtype=np.float32)
    out_grads[center] = 1.0
    out_grad = mx.nd.array(out_grads)
    be.backward([out_grad])
    vgrad = be.grad_arrays[0].asnumpy()
    out = out_o.reshape(out_o.shape[2:])
    nz_loc = np.nonzero(out)
    assert_allclose(np.sum(out),np.prod(kernel_shape),atol=1e-5)
    assert_allclose(np.sum(vgrad),np.prod(kernel_shape),atol=1e-5)

    # Now check whether the input gradient was computed correctly
    input_grad = mx.nd.array(vgrad)

    be = net.bind(default_context(), args={ 'input' : input_grad, 'test_convolution_weight' : kernel_weights})
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    assert_allclose(out_o[center],np.prod(kernel_shape),atol=1e-5)

    rnd_kernel_s = np.random.uniform(low=0.0, high=1.0, size=tuple([1,1]+list(kernel_shape))).astype(np.float32)
    impulse_error = mx.nd.array(out_o/np.sum(out_o)) # This should be 1.0 at [0,0,16,16]
    rnd_kernel = mx.nd.array(rnd_kernel_s)

    rnd_kernel2 = mx.nd.array(rnd_kernel_s)
    white_in = mx.nd.ones(shape=data_shape)
    white_in2 = mx.nd.ones(shape=data_shape)

    be = net.bind(default_context(), args={ 'input' : white_in, 'test_convolution_weight' : rnd_kernel},
                args_grad={'input' : white_in2, 'test_convolution_weight' : rnd_kernel2 } )

    be.forward(True)
    be.backward([impulse_error])
    out_orig = be.outputs[0].asnumpy()
    kernel_gradient = be.grad_arrays[1].asnumpy()

    dkernel = mx.nd.array(rnd_kernel_s + kernel_gradient)

    be = net.bind(default_context(), args={ 'input' : white_in, 'test_convolution_weight' : dkernel})

    be.forward(True)
    out = be.outputs[0].asnumpy()
    # Now do a simple check of the kernel gradient
    assert(out[center] - np.sum(kernel_gradient) - out_orig[center] < 0.001)


def test_convolution_dilated_impulse_response():
    # 1D
    for dil in [ (1,), (2,), (3,) ]:
        for ks in [ (1,), (2,), (3,), (4,)]:
            test_run_convolution_dilated_impulse_response(dil=dil, kernel_shape=ks)
    # 2D
    for dil in [ (1,1), (2,2), (3,3) ]:
        for ks in [ (3,3), (4,4), (2,3), (3,2), (1,1) ]:
            test_run_convolution_dilated_impulse_response(dil=dil, kernel_shape=ks)


def test_reshape():

    def test_reshape_new(src_shape, shape_args, reverse, dst_shape):
        net = mx.sym.Variable("data")
        net = mx.sym.Reshape(net, shape=shape_args, reverse=reverse)
        js = net.tojson()
        net = mx.sym.load_json(js)
        _, output_shape, __ = net.infer_shape(data=src_shape)
        assert output_shape[0] == dst_shape, \
            'Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s, ' \
            'Output Shape = %s' %(str(src_shape), str(shape_args), str(reverse),
                                  str(dst_shape), str(output_shape[0]))
        dat_npy = np.random.rand(*src_shape)
        grad_npy = np.random.rand(*dst_shape)
        exe = net.simple_bind(default_context(), data=src_shape)
        exe.arg_dict['data'][:] = dat_npy
        exe.forward(is_train=True)
        assert np.square(exe.outputs[0].asnumpy() - dat_npy.reshape(dst_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s'\
            %(str(src_shape), str(shape_args), str(reverse), str(dst_shape))
        exe.backward(out_grads=mx.nd.array(grad_npy))
        assert np.square(exe.grad_dict['data'].asnumpy() - grad_npy.reshape(src_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s'\
            %(str(src_shape), str(shape_args), str(reverse), str(dst_shape))
    # Test new api (Using shape)
    test_cases = [
        [(2, 3, 5, 5),  (0, -1),          False, (2, 75)],
        [(2, 3, 5, 5),  (0, 0, -1),       False, (2, 3, 25)],
        [(5, 3, 4, 5),  (0, -1, 0),       False, (5, 15, 4)],
        [(2, 3, 5, 4),  (-1, 0, 0),       False, (8, 3, 5)],
        [(2, 3, 5, 5),  (0, 0, 0, 0),     False, (2, 3, 5, 5)],
        [(2, 4, 5, 3),  (-1, 2, 2, 1),    False, (30, 2, 2, 1)],
        [(2, 3, 5, 6),  (-2,),            False, (2, 3, 5, 6)],
        [(2, 3, 5, 6),  (6, 1, -2),       False, (6, 1, 5, 6)],
        [(2, 3, 5, 6),  (-3, -3),         False, (6, 30)],
        [(2, 3, 5, 6),  (-3, -1),         False, (6, 30)],
        [(64,),         (-4, 16, 4),      False, (16, 4)],
        [(64,),         (-4, 16, -1),     False, (16, 4)],
        [(64, 1, 2, 3), (-4, 16, -1, -2), False, (16, 4, 1, 2, 3)],
        [(2, 3, 5, 5),  (0, -1),          True,  (5, 30)],
        [(2, 3, 5, 5),  (0, 0, -1),       True,  (3, 5, 10)],
        [(5, 3, 4, 5),  (0, -1, 0),       True,  (3, 20, 5)],
        [(2, 3, 5, 4),  (-1, 0, 0),       True,  (6, 5, 4)],
        [(2, 3, 4, 5),  (3, -1, 0),       True,  (3, 8, 5)],
        [(2, 3, 5, 5),  (5, 3, 0, -1),    True,  (5, 3, 5, 2)],
        [(2, 3, 5, 5),  (0, 0, 0, 0),     True,  (2, 3, 5, 5)],
        [(2, 3, 5, 6),  (-2,),            True,  (2, 3, 5, 6)],
        [(2, 3, 5, 6),  (-2, 1, 30),      True,  (2, 3, 1, 30)],
        [(2, 3, 5, 6),  (-3, -3),         True,  (6, 30)],
        [(64,),         (16, 4, -4),      True,  (16, 4)],
        [(64,),         (16, -1, -4),     True,  (16, 4)],
        [(1, 2, 3, 64), (-2, -1, 16, -4), True,  (1, 2, 3, 4, 16)]]
    for test_case in test_cases:
        test_reshape_new(*test_case)
    # Test old api
    net = mx.sym.Variable("data")
    net = mx.sym.Reshape(net, target_shape=(2, 0))
    js = net.tojson()
    net = mx.sym.load_json(js)
    _, output_shape, __ = net.infer_shape(data=(2, 3, 5, 5))
    assert(output_shape[0] == (2, 75))
    # Test for Flatten
    data = mx.sym.Variable("data")
    net = mx.sym.Flatten(data)
    exe = net.simple_bind(ctx=default_context(), data=(5, 4, 3, 7))
    data_npy = np.random.normal(size=(5, 4, 3, 7))
    out_grad_npy = np.random.normal(size=(5, 4 * 3 * 7))
    outputs = exe.forward(is_train=True, data=data_npy)[0].asnumpy()
    assert_allclose(outputs, data_npy.reshape((5, 4 * 3 * 7)))
    exe.backward(out_grads=[mx.nd.array(out_grad_npy, ctx=default_context())])
    assert_allclose(exe.grad_arrays[0].asnumpy(), out_grad_npy.reshape((5, 4, 3, 7)))


def test_reduce():
    sample_num = 500
    def test_reduce_inner(numpy_reduce_func, numpy_reduce_grad_func, mx_reduce_sym, nan_prob = 0):
        for i in range(sample_num):
            # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
            # Insert a NaN with probability equal to nan_prob
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 6, size=(ndim,))
            axis_num = np.random.randint(0, ndim, size=1)
            axis_flags = np.random.randint(0, 2, size=ndim)
            exclude = np.random.randint(0, 2)
            axes = []
            for (axis, flag) in enumerate(axis_flags):
                if flag:
                    axes.append(axis)
            if 0 == len(axes):
                axes = None
            elif 1 == len(axes):
                axes = axes[0]
            else:
                axes = tuple(axes)
            keepdims = np.random.randint(0, 2)
            a = mx.symbol.Variable('a')
            if axes is None:
                b = mx_reduce_sym(a, keepdims=keepdims)
            elif exclude and isinstance(axes, tuple) and len(axes) < ndim:
                naxes = [i for i in range(ndim) if i not in axes]
                b = mx_reduce_sym(a, axis=naxes, keepdims=keepdims, exclude=True)
            else:
                b = mx_reduce_sym(a, axis=axes, keepdims=keepdims)
            dat_npy = np.random.rand(*shape)
            if nan_prob > 0:
                dat_npy[np.random.rand(*shape) < nan_prob] = np.nan
            sum_groundtruth = np.array(numpy_reduce_func(dat_npy, axis=axes, keepdims=keepdims))
            if sum_groundtruth.shape == ():
                sum_groundtruth = np.array([sum_groundtruth])
            grad_nd = mx.nd.empty(shape)
            outgrad_npy = np.array(np.random.rand(*sum_groundtruth.shape))

            keepdim_shape = np_reduce(dat_npy, axes, 1, np.sum).shape
            grad_groundtruth = numpy_reduce_grad_func(outgrad=outgrad_npy, data=dat_npy,
                                                      outdata=sum_groundtruth,
                                                      axis=axes, keepdims=keepdims,
                                                      keepdim_shape=keepdim_shape)
            net = b.bind(default_context(), args={'a': mx.nd.array(dat_npy)},
                         args_grad={'a': grad_nd})
            net.forward(is_train=True)

            equal_forward = almost_equal_ignore_nan(net.outputs[0].asnumpy(), sum_groundtruth, 1E-4, 1E-4)
            assert equal_forward

            net.backward(out_grads=mx.nd.array(outgrad_npy))
            bc_grad_groundtruth = np.broadcast_to(grad_groundtruth, grad_nd.shape)
            equal_backward = almost_equal_ignore_nan(grad_nd.asnumpy(), bc_grad_groundtruth, 1E-4, 1E-4)
            assert equal_backward

    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.sum),
                      lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                        outgrad.reshape(keepdim_shape),
                      mx.symbol.sum)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.mean),
                      lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                        outgrad.reshape(keepdim_shape)/(data.size/outdata.size),
                      mx.symbol.mean)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.prod),
                      lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                        outgrad.reshape(keepdim_shape) * (outdata.reshape(keepdim_shape) / data),
                      mx.symbol.prod)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.nansum),
                      lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                        np.where(np.isnan(data), 0, outgrad.reshape(keepdim_shape)),
                      mx.symbol.nansum, 0.3)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.nanprod),
                      lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                        np.where(np.isnan(data), 0, outgrad.reshape(keepdim_shape) * (outdata.reshape(keepdim_shape) / data)),
                      mx.symbol.nanprod, 0.3)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.max),
                      lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                        outgrad.reshape(keepdim_shape) * (np.equal(data, outdata.reshape(keepdim_shape)).astype(np.float)),
                      mx.symbol.max)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.min),
                      lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                        outgrad.reshape(keepdim_shape) * (np.equal(data, outdata.reshape(keepdim_shape)).astype(np.float)),
                      mx.symbol.min)


def test_broadcast():
    sample_num = 200
    for i in range(sample_num):
        # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
        ndim = np.random.randint(1, 6)
        target_shape = np.random.randint(1, 6, size=(ndim,))
        axis = tuple(set(np.random.randint(0, ndim, np.random.randint(1, ndim + 1))))
        shape = target_shape.copy()
        size = tuple([shape[ele] for ele in axis])
        for ele in axis:
            shape[ele] = 1
        a = mx.symbol.Variable('a')
        sym_bcast_axis = mx.symbol.broadcast_axis(a, axis=axis, size=size)
        sym_bcast_to = mx.symbol.broadcast_to(a, shape=tuple(target_shape))
        def test_broadcasting_ele(sym_bcast):
            dat_npy = np.random.rand(*shape)
            groundtruth = dat_npy
            grad_nd = mx.nd.empty(shape)
            outgrad_npy = np.random.rand(*target_shape)
            grad_groundtruth = np_reduce(outgrad_npy, axis=axis, keepdims=True,
                                          numpy_reduce_func=np.sum)
            net = sym_bcast.bind(default_context(), args={'a': mx.nd.array(dat_npy)},
                                                 args_grad={'a': grad_nd})
            net.forward(is_train=True)
            assert (net.outputs[0].shape == target_shape).all()
            assert_almost_equal(net.outputs[0].asnumpy(), groundtruth, rtol=1e-4)
            net.backward(out_grads=mx.nd.array(outgrad_npy))
            assert_almost_equal(grad_nd.asnumpy(), grad_groundtruth, rtol=1e-4)
        test_broadcasting_ele(sym_bcast_axis)
        test_broadcasting_ele(sym_bcast_to)


def test_transpose():
    for ndim in range(1, 7):
        for t in range(5):
            dims = list(np.random.randint(1, 10, size=ndim))
            axes = list(range(ndim))
            random.shuffle(axes)
            axes = tuple(axes)
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.transpose(x, axes=axes)
            assert_allclose(np.transpose(x.asnumpy(), axes=axes), y.asnumpy())

            y = mx.nd.transpose(x)
            assert_allclose(np.transpose(x.asnumpy()), y.asnumpy())


def test_expand_dims():
    for ndim in range(1, 6):
        for axis in range(-ndim + 1, ndim):
            x = np.random.normal(size=list(np.random.randint(1, 10, size=ndim)))
            y = mx.nd.array(x)
            x1 = np.expand_dims(x, axis=axis)
            y1 = mx.nd.expand_dims(y, axis=axis)
            assert_allclose(x1, y1.asnumpy())
            assert_allclose(x1.shape, y1.shape)


def test_crop():
    for ndim in range(1, 6):
        for t in range(5):
            dims = []
            begin = []
            end = []
            idx = []
            for i in range(ndim):
                d = random.randint(1, 5)
                b = random.randint(0, d-1)
                e = random.randint(b+1, d)
                if b == 0 and random.randint(0, 1):
                    b = None
                elif b != 0 and random.randint(0, 1):
                    b -= d
                if e == d and random.randint(0, 1):
                    e = None
                elif e != d and random.randint(0, 1):
                    e -= d
                dims.append(d)
                begin.append(b)
                end.append(e)
                idx.append(slice(b, e))
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.crop(x, begin=tuple(begin), end=tuple(end))
            assert_allclose(x.asnumpy()[idx], y.asnumpy())

            vx = mx.sym.Variable('x')
            vy = mx.sym.crop(vx, begin=tuple(begin), end=tuple(end))
            check_numeric_gradient(vy, [x.asnumpy()])


def test_slice_axis():
    for ndim in range(1, 6):
        shape = np.random.randint(1, 11, size=(ndim,))
        for t in range(ndim):
            d = shape[t]
            b = random.randint(0, d-1)
            e = random.randint(b+1, d)
            if np.random.rand() > 0.6:
                e = None
            else:
                if e < d and np.random.rand() > 0.5:
                    e = e - d
            if np.random.rand() > 0.5:
                b = b - d
            idx = []
            for i in range(ndim):
                idx.append(slice(0, shape[i]))
            idx[t] = slice(b, e)

            X = mx.symbol.Variable('X')
            x = mx.nd.array(np.random.normal(size=shape))
            Y = mx.symbol.slice_axis(data=X, axis=t, begin=b, end=e)

            xgrad = mx.nd.empty(x.shape)
            exec1 = Y.bind(default_context(), args = [x], args_grad = {'X': xgrad})
            exec1.forward(is_train=True)
            y = exec1.outputs[0]
            assert_allclose(x.asnumpy()[idx], y.asnumpy())
            exec1.backward([y])
            xx = x.asnumpy()
            xx[:] = 0.0
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(xx, xgrad.asnumpy())
            x_grad_npy = np.random.normal(size=x.shape)
            xgrad = mx.nd.array(x_grad_npy)
            exec2 = Y.bind(default_context(), args=[x], args_grad={'X': xgrad}, grad_req="add")
            exec2.forward(is_train=True)
            exec2.backward([exec2.outputs[0]])
            xx = np.zeros(shape=x.shape, dtype=np.float32)
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(xx + x_grad_npy, xgrad.asnumpy(), atol=1E-5)


def test_flip():
    for ndim in range(1, 6):
        for t in range(5):
            dims = [random.randint(1,10) for i in range(ndim)]
            axis = random.randint(0, ndim-1)
            idx = [slice(None, None, -1) if i == axis else slice(None, None) for i in range(ndim)]
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.flip(x, axis=axis)
            assert_allclose(x.asnumpy()[idx], y.asnumpy())


def test_stn():
    np.set_printoptions(threshold=np.nan)
    num_filter = 2  # conv of loc net
    kernel = (3, 3)  # conv of loc net
    num_hidden = 6  # fc of loc net
    for n in [1, 2, 3, 4]:
        for c in [1, 2, 3, 4]:
            for h in [5, 9, 13, 17]:  # for convenience test, this third and forth input dim should be 4x + 1
                for w in [5, 9, 13, 17]:
                    data_shape = (n, c, h, w)
                    target_shape = (int((data_shape[2]+1)/2), int((data_shape[3]+1)/2))
                    data = mx.sym.Variable(name="data")
                    loc = mx.sym.Convolution(data=data, kernel=kernel, pad=(1, 1), num_filter=num_filter, name="loc_conv")
                    loc = mx.sym.Flatten(data=loc)
                    loc = mx.sym.FullyConnected(data=loc, num_hidden=num_hidden, name="loc_fc")
                    stn = mx.sym.SpatialTransformer(data=data, loc=loc, target_shape=target_shape,
                                                    transform_type="affine", sampler_type="bilinear")
                    arg_names = stn.list_arguments()
                    arg_shapes, out_shapes, _ = stn.infer_shape(data=data_shape)
                    # check shape
                    assert out_shapes[0] == (data_shape[0], data_shape[1], target_shape[0], target_shape[1])
                    dev = default_context()
                    #dev = mx.gpu(0)
                    args = {}
                    args['data'] = mx.random.normal(0, 1, data_shape, ctx=mx.cpu()).copyto(dev)
                    args['loc_conv_weight'] = mx.nd.zeros((num_filter, data_shape[1], kernel[0], kernel[1]), ctx=dev)
                    args['loc_conv_bias'] = mx.nd.zeros((num_filter,), ctx=dev)
                    args['loc_fc_weight'] = mx.nd.zeros((6, num_filter*data_shape[2]*data_shape[3]), ctx=dev)
                    args['loc_fc_bias'] = mx.nd.array([0.5, 0, 0, 0, 0.5, 0], ctx=dev)
                    grad_grad = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
                    exe = stn.bind(dev, args=args, args_grad=grad_grad)
                    exe.forward(is_train=True)
                    out = exe.outputs[0].asnumpy()
                    # check forward
                    assert_almost_equal(out, args['data'].asnumpy()[:, :, h//4:h-h//4, w//4:w-w//4], rtol=1e-2, atol=1e-4)
                    out_grad = mx.nd.ones(out.shape, ctx=dev)
                    exe.backward([out_grad])
                    # check backward
                    assert_almost_equal(out_grad.asnumpy(), grad_grad[0].asnumpy()[:, :, h//4:h-h//4, w//4:w-w//4], rtol=1e-2, atol=1e-4)


def test_dot(ctx=default_context()):
    np.random.seed(1234)
    dtypes = ['float32', 'float64']

    # Test normal dot.
    for data_type in dtypes:
        for m in range(1, 5):
            for k in range(1, 5):
                for n in range(1, 5):
                    a_npy = np.random.normal(0, 1, (m, k))
                    a_npy = a_npy.astype(data_type)
                    b_npy = np.random.normal(0, 1, (k, n))
                    b_npy = b_npy.astype(data_type)
                    c_npy = np.empty((m, n), dtype=data_type)
                    ograd_npy = np.random.normal(0, 1, (m, n))
                    ograd_npy = ograd_npy.astype(data_type)
                    agrad_npy = np.empty((m, k), dtype=data_type)
                    bgrad_npy = np.empty((k, n), dtype=data_type)
                    c_npy[:, :] = np.dot(a_npy[:, :], b_npy[:, :])
                    bgrad_npy[:, :] = np.dot(a_npy[:, :].T, ograd_npy[:, :])
                    agrad_npy[:, :] = np.dot(ograd_npy[:, :], b_npy[:, :].T)
                    a = mx.sym.Variable('a', dtype=data_type)
                    b = mx.sym.Variable('b', dtype=data_type)
                    c = mx.sym.dot(a, b)
                    exe = c.simple_bind(ctx=ctx, a=a_npy.shape, b=b_npy.shape)
                    outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
                    assert_almost_equal(outputs[0].asnumpy(), c_npy, rtol=1e-3)
                    exe.backward(out_grads=[mx.nd.array(ograd_npy, mx.cpu())])
                    assert_almost_equal(exe.grad_dict['a'].asnumpy(), agrad_npy, rtol=1e-3)
                    assert_almost_equal(exe.grad_dict['b'].asnumpy(), bgrad_npy, rtol=1e-3)

    # Test dot with transpose flag using gradient checker.
    def dot_sym(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y)

    def dot_sym_xT(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y, transpose_a=True)

    def dot_sym_yT(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y, transpose_b=True)

    def dot_sym_xT_yT(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y, transpose_a=True, transpose_b=True)

    for data_type in dtypes:
        for ashape, bshape in [((3, 4), (4, 5)), ((2, 3, 4), (4, 5, 6))]:
            m1_npy = np.random.uniform(-1, 1, ashape)
            m1_npy = m1_npy.astype(data_type)
            m2_npy = np.random.uniform(-1, 1, bshape)
            m2_npy = m2_npy.astype(data_type)
            check_numeric_gradient(dot_sym(data_type), [m1_npy, m2_npy], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)
            check_numeric_gradient(dot_sym_xT(data_type), [m1_npy.T, m2_npy], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)
            check_numeric_gradient(dot_sym_yT(data_type), [m1_npy, m2_npy.T], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)
            check_numeric_gradient(dot_sym_xT_yT(data_type), [m1_npy.T, m2_npy.T], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)


def test_batch_dot():
    dtypes = ['float32', 'float64']

    for data_type in dtypes:
        for batch_size in range(1, 5):
            for m in range(1, 5):
                for k in range(1, 5):
                    for n in range(1, 5):
                        transpose_a = (np.random.rand() > 0.5)
                        transpose_b = (np.random.rand() > 0.5)
                        a_npy = np.random.normal(0, 1, (batch_size, m, k))
                        a_npy = a_npy.astype(data_type)
                        b_npy = np.random.normal(0, 1, (batch_size, k, n))
                        b_npy = b_npy.astype(data_type)
                        c_npy = np.empty((batch_size, m, n), dtype=data_type)
                        ograd_npy = np.random.normal(0, 1, (batch_size, m, n))
                        ograd_npy = ograd_npy.astype(data_type)
                        agrad_npy = np.empty((batch_size, m, k), dtype=data_type)
                        bgrad_npy = np.empty((batch_size, k, n), dtype=data_type)
                        a_init_grad_npy = np.random.normal(size=(batch_size, m, k))
                        a_init_grad_npy = a_npy.astype(data_type)
                        b_init_grad_npy = np.random.normal(size=(batch_size, k, n))
                        b_init_grad_npy = b_npy.astype(data_type)
                        for i in range(batch_size):
                            c_npy[i, :, :] = np.dot(a_npy[i, :, :], b_npy[i, :, :])
                            bgrad_npy[i, :, :] = np.dot(a_npy[i, :, :].T, ograd_npy[i, :, :])
                            agrad_npy[i, :, :] = np.dot(ograd_npy[i, :, :], b_npy[i, :, :].T)
                            a = mx.sym.Variable('a', dtype=data_type)
                            b = mx.sym.Variable('b', dtype=data_type)
                            c = mx.sym.batch_dot(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
                        if transpose_a:
                            a_npy = np.transpose(a_npy, axes=(0, 2, 1))
                            agrad_npy = np.transpose(agrad_npy, axes=(0, 2, 1))
                            a_init_grad_npy = np.transpose(a_init_grad_npy, axes=(0, 2, 1))
                        if transpose_b:
                            b_npy = np.transpose(b_npy, axes=(0, 2, 1))
                            bgrad_npy = np.transpose(bgrad_npy, axes=(0, 2, 1))
                            b_init_grad_npy = np.transpose(b_init_grad_npy, axes=(0, 2, 1))
                            exe = c.simple_bind(ctx=default_context(),
                                a=a_npy.shape, b=b_npy.shape, grad_req='write')
                            exe_add = c.simple_bind(ctx=default_context(),
                                a=a_npy.shape, b=b_npy.shape, grad_req='add')
                            exe_add.grad_dict['a'][:] = a_init_grad_npy
                            exe_add.grad_dict['b'][:] = b_init_grad_npy
                            outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
                            assert_almost_equal(outputs[0].asnumpy(), c_npy, rtol=1e-3, atol=1e-4)
                            exe.backward(out_grads=[mx.nd.array(ograd_npy, ctx=exe._ctx)])
                            assert_almost_equal(exe.grad_dict['a'].asnumpy(), agrad_npy, rtol=1e-3, atol=1e-4)
                            assert_almost_equal(exe.grad_dict['b'].asnumpy(), bgrad_npy, rtol=1e-3, atol=1e-4)
                            exe_add.forward(is_train=True, a=a_npy, b=b_npy)
                            exe_add.backward(out_grads=[mx.nd.array(ograd_npy, ctx=exe._ctx)])
                            assert_almost_equal(exe_add.grad_dict['a'].asnumpy(),
                                agrad_npy + a_init_grad_npy, rtol=1e-3, atol=1e-4)
                            assert_almost_equal(exe_add.grad_dict['b'].asnumpy(),
                                bgrad_npy + b_init_grad_npy, rtol=1e-3, atol=1e-4)


def get_correlation(data1,data2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    return mx.sym.Correlation(data1=img1,data2=img2,kernel_size =kernel_size,max_displacement = max_displacement,
                              stride1 = stride1,stride2 = stride2,pad_size= pad_size,is_multiply = is_multiply)


def correlation_forward(data1,data2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply):

    # compute output's dimension
    paddedbottomheight = data1.shape[2] + 2 * pad_size
    paddedbottomwidth = data1.shape[3] + 2 * pad_size
    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    top_width = (paddedbottomwidth - border_size * 2) // stride1
    top_height = (paddedbottomheight - border_size  * 2) // stride1
    neighborhood_grid_radius = max_displacement // stride2
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1
    top_channels = neighborhood_grid_width * neighborhood_grid_width

    out = np.zeros((data1.shape[0], top_channels, top_height, top_width))
    tmp1 = np.zeros((data1.shape[0],data1.shape[1],paddedbottomheight, paddedbottomwidth))
    tmp2 = np.zeros((data1.shape[0],data1.shape[1],paddedbottomheight, paddedbottomwidth))

    tmp1[:, :, pad_size:pad_size + data1.shape[2], pad_size:pad_size + data1.shape[3]] = data1[:,:,:,:]
    tmp2[:, :, pad_size:pad_size + data2.shape[2], pad_size:pad_size + data2.shape[3]] = data2[:,:,:,:]

    for i in range(top_height):
        for j in range(top_width):
            for nbatch in range(data1.shape[0]):

                # x1,y1 is the location in data1 , i,j is the location in output
                x1 = j * stride1 + max_displacement
                y1 = i * stride1 + max_displacement

                for top_channel in range(top_channels):

                    s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2
                    s2p = (top_channel // neighborhood_grid_width - neighborhood_grid_radius) * stride2

                    # location in data2
                    x2 = x1 + s2o
                    y2 = y1 + s2p

                    for h in range(kernel_size):
                        for w in range(kernel_size):
                            for channel in range(data1.shape[1]):
                                if is_multiply:
                                    out[nbatch, top_channel, i, j] += tmp1[nbatch, channel,y1 + h, x1 + w] * tmp2[nbatch, channel, y2 + h,x2 + w]
                                else:
                                    out[nbatch, top_channel, i, j] += abs(tmp1[nbatch, channel, y1 + h, x1 + w] - tmp2[nbatch, channel, y2 + h, x2 + w])
    out /= float(kernel_size**2*data1.shape[1])
    return out,tmp1,tmp2


def correlation_backward(out_grad,tmp1,tmp2,data1,data2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply):

    # compute output's dimension
    paddedbottomheight = data1.shape[2] + 2 * pad_size
    paddedbottomwidth = data1.shape[3] + 2 * pad_size
    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    top_width = (paddedbottomwidth - border_size * 2) // stride1
    top_height = (paddedbottomheight - border_size  * 2) // stride1
    neighborhood_grid_radius = max_displacement // stride2
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1
    top_channels = neighborhood_grid_width * neighborhood_grid_width

    out = np.zeros((data1.shape[0], top_channels, top_height, top_width))
    tmp1_grad = np.zeros(tmp1.shape)
    tmp2_grad = np.zeros(tmp2.shape)

    for i in range(top_height):
        for j in range(top_width):
            for nbatch in range(data1.shape[0]):

                # x1,y1 is the location in data1 , i,j is the location in output
                x1 = j * stride1 + max_displacement
                y1 = i * stride1 + max_displacement

                for top_channel in range(top_channels):

                    s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2
                    s2p = (top_channel // neighborhood_grid_width - neighborhood_grid_radius) * stride2

                    # location in data2
                    x2 = x1 + s2o
                    y2 = y1 + s2p

                    for h in range(kernel_size):
                        for w in range(kernel_size):
                            for channel in range(data1.shape[1]):
                                if is_multiply:
                                    tmp1_grad[nbatch,channel,y1+h,x1+w]+= out_grad[nbatch,top_channel,i,j]*tmp2[nbatch, channel, y2 + h,x2 + w]
                                    tmp2_grad[nbatch,channel,y2+h,x2+w]+= out_grad[nbatch,top_channel,i,j]*tmp1[nbatch, channel, y1 + h,x1 + w]
                                else:
                                    sgn = 1 if (tmp1[nbatch, channel, y1 + h,x1 + w]>=tmp2[nbatch, channel, y2 + h,x2 + w]) else -1
                                    tmp1_grad[nbatch,channel,y1+h,x1+w]+= out_grad[nbatch,top_channel,i,j]*sgn
                                    tmp2_grad[nbatch,channel,y2+h,x2+w]+= out_grad[nbatch,top_channel,i,j]*(-sgn)

    tmp1_grad = tmp1_grad / float(kernel_size**2*data1.shape[1])
    tmp2_grad = tmp2_grad / float(kernel_size**2*data1.shape[1])
    return tmp1_grad[:,:,pad_size:pad_size+data1.shape[2],pad_size:pad_size+data1.shape[3]],tmp2_grad[:,:,pad_size:pad_size+data1.shape[2],pad_size:pad_size+data1.shape[3]],


def unittest_correlation(data_shape,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply):

    img1 = np.random.random(data_shape)
    img1 = img1.astype(np.float32)
    img2 = np.random.random(data_shape)
    img2 = img2.astype(np.float32)

    net1 = get_correlation(img1,img2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply)
    net2 = get_correlation(img1,img2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply )

    exe1 = net1.simple_bind(default_context(),img1=img1.shape,img2=img1.shape)
    exe1.arg_dict['img1'][:] = img1
    exe1.arg_dict['img2'][:] = img2

    #cpu forward
    exe1.forward(is_train=True)
    # python forward
    forward_result,tmp1,tmp2 = correlation_forward(img1,img2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply)

    # forward error
    assert_almost_equal(exe1.outputs[0].asnumpy(), forward_result, rtol=1e-4, atol=1e-4)

    # out_grad
    a = np.ones(forward_result.shape)
    out_grad1 = mx.nd.array(a,default_context())
    # cpu backward
    exe1.backward(out_grads=out_grad1)
    # python backward
    grad1,grad2 = correlation_backward(a,tmp1,tmp2,img1,img2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply)

    # backward error
    assert_almost_equal(exe1.grad_dict['img1'].asnumpy(), grad1, rtol=1e-3, atol=1e-4)
    assert_almost_equal(exe1.grad_dict['img2'].asnumpy(), grad2, rtol=1e-3, atol=1e-4)


def test_correlation():
    unittest_correlation((1,3,10,10), kernel_size = 1,max_displacement = 4,stride1 = 1,stride2 = 1,pad_size = 4,is_multiply = False)
    unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 5,stride1 = 1,stride2 = 1,pad_size = 5,is_multiply = False)
    unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 5,stride1 = 1,stride2 = 1,pad_size = 5,is_multiply = True)
    unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 10,stride1 = 1,stride2 = 2,pad_size = 10,is_multiply = True)
    unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 1,stride2 = 1,pad_size = 2,is_multiply = True)
    unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = True)
    unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = False)
    unittest_correlation((5,1,6,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = False)
    unittest_correlation((5,1,11,11), kernel_size = 5,max_displacement = 1,stride1 = 1,stride2 = 1,pad_size = 2,is_multiply = False)


def test_support_vector_machine_l1_svm():
    xpu = default_context()
    shape = (20, 10)

    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SVMOutput(data=X, label=L, use_linear=True)
    x = mx.nd.empty(shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    x_np = np.random.rand(*shape)
    l_np = np.random.randint(0, shape[1], (shape[0],))
    x[:] = x_np
    l[:] = l_np

    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)

    assert_almost_equal(x_np, exec1.outputs[0].asnumpy())

    exec1.backward()

    l_mask = np.equal(l_np.reshape(shape[0],1),range(shape[1]))
    l_mask = np.array(l_mask, dtype=np.float32)*2 -1
    grad_np = (-1) * l_mask * np.greater(1 - l_mask * x_np, 0)

    assert_almost_equal(grad_np, grad.asnumpy())


def test_support_vector_machine_l2_svm():
    xpu = default_context()
    shape = (20, 10)

    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SVMOutput(data=X, label=L)
    x = mx.nd.empty(shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    x_np = np.random.rand(*shape)
    x_np = x_np.astype(np.float32)
    l_np = np.random.randint(0, shape[1], (shape[0],))
    x[:] = x_np
    l[:] = l_np

    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)

    assert_almost_equal(x_np, exec1.outputs[0].asnumpy())

    exec1.backward()

    l_mask = np.equal(l_np.reshape(shape[0],1),range(shape[1]))
    l_mask = np.array(l_mask, dtype=np.float32)*2 -1
    grad_np = (-2)*l_mask*np.maximum(1-l_mask*x_np,0)
    grad_np = grad_np.astype(np.float32)
    assert_almost_equal(grad_np, grad.asnumpy())


def test_roipooling():
    np.random.seed(1234)

    data = mx.symbol.Variable(name='data')
    rois = mx.symbol.Variable(name='rois')
    test = mx.symbol.ROIPooling(data=data, rois=rois, pooled_size=(4, 4), spatial_scale=1)

    x1 = np.random.rand(4, 3, 12, 8).astype('float32')
    x2 = np.array([[0, 1.1, 1.1, 6.2, 6.2], [2, 6.1, 2.1, 8.2, 11.2], [1, 3.1, 1.1, 5.2, 10.2], [0, 3, 3, 3, 3]], dtype='float32')

    check_numeric_gradient(sym=test, location=[x1, x2],
                           grad_nodes={'data':'write', 'rois':'null'},
                           numeric_eps=1e-4, rtol=1e-1, atol=1e-4)
    check_numeric_gradient(sym=test, location=[x1, x2],
                           grad_nodes={'data':'add', 'rois':'null'},
                           numeric_eps=1e-4, rtol=1e-1, atol=1E-4)


def check_pad_with_shape(shape, xpu, pad_width, mode):
    # bind with label
    X = mx.symbol.Variable('X')
    Y = mx.symbol.Pad(data=X, mode=mode, pad_width=pad_width)
    x = mx.random.uniform(-1, 1, shape, ctx=mx.cpu()).copyto(xpu)
    # numpy result
    pad_grouped = list(zip(*[iter(list(pad_width))] * 2))
    np_out = np.pad(x.asnumpy(), pad_grouped, mode)
    # mxnet result
    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x], args_grad = {'X': grad})
    exec1.forward(is_train=True)
    out = exec1.outputs[0].asnumpy()
    # compare numpy + mxnet
    assert_almost_equal(out, np_out)
    # grad check
    check_numeric_gradient(Y, [x.asnumpy()], numeric_eps=1e-2, rtol=1e-2)


def test_pad():
    shape1 = (2, 3, 3, 5)
    pad1 = (0, 0, 0, 0, 1, 2, 3, 4)
    shape2 = (2, 3, 3, 5, 4)
    pad2 = (0, 0, 0, 0, 1, 2, 3, 4, 3, 1)
    check_pad_with_shape(shape1, default_context(), pad1, 'constant')
    check_pad_with_shape(shape1, default_context(), pad1, 'edge')
    check_pad_with_shape(shape2, default_context(), pad2, 'constant')
    check_pad_with_shape(shape2, default_context(), pad2, 'edge')
    check_pad_with_shape(shape1, default_context(), pad1, 'reflect')
    check_pad_with_shape(shape2, default_context(), pad2, 'reflect')


def np_instance_norm(data, weight, bias, eps):
    spatial_dims = data.shape[2::]
    num_spatial_vals = np.prod(np.array(spatial_dims))
    scale = 1/float(num_spatial_vals)
    sum_axis = tuple(range(2, data.ndim))
    mean = scale * np.sum(data, axis = sum_axis)
    mean = np.reshape(np.repeat(mean, num_spatial_vals), data.shape)
    var = scale * np.sum((data - mean)**2, axis = sum_axis)
    var = np.reshape(np.repeat(var, num_spatial_vals), data.shape)

    weightBatch = np.tile(weight, (data.shape[0], 1))
    weightBatch = np.reshape(np.repeat(weightBatch, num_spatial_vals), data.shape)
    biasBatch = np.tile(bias, (data.shape[0], 1))
    biasBatch = np.reshape(np.repeat(biasBatch, num_spatial_vals), data.shape)
    return weightBatch * (data - mean)/np.sqrt(var + eps) + biasBatch


def check_instance_norm_with_shape(shape, xpu):
    # bind with label
    eps = 0.001
    X = mx.symbol.Variable('X')
    G = mx.symbol.Variable('G')
    B = mx.symbol.Variable('B')

    Y = mx.symbol.InstanceNorm(data=X, beta=B, gamma=G, eps=eps)
    x = mx.random.normal(0, 1, shape, ctx=mx.cpu()).copyto(xpu)
    gamma = mx.random.normal(0, 1, shape[1], ctx=mx.cpu()).copyto(xpu)
    beta = mx.random.normal(0, 1, shape[1], ctx=mx.cpu()).copyto(xpu)

    np_out = np_instance_norm(x.asnumpy(), gamma.asnumpy(), beta.asnumpy(), eps)
    exec1 = Y.bind(xpu, args = {'X':x, 'G':gamma, 'B':beta})
    exec1.forward(is_train=False)
    out = exec1.outputs[0].asnumpy()
    assert_almost_equal(out, np_out, rtol=1e-4, atol=1e-4)
    check_numeric_gradient(Y, {'X':x.asnumpy(), 'G':gamma.asnumpy(), 'B':beta.asnumpy()},
                           numeric_eps=1e-2, rtol=1e-2, atol=1e-2)


def test_instance_normalization():
    check_instance_norm_with_shape((1, 1, 1), default_context())
    check_instance_norm_with_shape((2, 1, 2), default_context())
    check_instance_norm_with_shape((2,4,5,6), default_context())
    check_instance_norm_with_shape((3,3,2,3,2,1,1), default_context())


def check_l2_normalization(in_shape, mode, ctx=default_context(), norm_eps=1e-10):
    data = mx.symbol.Variable('data')
    out = mx.symbol.L2Normalization(data=data, mode=mode, eps=norm_eps)
    # TODO(szha): Seeding this masks failures. We need to do a deep dive for failures without this seed.
    np.random.seed(1234)
    in_data = np.random.uniform(-1, 1, in_shape)
    # calculate numpy results
    if mode == 'channel':
        assert in_data.ndim > 2
        np_norm = np.linalg.norm(in_data, axis=1) + norm_eps
        np_norm = np.repeat(1. / np.expand_dims(np_norm, axis=1), in_data.shape[1], axis=1)
        np_out = np.multiply(in_data, np_norm)
    elif mode == 'spatial':
        assert in_data.ndim > 2
        s = in_data.shape
        np_norm = np.linalg.norm(in_data.reshape((s[0], s[1], -1)), axis=2) + norm_eps
        np_norm = np.repeat(1. / np_norm[:, np.newaxis], in_data.size / s[0] / s[1], axis=2)
        np_out = np.multiply(in_data, np_norm.reshape(s))
    elif mode == 'instance':
        assert in_data.ndim > 1
        s = in_data.shape
        np_norm = np.linalg.norm(in_data.reshape((s[0], -1)), axis=1) + norm_eps
        np_norm = np.repeat(1. / np_norm[:, np.newaxis], in_data.size / s[0], axis=1)
        np_out = np.multiply(in_data, np_norm.reshape(s))
    else:
        raise RuntimeError('Unknown l2 normalization mode')
    exe = out.simple_bind(ctx=ctx, data=in_data.shape)
    output = exe.forward(is_train=True, data=in_data)
    # compare numpy + mxnet
    assert_almost_equal(exe.outputs[0].asnumpy(), np_out, rtol=1e-5)
    # check gradient
    check_numeric_gradient(out, [in_data], numeric_eps=1e-3, rtol=1e-2, atol=1e-3)


def test_l2_normalization():
    for mode in ['channel', 'spatial', 'instance']:
        for nbatch in [1, 4]:
            for nchannel in [3, 5]:
                for height in [4, 6]:
                    check_l2_normalization((nbatch, nchannel, height), mode)
                    for width in [5, 7]:
                        check_l2_normalization((nbatch, nchannel, height, width), mode)


# Numpy Implementation of Sequence Ops
def sequence_last_numpy(array, lengths, axis):
    # create new array of dims [batch, seqlen, ...]
    array2 = np.moveaxis(array, axis, 1)
    dims = array2.shape
    if lengths is None:
        return array2[:, -1]
    lengths = list(lengths)
    return np.array([array2[i, int(lengths[i]) - 1] for i in range(dims[0])])


def sequence_mask_numpy(array, lengths, axis, value):
    if lengths is None:
        return array
    arrayMask = array.copy()
    # conform to [batch, seqlen, ...]
    arrayMask = np.moveaxis(arrayMask, axis, 1)
    shape = arrayMask.shape
    lengths = list(lengths)
    for i in range(shape[0]):
        arrayMask[i, int(lengths[i]):] = value
    return np.moveaxis(arrayMask, 1, axis)


def sequence_reverse_numpy(array, lengths, axis):
    rarray = array.copy()
    # conform to [batch, seqlen, ...]
    rarray = np.moveaxis(rarray, axis, 1)
    shape = rarray.shape
    if lengths is None:
        lengths = [shape[1]] * shape[0]
    lengths = list(lengths)
    for i in range(shape[0]):
        j = int(lengths[i])
        rarray[i,:j] = rarray[i,:j][::-1]
    return np.moveaxis(rarray, 1, axis)


def check_sequence_func(ftype, mask_value=0, axis=0):
    # bind with label
    xpu = default_context()
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L') # lengths
    shapes = [(3, 4), (1, 1), (3, 4, 3, 1, 1)]
    for seqlenQ in [True, False]:
        for s in shapes:
            x = mx.random.uniform(-1, 1, s, ctx=mx.cpu()).copyto(xpu)
            batch = s[1] if (axis == 0) else s[0]
            seqlen = s[axis]
            l_np = np.random.randint(1, seqlen + 1, batch)
            l = mx.nd.array(l_np, ctx=mx.cpu()).copyto(xpu)
            if not seqlenQ:
                l_np = None
            args = {'data':X, 'use_sequence_length':seqlenQ, "axis":axis}
            if seqlenQ:
                args['sequence_length'] = L
            if ftype == "last":
                Y = mx.symbol.SequenceLast(**args)
                np_out = sequence_last_numpy(x.asnumpy(), l_np, axis)
            elif ftype == "mask":
                args['value'] = mask_value
                Y = mx.symbol.SequenceMask(**args)
                np_out = sequence_mask_numpy(x.asnumpy(), l_np, axis, mask_value)
            elif ftype == "reverse":
                Y = mx.symbol.SequenceReverse(**args)
                np_out = sequence_reverse_numpy(x.asnumpy(), l_np, axis)
            fargs = [x, l] if seqlenQ else [x]
            gargs = [x.asnumpy(), l_np] if seqlenQ else [x.asnumpy()]
            check_symbolic_forward(Y, fargs, [np_out])
            check_numeric_gradient(Y, gargs, grad_nodes={'X':'write'},
                numeric_eps=1e-2, rtol=1e-2)
            check_numeric_gradient(Y, gargs, grad_nodes={'X':'add'},
                numeric_eps=1e-3, rtol=1e-2, atol=1E-4)
            check_numeric_gradient(Y, gargs, grad_nodes={'X':'null'},
                numeric_eps=1e-3, rtol=1e-2, atol=1E-4)


def test_sequence_last():
    check_sequence_func("last", axis=0)
    check_sequence_func("last", axis=1)


def test_sequence_mask():
    check_sequence_func("mask", axis = 0, mask_value=-2.3)
    check_sequence_func("mask", axis = 1, mask_value=0.3)


def check_sequence_reverse(xpu):
    # sample data
    arr = np.array(
        [[[  1.,   2.,   3.],
          [  4.,   5.,   6.]],
         [[  7.,   8.,   9.],
          [ 10.,  11.,  12.]],
         [[ 13.,  14.,   15.],
          [ 16.,  17.,   18.]]])

    arr1 = np.array(
        [[[  13.,   14.,   15.],
          [  16.,   17.,   18.]],
         [[  7.,   8.,   9.],
          [ 10.,  11.,  12.]],
         [[ 1.,  2.,   3.],
          [ 4.,  5.,   6.]]])

    arr2 = np.array(
        [[[  7.,   8.,   9.],
          [  10.,   11.,   12.]],
         [[  1.,   2.,   3.],
          [ 4.,  5.,   6.]],
         [[ 13.,  14.,   15.],
          [ 16.,  17.,   18.]]])

    arr3 = np.array(
        [[[  7.,   8.,   9.],
          [  16.,   17.,   18.]],
         [[  1.,   2.,   3.],
          [ 10.,  11.,  12.]],
         [[ 13.,  14.,   15.],
          [ 4.,  5.,   6.]]])

    # test for matrix case
    seq_len_1 = [1, 2, 2]
    arr_4 = np.array([[7., 8., 9.], [16., 17., 5.4]], dtype=np.float32)
    arr_5 = np.array([[7., 17., 5.4], [16., 8., 9.]], dtype=np.float32)

    def test_wrapper(arr, xpu, sequence_length=None, use_sequence_length=False):
        # MxNet symbol creation
        seq = mx.sym.Variable('seq')
        if sequence_length and use_sequence_length:
            seq_len = mx.sym.Variable('seq_len')
        else:
           # ensure that both are disabled, not just one
           seq_len=None
           use_sequence_length=False
        rev = mx.sym.SequenceReverse(data=seq, sequence_length=seq_len, use_sequence_length=use_sequence_length)
        # MxNet symbol execution
        if sequence_length:
            bound = rev.bind(xpu, {'seq': mx.nd.array(arr), 'seq_len': mx.nd.array(sequence_length)})
        else:
            bound = rev.bind(xpu, {'seq': mx.nd.array(arr)})
        fwd = bound.forward()
        return fwd[0].asnumpy()

    # test cases
    assert_array_equal(test_wrapper(arr, xpu, use_sequence_length=False), arr1)
    assert_array_equal(test_wrapper(arr, xpu, sequence_length=[3, 3], use_sequence_length=True), arr1)
    assert_array_equal(test_wrapper(arr, xpu, sequence_length=[2, 2], use_sequence_length=True), arr2)
    assert_array_equal(test_wrapper(arr, xpu, sequence_length=[2, 3], use_sequence_length=True), arr3)
    assert_array_equal(test_wrapper(arr_4, xpu, sequence_length=seq_len_1, use_sequence_length=True), arr_5)


def test_sequence_reverse():
    check_sequence_func("reverse", axis=0)
    check_sequence_reverse(mx.cpu())


def mathematical_core_binary(name,
                             forward_mxnet_call,
                             forward_numpy_call,
                             backward_numpy_call1,
                             backward_numpy_call2,
                             data1_init=2.,
                             data2_init=3.,
                             grad_init=2.):
    data1 = mx.symbol.Variable('data')
    data2 = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp1 = np.random.rand(3, 4)
    data_tmp2 = np.random.rand(3, 4)
    data_tmp1[:] = data1_init
    data_tmp2[:] = data2_init

    arr_data1 = mx.nd.array(data_tmp1)
    arr_data2 = mx.nd.array(data_tmp2)

    arr_grad1 = mx.nd.empty(shape)
    arr_grad2 = mx.nd.empty(shape)

    test = forward_mxnet_call(data1, data2)
    exe_test = test.bind(default_context(), args=[arr_data1, arr_data2], args_grad=[arr_grad1, arr_grad2])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout = forward_numpy_call(data_tmp1, data_tmp2)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = grad_init
    exe_test.backward(out_grad)

    npout_grad = np.ones(shape)
    npout_grad[:] = grad_init

    npout_grad1 = npout_grad * backward_numpy_call1(data_tmp1, data_tmp2)
    npout_grad2 = npout_grad * backward_numpy_call2(data_tmp1, data_tmp2)
    arr_grad1 = arr_grad1.asnumpy()
    arr_grad2 = arr_grad2.asnumpy()

    assert_almost_equal(arr_grad1, npout_grad1)
    assert_almost_equal(arr_grad2, npout_grad2)


def mathematical_core(name, forward_mxnet_call, forward_numpy_call, backward_numpy_call, data_init=5., grad_init=2.):
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:] = data_init
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:] = 3

    test = forward_mxnet_call(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout = forward_numpy_call(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = grad_init
    npout_grad = out_grad.asnumpy()
    temp = backward_numpy_call(data_tmp)
    npout_grad = npout_grad * temp
    exe_test.backward(out_grad)
    arr_grad = arr_grad.asnumpy()
    # print(name)
    # print(arr_grad)
    # print(npout_grad)
    assert_almost_equal(arr_grad, npout_grad)


def test_special_functions_using_scipy():
    try:
        from scipy import special as scipy_special
    except:
        print("Could not import scipy. Skipping unit tests for special functions")
        return

    # gamma
    mathematical_core("gamma", lambda x: mx.sym.gamma(x), lambda x: scipy_special.gamma(x),
                     lambda x: scipy_special.gamma(x) * scipy_special.psi(x), 0.5, 0.5)

    # gammaln
    mathematical_core("gammaln", lambda x: mx.sym.gammaln(x), lambda x: scipy_special.gammaln(x),
                     lambda x: scipy_special.psi(x), 0.5, 0.5)


def rounding(name, forward_mxnet_call, forward_numpy_call, data_init=5., grad_init=2.):
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:] = data_init
    arr_data = mx.nd.array(data_tmp)

    test = forward_mxnet_call(data)
    exe_test = test.bind(default_context(), args=[arr_data])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0].asnumpy()
    npout = forward_numpy_call(data_tmp)
    assert_almost_equal(out, npout)


def test_mathematical():
    # rsqrt
    mathematical_core("rsqrt",
                      lambda x: mx.sym.rsqrt(x),
                      lambda x: 1 / np.sqrt(x),
                      lambda x: -(1.0 / (2.0 * x * np.sqrt(x))))
    # tan
    mathematical_core("tan", lambda x: mx.sym.tan(x), lambda x: np.tan(x), lambda x: np.tan(x) ** 2 + 1)
    # arcsin
    mathematical_core("arcsin", lambda x: mx.sym.arcsin(x), lambda x: np.arcsin(x),
                      lambda x: 1. / (1. - x ** 2) ** (1. / 2.), 0.5, 0.5)
    # arccos
    mathematical_core("arccos", lambda x: mx.sym.arccos(x), lambda x: np.arccos(x),
                      lambda x: -1. / (1. - x ** 2.) ** (1. / 2.), 0.5, 0.5)
    # arctan
    mathematical_core("arctan", lambda x: mx.sym.arctan(x), lambda x: np.arctan(x),
                      lambda x: 1. / (x ** 2. + 1.), 0.5, 0.5)
    # hypot
    mathematical_core_binary("hypot",
                             lambda x, y: mx.sym.hypot(x, y),
                             lambda x, y: np.hypot(x, y),
                             lambda x, y: x / np.hypot(x, y),
                             lambda x, y: y / np.hypot(x, y),
                             0.5, 0.5, 0.5)

    # hypot scalar
    mathematical_core("hypot scalar",
                      lambda x: mx.sym.hypot(x, 3),
                      lambda x: np.hypot(x, 3),
                      lambda x: x / np.hypot(x, 3),
                      0.5, 0.5)

    # degrees
    mathematical_core("degrees",
                      lambda x: mx.sym.degrees(x),
                      lambda x: np.degrees(x),
                      lambda x: 180./np.pi,
                      0.5, 0.5)
    # radians
    mathematical_core("radians",
                      lambda x: mx.sym.radians(x),
                      lambda x: np.radians(x),
                      lambda x: np.pi / 180.,
                      0.6, 1)
    # sinh
    mathematical_core("sinh", lambda x: mx.sym.sinh(x), lambda x: np.sinh(x), lambda x: np.cosh(x))

    # cosh
    mathematical_core("cosh", lambda x: mx.sym.cosh(x), lambda x: np.cosh(x), lambda x: np.sinh(x), 5, 5)

    # tanh
    mathematical_core("tanh", lambda x: mx.sym.tanh(x), lambda x: np.tanh(x), lambda x: 1. - np.tanh(x) ** 2, 0.5, 1)

    # arcsinh
    mathematical_core("arcsinh", lambda x: mx.sym.arcsinh(x), lambda x: np.arcsinh(x),
                      lambda x: 1./(x**2 + 1.)**(1./2.))

    # arccosh
    mathematical_core("arccosh", lambda x: mx.sym.arccosh(x), lambda x: np.arccosh(x),
                      lambda x: 1./(x**2 - 1.)**(1./2.))

    # arctanh
    mathematical_core("arctanh", lambda x: mx.sym.arctanh(x), lambda x: np.arctanh(x),
                      lambda x: -1./(x**2 - 1.), 0.5)

    # log1p
    mathematical_core("log1p", lambda x: mx.sym.log1p(x), lambda x: np.log1p(x),
                      lambda x: 1. / (1.0 + x), 0.5, 0.5)
    # expm1
    mathematical_core("expm1", lambda x: mx.sym.expm1(x), lambda x: np.expm1(x),
                      lambda x: np.exp(x), 0.5, 0.5)

    # log10
    mathematical_core("log10", lambda x: mx.sym.log10(x), lambda x: np.log10(x),
                      lambda x: 1. / (x * np.log(10.)))

    # log2
    mathematical_core("log2", lambda x: mx.sym.log2(x), lambda x: np.log2(x),
                      lambda x: 1. / (x * np.log(2.)))

    # rint
    rounding("rint", lambda x: mx.sym.rint(x), lambda x: np.rint(x))

    # fix
    rounding("fix", lambda x: mx.sym.fix(x), lambda x: np.fix(x))


def test_special_functions_using_scipy():
    try:
        from scipy import special as scipy_special
    except:
        print("Could not import scipy. Skipping unit tests for special functions")
        return

    # gamma
    mathematical_core("gamma", lambda x: mx.sym.gamma(x), lambda x: scipy_special.gamma(x),
                     lambda x: scipy_special.gamma(x) * scipy_special.psi(x), 0.5, 0.5)

    # gammaln
    mathematical_core("gammaln", lambda x: mx.sym.gammaln(x), lambda x: scipy_special.gammaln(x),
                     lambda x: scipy_special.psi(x), 0.5, 0.5)


def test_clip():
    data = mx.symbol.Variable('data')
    shape = (30, 30)
    data_tmp = np.random.uniform(-1, 1, shape)
    test = mx.sym.clip(data, a_max=0.6, a_min=-0.6)
    check_symbolic_forward(test, [data_tmp], [np.clip(data_tmp, -0.6, 0.6)])
    check_symbolic_backward(test, [data_tmp], [np.ones(shape)],
                            [np.where(data_tmp < 0.6, [1], [0]) * np.where(data_tmp > -0.6, [1], [0])])


def test_init():
    def test_basic_val_init(sym_func, np_func, shape, dtype):
        x = sym_func(shape=shape, dtype=dtype)
        exe = x.bind(default_context(), args=[], args_grad=[])
        exe.forward(is_train=True)
        assert_almost_equal(exe.outputs[0].asnumpy(), np_func(shape=shape, dtype=dtype))
        assert exe.outputs[0].asnumpy().dtype == dtype

    def test_arange():
        # General Random Tests
        dtype_list = [np.float32, np.float64, np.int32, np.uint8]
        config_list = [(10,),
                       (0, 10),
                       (5, 100, 4),
                       (50, -50, -2),
                       (-100, 100, 1),
                       (1.3, 456.6, 1.3)]
        for dtype in dtype_list:
            for config in config_list:
                repeats = random.choice([1, 3])
                np_out = np.repeat(np.arange(*config, dtype=dtype), repeats)
                nd_out = mx.nd.arange(*config, repeat=repeats, dtype=dtype)
                assert_almost_equal(np_out, nd_out.asnumpy())

    test_basic_val_init(mx.sym.zeros, np.zeros, (3, 4), np.float32)
    test_basic_val_init(mx.sym.ones, np.ones, 3, np.int32)
    test_basic_val_init(mx.sym.ones, np.ones, (2, 2, 3), np.float16)
    test_arange()


def test_order():
    ctx = default_context()

    def gt_topk(dat, axis, ret_typ, k, is_ascend):
        if ret_typ == "indices":
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            ret = np.take(dat.argsort(axis=axis), axis=axis, indices=indices, mode='wrap')
        elif ret_typ == "value":
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            ret = np.take(np.sort(dat, axis=axis), axis=axis, indices=indices, mode='wrap')
        else:
            assert dat.shape == (5, 5, 5, 5)
            assert axis is None or axis == 1
            ret = np.zeros(dat.shape)
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            gt_argsort = np.take(dat.argsort(axis=axis), axis=axis, indices=indices, mode='wrap')
            if axis is None:
                ret.ravel()[gt_argsort] = 1
            else:
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            ret[i, gt_argsort[i, :, j, k], j, k] = 1
        return ret

    dshape = (5, 5, 5, 5)
    a_npy = np.arange(np.prod(dshape)).astype(np.float32)
    np.random.shuffle(a_npy)
    a_npy = a_npy.reshape(dshape)
    a = mx.sym.Variable('a')

    for axis in [1, 3, None]:
        K = [1, 3, 5, 7] if axis is None else [1, 3, 5]
        for k in K:
            for is_ascend in [True, False]:
                b = mx.sym.topk(a, axis=axis, is_ascend=is_ascend, ret_typ="value", k=k)
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=k, is_ascend=is_ascend)
                check_numeric_gradient(b, location={'a': a_npy}, numeric_eps=1e-2, ctx=ctx)
                check_symbolic_forward(b, location={'a': a_npy}, expected=[out_npy])

    for axis in [1, 3, None]:
        for is_ascend in [True, False]:
            b = mx.sym.sort(a, axis=axis, is_ascend=is_ascend)
            if axis is None:
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=a_npy.size, is_ascend=is_ascend)
            else:
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=5, is_ascend=is_ascend)
            check_numeric_gradient(b, location={'a': a_npy}, numeric_eps=1e-2, ctx=ctx)
            check_symbolic_forward(b, location={'a': a_npy}, expected=[out_npy])

    b = mx.sym.topk(a, axis=3, is_ascend=is_ascend, ret_typ="indices", k=3)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 3))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=3, ret_typ="indices", k=3,
                                             is_ascend=False)])

    b = mx.sym.topk(a, axis=1, is_ascend=True, ret_typ="mask", k=3)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="mask", k=3,
                                             is_ascend=True)])

    b = mx.sym.argsort(a, axis=1, is_ascend=False)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=5,
                                             is_ascend=False)])

    b = mx.sym.argmax(a, axis=1, keepdims=True)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=1,
                                             is_ascend=False)])

    b = mx.sym.argmin(a, axis=1, keepdims=True)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=1,
                                             is_ascend=True)])


def test_blockgrad():
    a = mx.sym.Variable('a')
    b = mx.sym.BlockGrad(a)
    exe = b.simple_bind(ctx=default_context(), a=(10, 10))
    a_npy = np.random.rand(10, 10)
    exe.forward(is_train=True, a=a_npy)
    assert_almost_equal(exe.outputs[0].asnumpy(), a_npy)
    exe.backward()  # No error if BlockGrad works


def test_take():
    def check_output_n_grad(data_shape, idx_shape):
        exe = result.simple_bind(default_context(), a=data_shape,
                                 indices=idx_shape)
        data_real = np.random.normal(size=data_shape).astype('float32')
        idx_real = np.random.randint(low=0, high=data_shape[0], size=idx_shape)
        grad_out = np.ones(idx_shape + data_shape[1:], dtype='float32')
        grad_in = np.zeros(data_shape, dtype='float32')

        exe.arg_dict['a'][:] = mx.nd.array(data_real)
        exe.arg_dict['indices'][:] = mx.nd.array(idx_real)
        exe.forward(is_train=True)
        assert_almost_equal(exe.outputs[0].asnumpy(), data_real[idx_real])

        for i in np.nditer(idx_real):
            grad_in[i] += 1.0

        exe.backward([mx.nd.array(grad_out)])
        assert_almost_equal(exe.grad_dict['a'].asnumpy(), grad_in)

    data = mx.sym.Variable('a')
    idx = mx.sym.Variable('indices')
    idx = mx.sym.BlockGrad(idx)
    result = mx.sym.take(a=data, indices=idx)

    for data_ndim in range(2, 5):
        for idx_ndim in range(1, 4):
            data_shape = ()
            for _ in range(data_ndim):
                data_shape += (np.random.randint(low=3, high=6), )
            idx_shape = ()
            for _ in range(idx_ndim):
                idx_shape += (np.random.randint(low=3, high=5), )
            check_output_n_grad(data_shape, idx_shape)


def test_grid_generator():
    # transform_type =  affine
    test_case = [(20,21),(4,3),(6,12),(15,17)]
    for target_shape in test_case:
        affine_matrix =  mx.sym.Variable('affine')
        grid = mx.sym.GridGenerator(data=affine_matrix,transform_type='affine', target_shape=target_shape)
        exe = grid.simple_bind(ctx=default_context(), affine=(1,6), grad_req='write')

        # check forward
        exe.arg_dict['affine'][:] = np.array([[1.0,0,0,0,1.0,0]])
        exe.forward(is_train=True)
        output = exe.outputs[0].asnumpy()
        output[0,0,:,:] = (output[0,0,:,:] + 1) * (target_shape[1] - 1) / 2.0
        output[0,1,:,:] = (output[0,1,:,:] + 1) * (target_shape[0] - 1) / 2.0
        xv, yv = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
        assert_almost_equal(output[0,0], yv.T)
        assert_almost_equal(output[0,1], xv.T)

        # check backward
        out_grad = np.random.normal(size=(1,2)+target_shape)
        exe.backward(mx.nd.array(out_grad))
        tmp = np.zeros((3,target_shape[0]*target_shape[1]))
        tmp[0] = -1.0 + (np.arange(target_shape[0]*target_shape[1]) % target_shape[1]) * (2.0 / (target_shape[1]-1))
        tmp[1] = -1.0 + (np.arange(target_shape[0]*target_shape[1]) // target_shape[1]) * (2.0 / (target_shape[0]-1))
        tmp[2] = 1
        grad_est = np.dot(out_grad[0].reshape(2,target_shape[0]*target_shape[1]),tmp.T).reshape(1,6)
        assert_almost_equal(exe.grad_dict['affine'].asnumpy(), grad_est, rtol=1e-3, atol=1e-5)
        # check addto
        exe = grid.simple_bind(ctx=default_context(), affine=(1,6), grad_req='add')
        grid_grad_npy = np.random.normal(size=exe.grad_dict['affine'].shape)
        exe.grad_dict['affine'][:] = grid_grad_npy
        exe.arg_dict['affine'][:] = np.array([[1.0, 0, 0, 0, 1.0, 0]])
        exe.forward(is_train=True)
        exe.backward(mx.nd.array(out_grad))
        assert_almost_equal(exe.grad_dict['affine'].asnumpy(), grad_est + grid_grad_npy, rtol=1e-2, atol=1e-5)

    # transform_type = warp
    test_case = [(12,21),(4,3),(6,12)]
    for target_shape in test_case:
        flow = mx.sym.Variable('flow')
        grid = mx.sym.GridGenerator(data=flow,transform_type='warp', target_shape=target_shape)
        exe = grid.simple_bind(ctx=default_context(), flow=(1,2)+target_shape, grad_req='write')
        # check forward
        exe.arg_dict['flow'][:] = np.ones((1,2)+target_shape)
        exe.forward(is_train=True)
        output = exe.outputs[0].asnumpy()
        output[0,0,:,:] = (output[0,0,:,:] + 1) * (target_shape[1] - 1) / 2.0
        output[0,1,:,:] = (output[0,1,:,:] + 1) * (target_shape[0] - 1) / 2.0
        xv, yv = np.meshgrid(np.arange(target_shape[0])+1, np.arange(target_shape[1])+1)
        assert_almost_equal(output[0,0], yv.T)
        assert_almost_equal(output[0,1], xv.T)
        # check backward
        out_grad = np.random.normal(size=(1,2)+target_shape)
        exe.backward(mx.nd.array(out_grad))
        grad_est = np.zeros((1,2)+target_shape)
        grad_est[0,0] = out_grad[0,0] / ((target_shape[1]-1.0) / 2.0)
        grad_est[0,1] = out_grad[0,1] / ((target_shape[0]-1.0) / 2.0)
        assert_almost_equal(exe.grad_dict['flow'].asnumpy(), grad_est, rtol=1e-3)
        # check addto
        exe_add = grid.simple_bind(ctx=default_context(), flow=(1, 2) + target_shape, grad_req='add')
        flow_grad_npy = np.random.normal(size=exe_add.grad_dict['flow'].shape)
        exe_add.arg_dict['flow'][:] = np.ones((1, 2) + target_shape)
        exe_add.grad_dict['flow'][:] = flow_grad_npy
        exe_add.forward(is_train=True)
        exe_add.backward(mx.nd.array(out_grad))
        assert_almost_equal(exe_add.grad_dict['flow'].asnumpy(), grad_est + flow_grad_npy, rtol=1e-3, atol=1e-5)


def test_bilinear_sampler():
    np.random.seed(1234)
    from math import floor

    def between(x, lowerbound, upperbound):
        return x>=lowerbound and x<=upperbound

    def bilinear_forward_numpy(data, grid):

        batchsize = data.shape[0]
        input_height = data.shape[2]
        input_width = data.shape[3]
        num_channel = data.shape[1]

        output_height = grid.shape[2]
        output_width = grid.shape[3]
        out = np.zeros(data.shape[:2] + grid.shape[2:], dtype=np.float32)

        for i in range(batchsize):
            for yout in range(output_height):
                for xout in range(output_width):

                    xcoord = np.float32((grid[i, 0, yout, xout] + 1) * (input_width-1) / 2.0)
                    ycoord = np.float32((grid[i, 1, yout, xout] + 1) * (input_height-1) / 2.0)

                    xInTopLeft = int(floor(xcoord))
                    xWeightTopLeft = np.float32(1-(xcoord - xInTopLeft))

                    yInTopLeft = int(floor(ycoord))
                    yWeightTopLeft = np.float32(1-(ycoord - yInTopLeft))

                    # interpolation
                    for channel in range(num_channel):

                        inTopLeft = data[i,channel,yInTopLeft, xInTopLeft] \
                            if between(xInTopLeft,0,input_width-1) and between(yInTopLeft,0,input_height-1) else 0.0
                        inTopRight = data[i,channel,yInTopLeft, xInTopLeft+1] \
                            if between(xInTopLeft+1,0,input_width-1) and between(yInTopLeft,0,input_height-1) else 0.0
                        inBottomLeft = data[i,channel,yInTopLeft+1, xInTopLeft] \
                            if between(xInTopLeft,0,input_width-1) and between(yInTopLeft+1,0,input_height-1) else 0.0
                        inBottomRight = data[i,channel,yInTopLeft+1, xInTopLeft+1] \
                            if between(xInTopLeft+1,0,input_width-1) and between(yInTopLeft+1,0,input_height-1) else 0.0

                        out[i,channel,yout,xout] = xWeightTopLeft * yWeightTopLeft * inTopLeft\
                                +  (1-xWeightTopLeft)*yWeightTopLeft * inTopRight\
                                +  xWeightTopLeft * (1-yWeightTopLeft) * inBottomLeft\
                            +(1-xWeightTopLeft) * (1-yWeightTopLeft) * inBottomRight
        return out

    def bilinear_backward_numpy(out_grad, data, grid):

        data_grad = np.zeros(data.shape, dtype=np.float32)
        grid_grad = np.zeros(grid.shape, dtype=np.float32)

        batchsize = data.shape[0]
        input_height = data.shape[2]
        input_width = data.shape[3]
        num_channel = data.shape[1]
        output_height = grid.shape[2]
        output_width = grid.shape[3]

        for i in range(batchsize):
            for yout in range(output_height):
                for xout in range(output_width):

                    top_left_y_gw = np.float32(0.0);
                    top_left_x_gw = np.float32(0.0);

                    xcoord = np.float32((grid[i, 0, yout, xout] + 1) * (input_width-1) / 2.0)
                    ycoord = np.float32((grid[i, 1, yout, xout] + 1) * (input_height-1) / 2.0)

                    xInTopLeft = int(floor(xcoord))
                    xWeightTopLeft = np.float32(1-(xcoord - xInTopLeft))

                    yInTopLeft = int(floor(ycoord))
                    yWeightTopLeft = np.float32(1-(ycoord - yInTopLeft))

                    topLeftDotProduct = np.float32(0)
                    topRightDotProduct = np.float32(0)
                    bottomLeftDotProduct = np.float32(0)
                    bottomRightDotProduct = np.float32(0)

                    for channel in range(num_channel):
                        # left top
                        if between(xInTopLeft,0,input_width-1) and between(yInTopLeft,0,input_height-1):
                            topLeftDotProduct += data[i,channel,yInTopLeft, xInTopLeft] * \
                                out_grad[i,channel,yout,xout]
                            data_grad[i, channel, yInTopLeft, xInTopLeft] += xWeightTopLeft * \
                                yWeightTopLeft * out_grad[i,channel,yout,xout]
                        # right top
                        if between(xInTopLeft+1,0,input_width-1) and between(yInTopLeft,0,input_height-1):
                            topRightDotProduct += data[i, channel, yInTopLeft,xInTopLeft+1] * \
                                out_grad[i, channel, yout,xout]
                            data_grad[i, channel,yInTopLeft, xInTopLeft+1] += (1-xWeightTopLeft) * \
                                yWeightTopLeft * out_grad[i,channel,yout,xout]
                        # left bottom
                        if between(xInTopLeft,0,input_width-1) and between(yInTopLeft+1,0,input_height-1):
                            bottomLeftDotProduct += data[i, channel,yInTopLeft+1, xInTopLeft] * \
                                out_grad[i,channel,yout,xout]
                            data_grad[i,channel,yInTopLeft+1,xInTopLeft]+=xWeightTopLeft * \
                                (1-yWeightTopLeft)* out_grad[i,channel,yout,xout]
                        # right bottom
                        if between(xInTopLeft+1,0,input_width-1) and between(yInTopLeft+1,0,input_height-1):
                            bottomRightDotProduct += data[i,channel,yInTopLeft+1, xInTopLeft+1] * \
                                out_grad[i,channel,yout,xout]
                            data_grad[i,channel,yInTopLeft+1,xInTopLeft+1]+= (1-xWeightTopLeft) * \
                                (1-yWeightTopLeft)*out_grad[i,channel,yout,xout]

                    yf = np.float32(-xWeightTopLeft * topLeftDotProduct + xWeightTopLeft*bottomLeftDotProduct - \
                        (1-xWeightTopLeft)* topRightDotProduct + (1-xWeightTopLeft)*bottomRightDotProduct)
                    xf = np.float32(-yWeightTopLeft * topLeftDotProduct + yWeightTopLeft*topRightDotProduct - \
                        (1-yWeightTopLeft)*bottomLeftDotProduct + (1-yWeightTopLeft)*bottomRightDotProduct)

                    grid_grad[i,0,yout,xout] = xf * (input_width-1) / 2.0
                    grid_grad[i,1,yout,xout] = yf * (input_height-1) / 2.0

        return data_grad, grid_grad

    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    net = mx.sym.BilinearSampler(data=data,grid=grid)

    test_case = [[(1,3,15,16),(1,2,10,10)],
                 [(1,6,7,16),(1,2,10,4)],
                 [(1,7,3,16),(1,2,8,11)],
                 [(1,9,50,50),(1,2,50,50)]]

    for ctx in [default_context()]:
        for item in test_case:
            data_shape, grid_shape = item
            exe = net.simple_bind(data=data_shape,grid=grid_shape,ctx=ctx,grad_req='write')
            # check forward
            exe.arg_dict['data'][:] = np.random.uniform(low=-0.1, high=0.1,size=data_shape).astype(np.float32)
            exe.arg_dict['grid'][:] = np.random.uniform(low=-2, high=2, size=grid_shape).astype(np.float32)
            exe.forward(is_train=True)
            out = bilinear_forward_numpy(exe.arg_dict['data'].asnumpy(), exe.arg_dict['grid'].asnumpy())
            assert_almost_equal(exe.outputs[0].asnumpy(), out, rtol=1e-3,atol=1e-5)

            # check backward
            out_grad = np.random.uniform(low=-0.01, high=0.01,size=data_shape[:2] + grid_shape[2:]).astype(np.float32)
            exe.backward(mx.nd.array(out_grad))
            data_grad, grid_grad = bilinear_backward_numpy(out_grad,exe.arg_dict['data'].asnumpy(),
                                                       exe.arg_dict['grid'].asnumpy())
            assert_almost_equal(exe.grad_dict['data'].asnumpy(), data_grad, rtol=1e-3, atol=1e-5)
            assert_almost_equal(exe.grad_dict['grid'].asnumpy(), grid_grad, rtol=1e-3, atol=1e-5)

            # check kAddTo
            exe_addto = net.simple_bind(data=data_shape, grid=grid_shape, ctx=ctx, grad_req='add')
            data_initial_grid = np.random.normal(size=exe_addto.grad_dict['data'].shape).astype(np.float32)
            grid_initial_grid = np.random.normal(size=exe_addto.grad_dict['grid'].shape).astype(np.float32)
            exe_addto.arg_dict['data'][:] = exe.arg_dict['data'][:]
            exe_addto.arg_dict['grid'][:] = exe.arg_dict['grid'][:]
            exe_addto.grad_dict['data'][:] = data_initial_grid
            exe_addto.grad_dict['grid'][:] = grid_initial_grid
            exe_addto.forward(is_train=True)
            exe_addto.backward(mx.nd.array(out_grad))
            assert_almost_equal(exe_addto.grad_dict['data'].asnumpy(), data_grad + data_initial_grid, rtol=1e-3,atol=1e-5)
            assert_almost_equal(exe_addto.grad_dict['grid'].asnumpy(), grid_grad + grid_initial_grid, rtol=1e-3,atol=1e-5)


def test_index2d():
    for _ in range(30):
        n = np.random.randint(1, 100)
        m = np.random.randint(1, 500)
        data = mx.random.uniform(-1, 1, shape=(n, m), ctx=default_context())
        x = mx.nd.array(np.random.randint(0, m, size=n), ctx=default_context(), dtype='int32')
        r = mx.nd.batch_take(data, x)
        assert_almost_equal(r.asnumpy(), data.asnumpy()[np.arange(n), x.asnumpy()])


def test_cast():
    for srctype in [np.int32, np.float32, np.float16]:
        for dsttype in [np.float32, np.int32, np.float16]:
            x = mx.sym.Variable('x', dtype=srctype)
            y = mx.sym.Cast(x, dtype=dsttype)
            exe = y.simple_bind(ctx=default_context(), x=(10, 10))
            assert exe.arg_arrays[0].dtype == srctype
            assert exe.outputs[0].dtype == dsttype
            X = np.random.uniform(-10, 10, size=(10, 10))
            exe.arg_arrays[0][:] = X
            exe.forward(is_train=True)
            exe.backward(mx.nd.array(X, dtype=dsttype, ctx=default_context()))
            assert_almost_equal(exe.outputs[0].asnumpy(), X.astype(srctype).astype(dsttype), rtol=1e-3)
            assert_almost_equal(exe.grad_arrays[0].asnumpy(), X.astype(dsttype).astype(srctype), rtol=1e-3)


def test_repeat():
    def test_repeat_forward():
        ndim_max = 6 # max number of dims of the ndarray
        size_max = 10 # max number of elements in each dim
        repeats = 3
        for ndim in range(1, ndim_max+1):
            shape = ()
            for i in range(0, ndim):
                shape += (np.random.randint(1, size_max+1), )
            a = np.random.random_sample(size=shape)
            aa = np.repeat(a, repeats)
            b = mx.nd.array(a, ctx=default_context())
            bb = mx.nd.repeat(b, repeats).asnumpy()
            assert_almost_equal(aa, bb)

            for axis in range(0, ndim):
                aa = np.repeat(a, repeats, axis)
                bb = mx.nd.repeat(b, repeats, axis).asnumpy()
                assert_almost_equal(aa, bb)

    def test_repeat_backward(axis):
        data = mx.sym.Variable('data')
        n1 = 3
        n2 = 4
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        arr_data = mx.nd.array(data_tmp)
        arr_grad = mx.nd.empty(shape)
        repeats = 2
        test = mx.sym.repeat(data, repeats=repeats, axis=axis)
        exe = test.bind(ctx=default_context(), args=[arr_data], args_grad=[arr_grad])
        npout_grad = np.random.randint(0, 10, n1 * n2 * repeats)
        if axis == 0:
            npout_grad = npout_grad.reshape(n1 * repeats, n2)
        elif axis == 1:
            npout_grad = npout_grad.reshape(n1, n2 * repeats)
        else:
            raise RuntimeError("Invalid axis value")
        out_grad = mx.nd.array(npout_grad)
        exe.backward(out_grad)

        expected_grad = np.zeros(shape)
        if axis == 0:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    k = i * repeats
                    expected_grad[i][j] = sum(npout_grad[k:k + repeats, j])
        elif axis == 1:
            for j in range(shape[1]):
                for i in range(shape[0]):
                    k = j * repeats
                    expected_grad[i][j] = sum(npout_grad[i, k:k + repeats])
        else:
            raise RuntimeError("Invalid axis value")

        assert_almost_equal(expected_grad, arr_grad.asnumpy(), rtol=1e-3)

    def test_repeat_numeric_gradient():
        data = mx.sym.Variable('data')
        n1 = 3
        n2 = 4
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        repeats = 2

        test = mx.sym.repeat(data, repeats=repeats, axis=0)
        check_numeric_gradient(test, [data_tmp], numeric_eps=1e-3, rtol=1e-2)

    test_repeat_forward()
    test_repeat_backward(axis=0)
    test_repeat_backward(axis=1)
    test_repeat_numeric_gradient()


def test_reverse():
    data = mx.symbol.Variable('data')
    shape = (5, 5, 5)
    data_tmp = np.random.uniform(-1, 1, shape)
    test = mx.sym.reverse(data, axis=[1, 2])
    grad = np.random.uniform(-1, 1, shape)
    check_numeric_gradient(test, [data_tmp], numeric_eps=2E-2)
    check_symbolic_forward(test, [data_tmp], [data_tmp[:, ::-1, ::-1]])
    check_symbolic_backward(test, [data_tmp], [grad], [grad[:, ::-1, ::-1]])


def test_tile():
    def test_normal_case():
        ndim_max = 3 # max number of dims of the ndarray
        size_max = 10 # max number of elements in each dim
        length_max = 3 # max length of reps
        rep_max = 10 # max number of tiling in each dim
        for ndim in range(ndim_max, ndim_max+1):
            shape = ()
            for i in range(0, ndim):
                shape += (np.random.randint(1, size_max+1), )
            a = np.random.randint(0, 100, shape)
            a = np.asarray(a, dtype=np.int32)
            if ndim == 0:
                a = np.array([])
            b = mx.nd.array(a, ctx=default_context(), dtype=a.dtype)

            reps_len = np.random.randint(0, length_max+1)
            reps_tuple = ()
            for i in range(1, reps_len):
                reps_tuple += (np.random.randint(0, rep_max), )
            reps_array = np.asarray(reps_tuple)

            a_tiled = np.tile(a, reps_array)
            b_tiled = mx.nd.tile(b, reps_tuple).asnumpy()
            assert same(a_tiled, b_tiled)

    def test_empty_tensor():
        shape = (2, 3, 0, 4)
        a = np.array([], dtype=np.int32).reshape(shape)
        b = mx.nd.array(a, ctx=default_context(), dtype=a.dtype)
        reps = (2, 4, 6)

        a_tiled = np.tile(a, reps)
        b_tiled = mx.nd.tile(b, reps).asnumpy()
        assert same(a_tiled, b_tiled)

    def test_empty_reps():
        a = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
        b = mx.nd.array(a, ctx=default_context(), dtype=a.dtype)
        a_tiled = np.tile(a, ())
        b_tiled = mx.nd.tile(b, ()).asnumpy()
        assert same(a_tiled, b_tiled)

    def test_zero_reps():
        a = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
        b = mx.nd.array(a, ctx=default_context(), dtype=a.dtype)
        reps = (2, 0, 4, 5)
        a_tiled = np.tile(a, reps)
        b_tiled = mx.nd.tile(b, reps).asnumpy()
        assert same(a_tiled, b_tiled)

    def test_tile_backward():
        data = mx.sym.Variable('data')
        n1 = 2
        n2 = 2
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        arr_data = mx.nd.array(data_tmp)
        arr_grad = mx.nd.empty(shape)
        reps1 = 2
        reps2 = 2
        reps = (reps1, reps2)
        test = mx.sym.tile(data, reps=reps)
        exe = test.bind(ctx=mx.context.Context.default_ctx, args=[arr_data], args_grad=[arr_grad])
        npout_grad = np.random.randint(0, 10, n1 * n2 * reps1 * reps2).reshape(n1 * reps1, n2 * reps2)
        out_grad = mx.nd.array(npout_grad)
        exe.backward(out_grad)

        expected_grad = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                expected_grad[i][j] += sum(sum(npout_grad[i:(n1 * reps1):reps1, j:(n2 * reps2):reps2]))

        assert_almost_equal(expected_grad, arr_grad.asnumpy(), rtol=1e-3)

    def test_tile_numeric_gradient():
        data = mx.sym.Variable('data')
        n1 = 2
        n2 = 2
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        reps1 = 2
        reps2 = 2
        reps = (reps1, reps2)
        test = mx.sym.tile(data, reps=reps)
        check_numeric_gradient(test, [data_tmp], numeric_eps=1e-2, rtol=1e-2)

    test_normal_case()
    test_empty_tensor()
    test_empty_reps()
    test_zero_reps()
    test_tile_backward()
    test_tile_numeric_gradient()


def test_one_hot():
    def test_normal_case(index_type=np.int32):
        ndim_max = 6
        dim_size_max = 20
        depth = int(dim_size_max / 2)
        on_value = 1
        off_value = 0
        for ndim in range(1, ndim_max+1):
            shape = ()
            for i in range(1, ndim+1):
                shape += (np.random.randint(1, dim_size_max+1), )
            indices = np.random.randint(-dim_size_max, dim_size_max+1,
                                        size=np.prod(shape)).reshape(shape)
            mx_one_hot_array = mx.nd.one_hot(
                mx.nd.array(indices, ctx=default_context(), dtype=index_type),
                depth=depth, dtype=np.int32)
            expected_array = np.zeros((np.prod(shape), depth), dtype=np.int32)
            expected_array[:] = off_value
            indices_1d = indices.flatten()
            row = 0
            for idx in indices_1d:
                if 0 <= idx < depth:
                    expected_array[row, idx] = on_value
                row += 1
            expected_array = expected_array.reshape(shape + (depth, ))
            one_hot_array = mx_one_hot_array.asnumpy()
            assert same(expected_array, one_hot_array)

    def test_empty_indices():
        shape = (2, 0, 9, 3)
        indices = np.array([]).reshape(shape)
        depth = 10
        mx_one_hot_array = mx.nd.one_hot(
            mx.nd.array(indices, ctx=default_context(), dtype=np.int32),
            depth=depth, dtype=np.int32).asnumpy()
        expected_array = np.array([], dtype=np.int32).reshape(shape + (depth, ))
        assert same(expected_array, mx_one_hot_array)

    def test_zero_depth():
        shape = (2, 4, 9, 3)
        indices = np.ones(shape)
        depth = 0
        mx_one_hot_array = mx.nd.one_hot(
            mx.nd.array(indices, ctx=default_context(), dtype=np.int32),
            depth=depth, dtype=np.int32).asnumpy()
        expected_array = np.array([], dtype=np.int32).reshape(shape + (depth, ))
        assert same(expected_array, mx_one_hot_array)

    test_normal_case(index_type=np.int32)
    test_normal_case(index_type=np.float64)
    test_normal_case(index_type=np.float32)
    test_normal_case(index_type=np.float16)
    test_empty_indices()
    test_zero_depth()


def test_where():
    def get_forward_expected_output(condition, x, y):
        original_shape = x.shape
        out = np.zeros(original_shape)
        if condition.shape == x.shape:
            for index, c in np.ndenumerate(condition):
                if c != 0:
                    out[index] = x[index]
                else:
                    out[index] = y[index]
        elif condition.shape == (x.shape[0], ):
            s = x.shape
            m = s[0]
            n = int(np.prod(s)/s[0])
            x2d = x.reshape((m, n))
            y2d = y.reshape((m, n))
            out = out.reshape((m, n))
            for i in range(0, m):
                if condition[i] != 0:
                    for j in range(0, n):
                        out[i, j] = x2d[i, j]
                else:
                    for j in range(0, n):
                        out[i, j] = y2d[i, j]
        else:
            raise RuntimeError("Invalid condition shape for where op")

        out = out.reshape(original_shape)
        return out

    def get_forward_inputs_same_shape(shape):
        condition_np = np.random.randint(0, 2, np.prod(shape)).reshape(shape)
        x_np = np.random.randint(1, 6, np.prod(shape)).reshape(shape)
        y_np = np.random.randint(7, 11, np.prod(shape)).reshape(shape)
        return condition_np, x_np, y_np

    def get_forward_inputs_condition_vector(shape):
        condition_np = np.random.randint(0, 2, shape[0])
        x_np = np.random.randint(1, 6, np.prod(shape)).reshape(shape)
        y_np = np.random.randint(7, 11, np.prod(shape)).reshape(shape)
        return condition_np, x_np, y_np

    def get_backward_input(shape):
        return np.random.randint(20, 30, np.prod(shape)).reshape(shape)

    def get_backward_expected_outputs(grad_in, condition):
        shape = grad_in.shape
        grad_cond = np.zeros(condition.shape)
        grad_x = np.empty(shape)
        grad_y = np.empty(shape)

        for index, c in np.ndenumerate(condition):
            if 0 != c:
                grad_x[index] = grad_in[index]
                grad_y[index] = 0
            else:
                grad_x[index] = 0
                grad_y[index] = grad_in[index]

        return grad_cond, grad_x, grad_y

    def test_where_helper(shape, same_shape):
        if same_shape:
            condition_np, x_np, y_np = get_forward_inputs_same_shape(shape)
        else:
            condition_np, x_np, y_np = get_forward_inputs_condition_vector(shape)

        out_expected = get_forward_expected_output(condition_np, x_np, y_np)

        grad_in_np = get_backward_input(shape)
        grad_expected_cond, grad_expected_x, grad_expected_y\
            = get_backward_expected_outputs(grad_in_np, condition_np)

        condition = mx.sym.Variable('condition')
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        grad_in_mx = mx.nd.array(grad_in_np, dtype=np.int32)
        where_sym = mx.sym.where(condition, x, y)

        # test req='write'
        where_exe_write = where_sym.simple_bind(ctx=default_context(),
                                                condition=condition_np.shape,
                                                x=x_np.shape, y=y_np.shape,
                                                grad_req='write')
        # test forward req='write'
        outputs = where_exe_write.forward(is_train=True, condition=condition_np,
                                          x=x_np, y=y_np)
        assert same(outputs[0].asnumpy(), out_expected)
        # test backward req='write'
        where_exe_write.backward(grad_in_mx)
        assert same(where_exe_write.grad_dict['x'].asnumpy(), grad_expected_x)
        assert same(where_exe_write.grad_dict['y'].asnumpy(), grad_expected_y)
        assert same(where_exe_write.grad_dict['condition'].asnumpy(), grad_expected_cond)

        # test req='add'
        x_grad_init = np.random.randint(30, 40, np.prod(shape)).reshape(shape)
        y_grad_init = np.random.randint(40, 50, np.prod(shape)).reshape(shape)
        where_exe_add = where_sym.simple_bind(ctx=default_context(),
                                              condition=condition_np.shape,
                                              x=x_np.shape, y=y_np.shape,
                                              grad_req='add')
        where_exe_add.grad_dict['x'][:] = x_grad_init
        where_exe_add.grad_dict['y'][:] = y_grad_init
        # test forward req='add'
        outputs = where_exe_add.forward(is_train=True, condition=condition_np, x=x_np, y=y_np)
        assert same(outputs[0].asnumpy(), out_expected)
        # test backward req='add'
        where_exe_add.backward(grad_in_mx)
        x_ograd = where_exe_add.grad_dict['x'].asnumpy()
        y_ograd = where_exe_add.grad_dict['y'].asnumpy()
        assert same(x_ograd, grad_expected_x+x_grad_init)
        assert same(y_ograd, grad_expected_y+y_grad_init)

    def test_where_numeric_gradient(shape, same_shape):
        condition = mx.sym.Variable('condition')
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        where_sym = mx.sym.where(condition, x, y)
        if same_shape:
            condition_np, x_np, y_np = get_forward_inputs_same_shape(shape)
        else:
            condition_np, x_np, y_np = get_forward_inputs_condition_vector(shape)
        check_numeric_gradient(where_sym, [condition_np, x_np, y_np], grad_nodes=['x', 'y'])

    test_where_helper((5, 9), True)
    test_where_helper((5, 9), False)
    test_where_helper((5, 7, 9), True)
    test_where_helper((5, 7, 9), False)
    test_where_helper((10, 8, 15, 3), True)
    test_where_helper((10, 8, 15, 3), False)
    test_where_numeric_gradient((5, 9), True)
    test_where_numeric_gradient((5, 9), False)
    test_where_numeric_gradient((5, 7, 9), True)
    test_where_numeric_gradient((5, 7, 9), False)


def test_new_softmax():
    for ndim in range(1, 5):
        for _ in range(5):
            shape = np.random.randint(1, 5, size=ndim)
            axis = np.random.randint(0, ndim)
            data = np.random.uniform(-2, 2, size=shape)
            sym = mx.sym.softmax(axis=axis)
            check_symbolic_forward(sym, [data], [np_softmax(data, axis=axis)])
            check_numeric_gradient(sym, [data], rtol=0.05, atol=1e-3)


def test_log_softmax():
    for ndim in range(1, 5):
        for _ in range(5):
            shape = np.random.randint(1, 5, size=ndim)
            axis = np.random.randint(0, ndim)
            data = np.random.uniform(-2, 2, size=shape)
            sym = mx.sym.log_softmax(axis=axis-ndim)
            check_symbolic_forward(sym, [data], [np.log(np_softmax(data, axis=axis)+1e-20)])
            check_numeric_gradient(sym, [data], rtol=0.05, atol=1e-3)


def test_pick():
    def test_pick_helper(index_type=np.int32):
        for _ in range(100):
            ndim = np.random.randint(1, 5)
            bshape = np.random.randint(1, 10, size=ndim)
            axis = np.random.randint(0, ndim)
            sshape = bshape.copy()
            sshape[axis] = 1
            data = np.random.uniform(-1, 1, size=bshape)
            index = np.random.randint(0, bshape[axis], size=sshape)
            exp = []
            for i in range(ndim):
                if i == axis:
                    exp.append(index)
                else:
                    ishape = [1 for _ in range(ndim)]
                    ishape[i] = bshape[i]
                    exp.append(np.arange(bshape[i]).reshape(ishape))
            expected = data[exp]
            data = mx.nd.array(data, dtype='float32')
            index = mx.nd.array(index, dtype=index_type)
            out = mx.nd.pick(data, index, axis=axis, keepdims=True)
            assert_almost_equal(out.asnumpy(), expected)

            data_holder = data
            index_holder = index
            data = mx.sym.Variable('data')
            index = mx.sym.Variable('index')
            sym = mx.sym.pick(data, index, axis=axis, keepdims=True)
            check_numeric_gradient(sym, [data_holder, index_holder], grad_nodes=['data'])

    test_pick_helper(np.int32)
    test_pick_helper(np.float32)


def check_ctc_loss(acts, labels, loss_truth):
    in_var = mx.sym.Variable('input')
    labels_var = mx.sym.Variable('labels')
    ctc = mx.sym.contrib.ctc_loss(in_var, labels_var)
    acts_nd = mx.nd.array(acts, ctx=default_context())
    labels_nd = mx.nd.array(labels, ctx=default_context())
    exe = ctc.bind(ctx=default_context(), args=[acts_nd, labels_nd])
    # test forward without grad calc
    exe.forward(is_train=True)
    outTest = exe.outputs[0]
    # test forward without grad calc
    exe.forward(is_train=False)
    outTrain = exe.outputs[0]
    # make sure losses calculated with both modes are the same
    assert_almost_equal(outTest.asnumpy(), outTrain.asnumpy())
    # test against ground truth, if available
    if loss_truth is not None:
        assert_almost_equal(outTest.asnumpy(), loss_truth)
    # test grad
    check_numeric_gradient(ctc, [acts, labels], grad_nodes=['input'], rtol=0.05, atol=1e-3)


def test_ctc_loss():
    # Test 1: check that batches are same + check against Torch WarpCTC
    acts = np.array([
        [[1.2, 3.4, 1.2, -0.1, -2.34], [1.2, 3.4, 1.2, -0.1, -2.34]],
        [[0.1, 0.2, 0.3, 0.22, 0.123], [0.1, 0.2, 0.3, 0.22, 0.123]],
        [[-15, -14, -13, -12, -11], [-15, -14, -13, -12, -11]]],
                    dtype=np.float32)
    labels = np.array([[2, 3, 0], [2, 3, 0]])
    true_loss = np.array([4.04789, 4.04789], dtype=np.float32) # from Torch
    check_ctc_loss(acts, labels, true_loss)
    # Test 2:
    acts2 = np.array([
        [[-5, -4, -3, -2, -1], [1.2, 3.4, 1.2, -0.1, -2.34]],
        [[-10, -9, -8, -7, -6], [0.1, 0.2, 0.3, 0.22, 0.123]],
        [[-15, -14, -13, -12, -11], [-15, -14.2, -13.5, -12.2, -11.22]]], dtype=np.float32)
    labels2 = np.array([[2, 3, 1], [2, 0, 0]], dtype=np.float32)
    true_loss = np.array([7.3557, 5.4091], dtype=np.float32) # from Torch
    check_ctc_loss(acts2, labels2, true_loss)


def test_ctc_loss_grad():
    def check_ctc_loss_grad(blank_label): # from tf
        vocab_size = 5
        max_label_len = 5
        padding_mask = -1+ (blank_label=='first')

        targets_0 = [0, 1, 2, 1, 0]
        loss_log_prob_0 = -3.34211
        input_prob_matrix_0 = np.asarray(
            [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
            dtype=np.float32)
        gradient_log_prob_0 = np.asarray(
            [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
            dtype=np.float32)

        targets_1 = [0, 1, 1, 0]
        loss_log_prob_1 = -5.42262
        input_prob_matrix_1 = np.asarray(
            [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
             [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
            dtype=np.float32)
        gradient_log_prob_1 = np.asarray(
            [[-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544],
             [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
            dtype=np.float32)

        inputs = [
            np.vstack(
                [input_prob_matrix_0[t, :], input_prob_matrix_1[t, :]])
            for t in range(5)
        ] + 2 * [np.nan * np.ones((2, vocab_size+1), np.float32)]
        inputs = np.log(np.asarray(inputs, dtype=np.float32))

        grad_truth = np.array([
            np.vstack(
                [gradient_log_prob_0[t, :], gradient_log_prob_1[t, :]])
            for t in range(5)
        ] + 2 * [np.zeros((2, vocab_size+1), np.float32)])

        if blank_label == 'first':
            inputs = np.roll(inputs, 1, axis=2)
            grad_truth = np.roll(grad_truth, 1, axis=2)

        labels = (np.asarray([x + [padding_mask]*(max_label_len-len(x))
                             for x in [targets_0, targets_1]])+(blank_label == 'first'))

        seq_lens = np.array([5, 5], dtype=np.int32)
        label_lens = np.array([5, 4], dtype=np.int32)
        loss_truth = np.array([-loss_log_prob_0, -loss_log_prob_1], np.float32)

        with default_context():
            data = mx.nd.array(inputs)
            label = mx.nd.array(labels)
            data.attach_grad()
            with mx.autograd.record():
                l = mx.contrib.ndarray.CTCLoss(data, label,
                                               use_data_lengths=True,
                                               use_label_lengths=True,
                                               data_lengths=mx.nd.array(seq_lens),
                                               label_lengths=mx.nd.array(label_lens),
                                               blank_label=blank_label)
                l.backward()
            assert_almost_equal(l.asnumpy(), loss_truth, atol=1e-5, rtol=1e-5)
            assert_almost_equal(data.grad.asnumpy(), grad_truth, atol=1e-5, rtol=1e-5)

    check_ctc_loss_grad('first')
    check_ctc_loss_grad('last')


def test_quantization_op():
    min0 = mx.nd.array([0.0])
    max0 = mx.nd.array([1.0])
    a  = mx.nd.array([[0.1392, 0.5928], [0.6027, 0.8579]])
    qa, min1, max1 = mx.nd.contrib.quantize(a, min0, max0, out_type='uint8')
    a_ = mx.nd.contrib.dequantize(qa, min1, max1, out_type='float32')

    qa_real = mx.nd.array([[35, 151], [154, 219]])
    a_real  = mx.nd.array([[0.13725491, 0.59215689], [0.60392159, 0.8588236]])

    assert same(qa.asnumpy(), qa_real.asnumpy())
    assert same(a_.asnumpy(),  a_real.asnumpy())


def test_reciprocal_op():
    data_tmp = np.random.rand(3, 4) * 10 - 5
    # Avoid possible division by 0 errors
    data_tmp[data_tmp == 0] = 1.0
    data = mx.symbol.Variable('data')
    test = mx.sym.reciprocal(data)

    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [np.reciprocal(data_tmp)])

def test_cbrt_op():
    data_tmp = np.random.rand(3, 4) * 10 - 5
    data = mx.symbol.Variable('data')
    test = mx.sym.cbrt(data)

    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [np.cbrt(data_tmp)])

def test_rcbrt_op():
    data_tmp = np.random.rand(3, 4) * 10 - 5
    # Avoid possible division by 0 errors
    data_tmp[data_tmp == 0] = 1.0
    data = mx.symbol.Variable('data')
    test = mx.sym.rcbrt(data)

    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [1/np.cbrt(data_tmp)])

def test_custom_op():
    class Sqr(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            self.assign(out_data[0], req[0], in_data[0]*in_data[0])
            aux[0][:] = 1

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], 2*in_data[0]*out_grad[0])
            assert (aux[0].asnumpy() == 1).all()

    @mx.operator.register("sqr")
    class SqrProp(mx.operator.CustomOpProp):
        def __init__(self):
            super(SqrProp, self).__init__(need_top_grad=True)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def list_auxiliary_states(self):
            return ['aux']

        def infer_shape(self, in_shape):
            return in_shape, [in_shape[0]], [in_shape[0]]

        def infer_type(self, in_type):
            return in_type, [in_type[0]], [in_type[0]]

        def create_operator(self, ctx, shapes, dtypes):
            return Sqr()

    data = mx.symbol.Variable('data')
    aux = mx.symbol.Variable('aux')
    op = mx.symbol.Custom(data=data, aux=aux, name='sqr', op_type='sqr')
    x = mx.nd.array(np.random.uniform(-1, 1, size=(4, 10)))
    aux = mx.nd.zeros_like(x)
    check_numeric_gradient(op, [x], [aux])

    data = mx.symbol.cast(data, dtype='float64')
    op = mx.symbol.cast(op, dtype='float32')
    x = mx.nd.array(np.random.uniform(-1, 1, size=(4, 10)))
    aux = mx.nd.zeros_like(x)
    check_numeric_gradient(op, [x], [aux])

    x.attach_grad()
    with mx.contrib.autograd.train_section():
        y = mx.nd.Custom(x, aux, op_type='sqr')
        y.backward()


def test_psroipooling():
    for num_rois in [1, 2]:
        for num_classes, num_group in itertools.product([2, 3], [2, 3]):
            for image_height, image_width in itertools.product([168, 224], [168, 224]):
                for grad_nodes in [['im_data']]:
                    spatial_scale = 0.0625
                    feat_height = np.int(image_height * spatial_scale)
                    feat_width = np.int(image_width * spatial_scale)
                    im_data = np.random.rand(1, num_classes*num_group*num_group, feat_height, feat_width)
                    rois_data = np.zeros([num_rois, 5])
                    rois_data[:, [1,3]] = np.sort(np.random.rand(num_rois, 2)*(image_width-1))
                    rois_data[:, [2,4]] = np.sort(np.random.rand(num_rois, 2)*(image_height-1))

                    im_data_var = mx.symbol.Variable(name="im_data")
                    rois_data_var = mx.symbol.Variable(name="rois_data")
                    op = mx.sym.contrib.PSROIPooling(data=im_data_var, rois=rois_data_var, spatial_scale=spatial_scale,
                                                     group_size=num_group, pooled_size=num_group,
                                                     output_dim=num_classes, name='test_op')
                    rtol, atol = 1e-2, 1e-3
                    # By now we only have gpu implementation
                    if mx.Context.default_ctx.device_type == 'gpu':
                        check_numeric_gradient(op, [im_data, rois_data], rtol=rtol, atol=atol,
                                               grad_nodes=grad_nodes, ctx=mx.gpu(0))


def test_deformable_convolution():
    for num_batch in [1, 2]:
        for num_channel_data, num_deformable_group in itertools.product([4, 8], [1, 2]):
            for input_height, input_width in itertools.product([5, 6], [5, 6]):
                for dilate in [(1, 1), (2, 2)]:
                    for grad_nodes in [['im_data'], ['offset_data'], ['weight']]:
                        output_height = input_height
                        output_width = input_width
                        im_data = np.random.rand(num_batch, num_channel_data, input_height, input_width)
                        offset_data = \
                            np.random.rand(num_batch, num_deformable_group * 3 * 3 * 2, output_height, output_width)\
                            * 0.8 + 0.1

                        weight = np.random.normal(0, 0.001, (num_channel_data, num_channel_data, 3, 3))
                        bias = np.zeros(num_channel_data)

                        im_data_var = mx.symbol.Variable(name="im_data")
                        offset_data_var = mx.symbol.Variable(name="offset_data")
                        weight_var = mx.symbol.Variable(name="weight")
                        bias_var = mx.symbol.Variable(name="bias")
                        op = mx.sym.contrib.DeformableConvolution(name='test_op', data=im_data_var,
                                                                  offset=offset_data_var,
                                                                  weight=weight_var, bias=bias_var,
                                                                  num_filter=num_channel_data, pad=dilate,
                                                                  kernel=(3, 3), stride=(1, 1), dilate=dilate,
                                                                  num_deformable_group=num_deformable_group)
                        if grad_nodes[0] == 'offset_data':
                            # wider tolerance needed for coordinate differential
                            rtol, atol = 1.0, 1e-2
                        else:
                            rtol, atol = 0.05, 1e-3
                        # By now we only have gpu implementation
                        if mx.Context.default_ctx.device_type == 'gpu':
                            check_numeric_gradient(op, [im_data, offset_data, weight, bias], rtol=rtol, atol=atol,
                                                   grad_nodes=grad_nodes, ctx=mx.gpu(0))


def test_deformable_psroipooling():
    for num_rois in [1, 2]:
        for num_classes, num_group in itertools.product([2, 3], [2, 3]):
            for image_height, image_width in itertools.product([168, 224], [168, 224]):
                for grad_nodes in [['im_data'], ['offset_data']]:
                    spatial_scale = 0.0625
                    feat_height = np.int(image_height * spatial_scale)
                    feat_width = np.int(image_width * spatial_scale)
                    im_data = np.random.rand(1, num_classes*num_group*num_group, feat_height, feat_width)
                    rois_data = np.zeros([num_rois, 5])
                    rois_data[:, [1,3]] = np.sort(np.random.rand(num_rois, 2)*(image_width-1))
                    rois_data[:, [2,4]] = np.sort(np.random.rand(num_rois, 2)*(image_height-1))
                    offset_data = np.random.rand(num_rois, 2*num_classes, num_group, num_group) * 0.1

                    im_data_var = mx.symbol.Variable(name="im_data")
                    rois_data_var = mx.symbol.Variable(name="rois_data")
                    offset_data_var = mx.symbol.Variable(name="offset_data")
                    op = mx.sym.contrib.DeformablePSROIPooling(data=im_data_var, rois=rois_data_var,
                                                               trans=offset_data_var, spatial_scale=spatial_scale,
                                                               sample_per_part=4, group_size=num_group,
                                                               pooled_size=num_group, output_dim=num_classes,
                                                               trans_std=0.1, no_trans=False, name='test_op')
                    if grad_nodes[0] == 'offset_data':
                        # wider tolerance needed for coordinate differential
                        rtol, atol = 1.0, 1e-2
                    else:
                        rtol, atol = 1e-2, 1e-3
                    # By now we only have gpu implementation
                    if mx.Context.default_ctx.device_type == 'gpu':
                        check_numeric_gradient(op, [im_data, rois_data, offset_data], rtol=rtol, atol=atol,
                                               grad_nodes=grad_nodes, ctx=mx.gpu(0))


# Helper functions for test_laop

def _make_symm_symbol(a, ndims):
    assert ndims >= 2
    tr_shape = list(range(ndims))
    tr_shape[-1] = ndims-2
    tr_shape[-2] = ndims-1
    tr_shape = tuple(tr_shape)
    return 0.5 * (a + mx.sym.transpose(a, axes=tr_shape))

def _make_lower_triangle_symm(a, ndims, m, dtype=np.float32):
    assert ndims >= 2
    # The last two dimensions must both be m
    # Create mask for lower triangle and diagonal
    index = mx.sym.arange(start=0, stop=m, step=1, dtype=np.int32)
    lt_mask = mx.sym.one_hot(index, depth=m, dtype=dtype)
    for j in range(1, m):
        part1 = mx.sym.zeros(shape=(j, m), dtype=dtype)
        index = mx.sym.arange(start=0, stop=m-j, step=1, dtype=np.int32)
        part2 = mx.sym.one_hot(index, depth=m, dtype=dtype)
        lt_mask = lt_mask + mx.sym.concat(*[part1, part2], dim=0)
    shp = tuple([1]*(ndims-2) + [m, m])
    lt_mask = mx.sym.reshape(lt_mask, shape=shp)
    return mx.sym.broadcast_mul(a, lt_mask)

def test_laop():
    dtype = np.float64
    rtol_fw = 1e-7
    atol_fw = 1e-9
    num_eps = 1e-6
    rtol_bw = 1e-5
    atol_bw = 1e-6
    # enable numerical checking of gradients
    grad_check = 1

    data1 = mx.symbol.Variable('data1')
    data2 = mx.symbol.Variable('data2')
    data3 = mx.symbol.Variable('data3')

    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                               atol=atol_bw, dtype=dtype)
    rep_3x = lambda a, m, n :\
        np.reshape(np.tile(np.array(a).flatten(), 3), (3, 1, m, n))

    # Test gemm separately from other la-operators.
    shape1 = (2, 3)
    shape2 = (3, 2)
    shape3 = (3, 3)
    shape4 = (2, 2)
    #Ensure that ithis tests don't get changed by other calls to random.
    np.random.seed(42)
    data_in1 = np.random.uniform(1, 10, shape1)
    data_in2 = np.random.uniform(1, 10, shape2)
    data_in3 = np.random.uniform(1, 10, shape3)
    data_in4 = np.random.uniform(1, 10, shape4)
    # Check all transpositions of gemm operator.
    data_in1_t = np.transpose(data_in1)
    data_in2_t = np.transpose(data_in2)
    res_gemm = 4. * np.dot(data_in1, data_in2) + 7. * data_in4
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.)
    check_fw(test_gemm, [data_in1, data_in2, data_in4], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2, data_in4])
    res_gemm = 4. * np.dot(data_in1_t, data_in2_t) + 7. * data_in3
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.,
                                   transpose_a=True, transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in2, data_in3], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2, data_in3])
    res_gemm = 4. * np.dot(data_in1_t, data_in1) + 7. * data_in3
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.,
                                   transpose_a=True)
    check_fw(test_gemm, [data_in1, data_in1, data_in3], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1, data_in3])
    res_gemm = 4. * np.dot(data_in1, data_in1_t) + 7. * data_in4
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.,
                                   transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in1, data_in4], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1, data_in4])

    # Check batch of gemm.
    a = rep_3x(data_in1, 2, 3)
    b = rep_3x(data_in2, 3, 2)
    c = rep_3x(data_in4, 2, 2)
    r = 4. * np.dot(data_in1, data_in2) + 7. * data_in4
    r = rep_3x(r, 2, 2)
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.)
    check_fw(test_gemm, [a, b, c], [r])
    if grad_check == 1:
        check_grad(test_gemm, [a, b, c])

    # Check gemm2 operator same way as gemm.
    res_gemm = 4. * np.dot(data_in1, data_in2)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4.)
    check_fw(test_gemm, [data_in1, data_in2], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2])
    res_gemm = 4. * np.dot(data_in1_t, data_in2_t)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., transpose_a=True,
                                    transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in2], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2])
    res_gemm = 4. * np.dot(data_in1_t, data_in1)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., transpose_a=True)
    check_fw(test_gemm, [data_in1, data_in1], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1])
    res_gemm = 4. * np.dot(data_in1, data_in1_t)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in1], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1])

    # Check batch of gemm2.
    a = rep_3x(data_in1, 2, 3)
    b = rep_3x(data_in2, 3, 2)
    r = rep_3x(4. * np.dot(data_in1, data_in2), 2, 2)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4.)
    check_fw(test_gemm, [a, b], [r])
    if grad_check == 1:
        check_grad(test_gemm, [a, b])

    # Now test all the other operators.

    # Tests with trivial 1x1 matrices.
    shape = (4, 4, 1, 1)
    data_in = np.random.uniform(1, 10, shape)
    # test potrf
    # Note: Have to symmetrize input, for gradient test to work
    res_potrf = np.sqrt(data_in)
    test_potrf = mx.sym.linalg.potrf(data1)
    check_fw(test_potrf, [data_in], [res_potrf])
    if grad_check == 1:
        check_grad(test_potrf, [data_in])
    # test potri
    ones = mx.nd.ones(shape).asnumpy()
    res_potri = np.divide(ones, data_in * data_in)
    test_potri = mx.sym.linalg.potri(data1)
    check_fw(test_potri, [data_in], [res_potri])
    if grad_check == 1:
        check_grad(test_potri, [data_in])
    # test trsm
    trian_in = data_in * 7.
    test_trsm = mx.sym.linalg.trsm(data1, data2, alpha=7.)
    check_fw(test_trsm, [trian_in, data_in], [ones])
    if grad_check == 1:
        check_grad(test_trsm, [trian_in,data_in])
    # test trmm
    trian_in = np.divide(ones, trian_in)
    test_trmm = mx.sym.linalg.trmm(data1, data2, alpha=7., transpose=True,
                                   rightside=True)
    check_fw(test_trmm, [trian_in, data_in], [ones])
    if grad_check == 1:
        check_grad(test_trmm, [trian_in, data_in])
    # test sumlogdiag
    res_sumlogdiag = np.reshape(np.log(data_in), (4, 4))
    test_sumlogdiag = mx.sym.linalg.sumlogdiag(data1)
    check_fw(test_sumlogdiag, [data_in], [res_sumlogdiag])
    if grad_check == 1:
        check_grad(test_sumlogdiag, [data_in])

    # more elaborate example of Cholesky factorization
    matrix = np.array([[9., 3., -6., 12.],
                       [3., 26., -7., -11.],
                       [-6., -7., 9., 7.],
                       [12., -11., 7., 65.]])
    trian  = np.array([[3., 0., 0., 0.],
                       [1., 5., 0., 0.],
                       [-2., -1., 2., 0.],
                       [4., -3., 6., 2.]])
    pow    = np.array([[2., 1., 1., 1.],
                       [1., 4., 1., 1.],
                       [1., 1., 8., 1.],
                       [1., 1., 1., 16.]])
    inv    = np.array([[8.95/3., 0.05/3., 2.65, -2.5/3.],
                       [0.05/3., 0.05, 0.05, 0.],
                       [2.65, 0.05, 2.5, -0.75],
                       [-2.5/3., 0., -0.75, 0.25]])
    ident  = np.eye(4)

    # test potrf
    test_potrf = mx.sym.linalg.potrf(_make_symm_symbol(data1, ndims=4))
    a = rep_3x(matrix, 4, 4)
    r = rep_3x(trian, 4, 4)
    check_fw(test_potrf, [a], [r])
    if grad_check == 1:
        check_grad(test_potrf, [a])

    #test potri
    data1_ltri = _make_lower_triangle_symm(
        data1, ndims=4, m=4, dtype=dtype)
    test_potri = mx.sym.linalg.potri(data1_ltri)
    a = rep_3x(trian, 4, 4)
    r = rep_3x(inv, 4, 4)
    check_fw(test_potri, [a], [r])
    if grad_check == 1:
        check_grad(test_potri, [a])

    # test trsm
    test_trsm = mx.sym.linalg.trsm(data1_ltri, data2, alpha=7.)
    a = rep_3x(trian, 4, 4)
    b = rep_3x(matrix, 4, 4)
    r = rep_3x(7. * np.transpose(trian), 4, 4)
    check_fw(test_trsm, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trsm, [a, b])

    test_trsm2 = mx.sym.linalg.trsm(
        data1_ltri, data2, alpha=-2., rightside=True, transpose=True)
    r = rep_3x(-2. * trian, 4, 4)
    check_fw(test_trsm2, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trsm2, [a, b])

    test_trsm3 = mx.sym.linalg.trsm(
        data1_ltri, data2, alpha=0.5, transpose=True)
    b = rep_3x(np.transpose(trian), 4, 4)
    r = rep_3x(0.5 * ident, 4, 4)
    check_fw(test_trsm3, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trsm3, [a, b])

    test_trsm4 = mx.sym.linalg.trsm(
        data1_ltri, data2, alpha=-0.5, rightside=True)
    b = rep_3x(trian, 4, 4)
    r = rep_3x(-0.5 * ident, 4, 4)
    check_fw(test_trsm4, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trsm4, [a, b])

    # test trmm
    test_trmm = mx.sym.linalg.trmm(
        data1_ltri, data2, alpha=7., transpose=True, rightside=True)
    a = rep_3x(trian, 4, 4)
    b = rep_3x(matrix, 4, 4)
    r = rep_3x(7. * np.dot(matrix, trian.T), 4, 4)
    check_fw(test_trmm, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trmm, [a, b])

    test_trmm2 = mx.sym.linalg.trmm(data1_ltri, data2, alpha=-2.)
    r = rep_3x(-2. * np.dot(trian, matrix), 4, 4)
    check_fw(test_trmm2, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trmm2, [a, b])

    test_trmm3 = mx.sym.linalg.trmm(data1_ltri, data2, rightside=True)
    r = rep_3x(np.dot(matrix, trian), 4, 4)
    check_fw(test_trmm3, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trmm3, [a, b])

    test_trmm4 = mx.sym.linalg.trmm(
        data1_ltri, data2, alpha=1.2, transpose=True)
    r = rep_3x(1.2 * np.dot(trian.T, matrix), 4, 4)
    check_fw(test_trmm4, [a, b], [r])
    if grad_check == 1:
        check_grad(test_trmm4, [a, b])

    # test sumlogdiag
    a = rep_3x(pow, 4, 4)
    r = np.reshape(np.tile(10. * np.log(np.array([2.])), 3), (3,))
    check_fw(test_sumlogdiag, [a], [r])
    if grad_check == 1:
        check_grad(test_sumlogdiag, [a])


# Tests for operators linalg.syrk, linalg.gelqf

def _gelqf_combined_symbol(a):
    q, l = mx.sym.linalg.gelqf(a)
    q_qt = mx.sym.linalg.syrk(q, transpose=False, alpha=1., name='Q_times_Qt')
    l_q = mx.sym.linalg.trmm(l, q, alpha=1., name='L_times_Q')
    return mx.sym.Group([q_qt, l_q])

# NOTE: If we leave the unused output dangling, things break if dtype=np.float64. Namely, the
# backward gradient for the unused output is of dtype np.float32 then.
# ==> Very annoying!
def _gelqf_first_output(a):
    q, l = mx.sym.linalg.gelqf(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(l), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(q, bogus_scal)

def _gelqf_second_output(a):
    q, l = mx.sym.linalg.gelqf(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(q), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(l, bogus_scal)

def _syevd_combined_symbol(a):
    u, lam = mx.sym.linalg.syevd(a)
    u_ut = mx.sym.linalg.syrk(u, transpose=False, alpha=1., name='U_times_Ut')
    lam_u = mx.sym.broadcast_mul(mx.sym.reshape(lam, shape=(-2, 1)), u)
    ut_lam_u = mx.sym.linalg.gemm2(u, lam_u, alpha=1., transpose_a=True,
                                   transpose_b=False, name='Ut_L_U')
    return mx.sym.Group([u_ut, ut_lam_u])

def test_laop_2():
    np.random.seed(1896893923)
    dtype = np.float64
    rtol_fw = 1e-7
    atol_fw = 1e-9
    num_eps = 1e-6
    rtol_bw = 1e-5
    atol_bw = 1e-6
    # enable numerical checking of gradients
    grad_check = 1

    data1 = mx.symbol.Variable('data1')

    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                               atol=atol_bw, dtype=dtype)
    rep_3x = lambda a, m, n :\
        np.reshape(np.tile(np.array(a).flatten(), 3), (3, 1, m, n))

    # Tests for linalg.syrk
    mnalpha_lst = [(2, 3, 1.), (5, 3, -2.), (1, 6, 5.), (3, 3, 0.5), (4, 1, 10.), (1, 1, 1.)]
    for m, n, alpha in mnalpha_lst:
        #print('syrk: m={}, n={}, alpha={}'.format(m, n, alpha))
        data_in1 = np.random.uniform(1, 10, (m, n))
        res_syrk1 = alpha * np.dot(data_in1, data_in1.T)
        test_syrk1 = mx.sym.linalg.syrk(data1, transpose=False, alpha=alpha)
        check_fw(test_syrk1, [data_in1], [res_syrk1])
        if grad_check == 1:
            check_grad(test_syrk1, [data_in1])
        res_syrk2 = alpha * np.dot(data_in1.T, data_in1)
        test_syrk2 = mx.sym.linalg.syrk(data1, transpose=True, alpha=alpha)
        check_fw(test_syrk2, [data_in1], [res_syrk2])
        if grad_check == 1:
            check_grad(test_syrk2, [data_in1])
        # Batch mode (3x the same thing)
        a_batch = rep_3x(data_in1, m, n)
        r1_batch = rep_3x(res_syrk1, m, m)
        check_fw(test_syrk1, [a_batch], [r1_batch])
        if grad_check == 1:
            check_grad(test_syrk1, [a_batch])
        r2_batch = rep_3x(res_syrk2, n, n)
        check_fw(test_syrk2, [a_batch], [r2_batch])
        if grad_check == 1:
            check_grad(test_syrk2, [a_batch])

    # Tests for linalg.gelqf
    # Currently disabled on GPU as they need cuda8
    # and MxNet builds use cuda 7.5
    if not (default_context() == mx.cpu()):
        return

    test_gelqf2 = _gelqf_combined_symbol(data1)  # Outputs (dot(Q, Q.T), dot(L, Q))
    test_gelqf_q = _gelqf_first_output(data1)  # Output Q (L is not dangling)
    test_gelqf_l = _gelqf_second_output(data1)  # Output L (Q is not dangling)
    mn_lst = [(4, 4), (1, 1), (5, 20), (1, 10), (15, 50)]
    for m, n in mn_lst:
        #print('gelqf: m={}, n={}'.format(m, n))
        data_in1 = np.random.normal(0., 10., (m, n))
        res_eye = np.eye(m)
        res_a = data_in1
        check_fw(test_gelqf2, [data_in1], [res_eye, res_a])
        if grad_check == 1:
            # A => Q
            check_grad(test_gelqf_q, [data_in1])
            # A => L
            check_grad(test_gelqf_l, [data_in1])
        # Batch mode (3x the same thing)
        a_batch = rep_3x(data_in1, m, n)
        reye_batch = rep_3x(res_eye, m, m)
        ra_batch = a_batch
        check_fw(test_gelqf2, [a_batch], [reye_batch, ra_batch])
        if grad_check == 1:
            # A => Q
            check_grad(test_gelqf_q, [a_batch])
            # A => L
            check_grad(test_gelqf_l, [a_batch])


# Tests for operator linalg.syevd

def _syevd_first_output(a):
    u, lam = mx.sym.linalg.syevd(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(lam), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(u, bogus_scal)

def _syevd_second_output(a):
    u, lam = mx.sym.linalg.syevd(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(u), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(lam, bogus_scal)

def _syevd_forward(a):
    lam, ut = np.linalg.eig(a)
    ind = np.argsort(lam)
    lam = lam[ind]
    u = ut[:, ind].T
    for i in range(0, a.shape[0]):
        _syevd_forw_eigvec_sign(u[i])
    return u, lam

def _syevd_forw_eigvec_sign(v):
    ind = np.argmax(np.abs(v))
    if v[ind] < 0.:
        v[:] = -v

def _syevd_backward(grad_u, grad_l, u, l):
    n = l.size
    assert grad_l.size == n
    assert grad_u.shape == (n, n)
    assert u.shape == (n, n)
    temp = np.dot(grad_u, u.T)
    temp2 = np.diag(grad_l)
    for i in range(1, n):
        for j in range(0, i):
            denom = 2. * (l[i] - l[j])
            elem = (temp[i, j] - temp[j, i])/denom
            temp2[i, j] = elem
            temp2[j, i] = elem
    temp3 = np.dot(u.T, temp2)
    return np.dot(temp3, u)

def test_laop_3():
    # Currently disabled on GPU as syevd needs cuda8
    # and MxNet builds use cuda 7.5
    if not (default_context() == mx.cpu()):
        return

    np.random.seed(1896893923)
    dtype = np.float64
    rtol_fw = 1e-6
    atol_fw = 1e-6
    num_eps = 1e-4
    rtol_bw = 1e-2
    atol_bw = 1e-2
    # enable numerical checking of gradients
    grad_check = 1

    data1 = mx.symbol.Variable('data1')
    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                               atol=atol_bw, dtype=dtype)
    rep_3x = lambda a, m, n :\
        np.reshape(np.tile(np.array(a).flatten(), 3), (3, 1, m, n))
    check_bw = lambda sym, location, out_grads, expected :\
        check_symbolic_backward(sym, location, out_grads, expected,
                                rtol=rtol_fw, atol=atol_fw, dtype=dtype)

    # Tests for linalg.syevd
    test_syevd2 = _syevd_combined_symbol(data1)  # Outputs (U U^T, U^T (diag L) U)
    data1_s2 = _make_symm_symbol(data1, ndims=2)
    test_syevd_u_2 = _syevd_first_output(data1_s2)
    test_syevd_l_2 = _syevd_second_output(data1_s2)
    data1_s4 = _make_symm_symbol(data1, ndims=4)
    test_syevd_u_4 = _syevd_first_output(data1_s4)
    test_syevd_l_4 = _syevd_second_output(data1_s4)
    n_lst = [4, 1, 2, 10, 14]
    for n in n_lst:
        #print('\n** syevd: n={}'.format(n))
        data_in1 = np.random.normal(0., 10., (n, n))
        data_in1 = 0.5 * (data_in1 + data_in1.T)
        res_eye = np.eye(n)
        res_a = data_in1
        check_fw(test_syevd2, [data_in1], [res_eye, res_a])
        # Check backward
        grad_u = np.random.normal(0., 2., (n, n))
        grad_l = np.random.normal(0., 2., (n,))
        bw_u, bw_l = _syevd_forward(data_in1)
        grad_a = _syevd_backward(grad_u, grad_l, bw_u, bw_l)
        check_bw(mx.sym.linalg.syevd(data1), [data_in1], [grad_u, grad_l], [grad_a])
        if grad_check == 1:
            # A => U
            check_grad(test_syevd_u_2, [data_in1])
            # A => L
            check_grad(test_syevd_l_2, [data_in1])
        # Batch mode (3x the same thing)
        a_batch = rep_3x(data_in1, n, n)
        reye_batch = rep_3x(res_eye, n, n)
        ra_batch = a_batch
        check_fw(test_syevd2, [a_batch], [reye_batch, ra_batch])
        if grad_check == 1:
            # A => U
            check_grad(test_syevd_u_4, [a_batch])
            # A => L
            check_grad(test_syevd_l_4, [a_batch])


def test_laop_4():
    # Currently disabled on GPU as syevd needs cuda8
    # and MxNet builds use cuda 7.5
    if not (default_context() == mx.cpu()):
        return

    np.random.seed(1896893923)
    rtol_fw = 1e-6
    atol_fw = 1e-6

    data1 = mx.symbol.Variable('data1')

    check_fw = lambda sym, location, expected, dtype :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)

    a_np = np.array([[1., 2.], [2., 4.]])
    u_np = np.array([[0.89442718, -0.44721359], [0.44721359, 0.89442718]])
    l_np = np.array([0., 5.])
    test_syevd = mx.sym.linalg.syevd(data1)
    # float64
    #print('float64')
    check_fw(test_syevd, [a_np], [u_np, l_np], np.float64)
    # float32
    #print('float32')
    check_fw(test_syevd, [a_np], [u_np, l_np], np.float32)


def test_stack():
    for _ in range(100):
        ndim = random.randint(1, 5)
        axis = random.randint(0, ndim)
        if random.randint(0, 1):
            axis = axis - ndim - 1
        nin = random.randint(1, 3)
        dshape = [random.randint(1, 5) for _ in range(ndim)]
        inputs = [np.random.uniform(size=dshape) for _ in range(nin)]
        output = np.stack(inputs, axis=axis)
        sym_ins = [mx.sym.var('x%d'%i) for i in range(nin)]
        out = mx.sym.stack(*sym_ins, axis=axis)
        check_symbolic_forward(out, inputs, [output])
        check_numeric_gradient(out, inputs)


def test_dropout():
    def zero_count(array, ratio):
        zeros = 0
        for i in array:
            if i == 0:
                zeros += 1
            elif math.isnan(i):
                assert ratio == 1  # Only valid for ratio = 1
                zeros += 1
        return zeros

    def check_correctness(executor, input, ratio):
        input = input.ravel()
        output = executor.outputs[0].asnumpy().ravel()
        input_sum = np.sum(input)
        output_sum = np.sum(output)

        # Make sure input zeroes are none (test data setup check)
        assert zero_count(input, ratio) == 0

        # count number of zeroes in output
        output_zeroes = zero_count(output, ratio)

        # Hopefully should be within ratio/2 %
        error = abs(output_sum - input_sum) / input_sum
        if ratio == 1.0:
            assert output_zeroes == len(input)
        elif ratio > 0.2:
            assert output_zeroes > 0
            assert error < (ratio/2)
        elif ratio == 0:
            assert output_zeroes == 0

    def check_dropout_ratio(ratio, shape):
        # test dropout
        x = mx.sym.var('data')
        y = mx.sym.Dropout(x, p=ratio)
        exe = y.simple_bind(ctx=default_context(), data=shape)

        if ratio == 1:
            max_value = float('nan')
        else:
            max_value = 1 if ratio == 0 else 1/ratio

        if ratio == 1:
            min_value = float('nan')
        else:
            min_value = 1 if ratio == 0 else 0

        exe.arg_arrays[0][:] = 1
        exe.forward(is_train=True)
        if not math.isnan(max_value):
            assert exe.outputs[0].asnumpy().max() > 0
        else:
            assert math.isnan(exe.outputs[0].asnumpy().max())
        if not math.isnan(min_value):
            assert exe.outputs[0].asnumpy().min() == min_value
        else:
            assert math.isnan(exe.outputs[0].asnumpy().min())

        check_correctness(exe, exe.arg_arrays[0].asnumpy(), ratio)

        if ratio == 0.5:
            exe.backward([mx.nd.ones(shape)])
            assert (exe.grad_arrays[0].asnumpy() == exe.outputs[0].asnumpy()).all()

            exe.forward(is_train=False)
            assert (exe.outputs[0].asnumpy() == exe.arg_arrays[0].asnumpy()).all()
            exe.backward([mx.nd.ones(shape)], is_train=False)
            assert (exe.grad_arrays[0].asnumpy() == exe.arg_arrays[0].asnumpy()).all()

            # test permanent dropout
            x = mx.sym.var('data')
            y = mx.sym.Dropout(x, p=ratio, mode='always')
            exe = y.simple_bind(ctx=default_context(), data=shape)

            exe.arg_arrays[0][:] = 1
            exe.forward(is_train=True)
            assert exe.outputs[0].asnumpy().max() == max_value
            assert exe.outputs[0].asnumpy().min() == min_value
            exe.backward([mx.nd.ones(shape)])
            assert (exe.grad_arrays[0].asnumpy() == exe.outputs[0].asnumpy()).all()

            exe.forward(is_train=False)
            assert exe.outputs[0].asnumpy().max() == max_value
            assert exe.outputs[0].asnumpy().min() == min_value
            exe.backward([mx.nd.ones(shape)], is_train=False)
            assert (exe.grad_arrays[0].asnumpy() == exe.outputs[0].asnumpy()).all()

    shape = (100, 100)
    check_dropout_ratio(0.5, shape)
    check_dropout_ratio(0.0, shape)
    check_dropout_ratio(1.0, shape)
    check_dropout_ratio(0.75, shape)
    check_dropout_ratio(0.25, shape)

def test_scatter_gather_nd():
    def check(data, idx):
        data.attach_grad()
        with mx.autograd.record():
            y = mx.nd.gather_nd(data, idx)
            y.backward(y)
        npidx = tuple(i.asnumpy() for i in idx)
        assert (data.asnumpy()[npidx] == y.asnumpy()).all()
        npdata = np.zeros_like(data.asnumpy())
        npdata[npidx] = y.asnumpy()
        assert (npdata == data.grad.asnumpy()).all()
        assert (mx.nd._internal._backward_gather_nd(y, idx, shape=data.shape).asnumpy() == data.grad.asnumpy()).all()
    for dtype in ['int32', 'int64', 'float16', 'float32', 'float64']:
        data = mx.nd.arange(360, dtype=dtype).reshape((3,4,5,6))
        idx = mx.nd.array([[1,1,2], [3, 3, 0], [3,2,1]], dtype='int32')
        check(data, idx)

        idx = mx.nd.array([[1,1,2], [3,3,0], [3,2,1], [5,2,4]], dtype='int32')

        check(data, idx)

        data = mx.nd.array([2, 3, 0], dtype=dtype)
        idx = mx.nd.array([[1, 1, 0], [0, 1, 0]], dtype='int32')
        assert (mx.nd.scatter_nd(data, idx, shape=(2, 2)).asnumpy() == [[0, 0], [2, 3]]).all()

        data = mx.nd.array([2, 3, 0], dtype=dtype)
        idx = mx.nd.array([[1, 1, 0], [1, 1, 0]], dtype='int32')
        assert (mx.nd._internal._backward_gather_nd(data, idx, shape=(2, 2)).asnumpy() == [[0, 0], [0, 5]]).all()
        data_npy = np.random.randint(0, 10, (100,))
        data = mx.nd.array(data_npy, dtype=dtype)
        idx = mx.nd.zeros(shape=(1, 100), dtype='int32')
        assert (mx.nd._internal._backward_gather_nd(data, idx, shape=(1,)).asscalar() == data_npy.sum())
        if dtype == 'int64':
            data = mx.nd.array([2123162361283621, -31231236374787,
                                -112372937128970, -1378278798172378], dtype=dtype)
            idx = mx.nd.array([[0, 0, 0, 0]], dtype='int32')
            assert (mx.nd._internal._backward_gather_nd(data, idx, shape=(1,)).asscalar() == data.asnumpy().sum())

def compare_forw_backw_unary_op(
        name, forward_mxnet_call, forward_numpy_call,
        backward_numpy_call, shape, input_low, input_high, rtol, atol,
        dtype=np.float32):
    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol,
                               atol=atol, dtype=dtype)
    check_bw = lambda sym, location, out_grads, expected :\
        check_symbolic_backward(sym, location, out_grads, expected,
                                rtol=rtol, atol=atol, dtype=dtype)
    op_name = 'unary_op={}, dtype={}'.format(name, dtype)
    data = mx.symbol.Variable(op_name + '_data', dtype=dtype)
    # Comparison: Forward expression
    data_np = np.random.uniform(input_low, input_high, shape).astype(dtype)
    res_np = forward_numpy_call(data_np)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data), mx.sym.zeros_like(data),
        name=op_name)
    check_fw(op_ex, [data_np], [res_np])
    # Comparison: Backward expression
    res_grad = np.random.uniform(-2.0, 2.0, shape).astype(dtype)
    data_grad = backward_numpy_call(data_np) * res_grad
    check_bw(op_ex, [data_np], [res_grad], [data_grad])

def finite_diff_unary_op(
        name, forward_mxnet_call, shape, input_low, input_high, rtol, atol,
        num_eps):
    # Finite difference tests are done in float64
    dtype = np.float64
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol,
                               atol=atol, dtype=dtype)
    data_np = np.random.uniform(input_low, input_high, shape).astype(dtype)
    data = mx.symbol.Variable('data', dtype=dtype)
    op_name = 'unary_op={}, dtype={}'.format(name, dtype)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data), mx.sym.zeros_like(data),
        name=op_name)
    check_grad(op_ex, [data_np])

def np_smooth_l1(x, sigma):
    issq = 1. / sigma / sigma
    absx = np.abs(x)
    temp = x * sigma
    return np.where(absx < issq, 0.5 * (temp ** 2), absx - 0.5 * issq)

def np_smooth_l1_grad(x, sigma):
    ssq = sigma * sigma
    return np.where(np.abs(x) < 1. / ssq, x * ssq, np.sign(x))

# Tests for unary operators (basic mathematical functions):
# - Forward: Comparison to NumPy (several dtype)
# - Backward: Comparison to NumPy (several dtype)
# - Finite difference tests (only dtype = float64)
def test_unary_math_operators():
    np.random.seed(192837465)
    have_scipy = True
    try:
        from scipy import special as scipy_special
    except:
        print("Could not import scipy. Skipping unit tests for special functions")
        have_scipy = False
    shape=(9, 10)
    dtype_l = [np.float64, np.float32, np.float16]
    rtol_l = [1e-7, 1e-6, 1e-2]
    rtol_less_l = [1e-6, 1e-5, 1e-2]
    atol_l = [1e-7, 1e-6, 1e-2]
    atol_less_l = [1e-6, 1e-5, 1e-2]
    rtol_fd = 1e-5
    atol_fd = 1e-6
    num_eps = 1e-6
    unary_ops = {
        'arccos' : [lambda x: mx.sym.arccos(x),
                    lambda x: np.arccos(x),
                    lambda x: -1. / np.sqrt(1. - x ** 2.),
                    -0.95, 0.95],
        'arccosh': [lambda x: mx.sym.arccosh(x),
                    lambda x: np.arccosh(x),
                    lambda x: 1. / np.sqrt(x ** 2 - 1.),
                    1.05, 10.0],
        'arcsin': [lambda x: mx.sym.arcsin(x),
                   lambda x: np.arcsin(x),
                   lambda x: 1. / np.sqrt(1. - x ** 2),
                   -0.95, 0.95],
        'arcsinh': [lambda x: mx.sym.arcsinh(x),
                    lambda x: np.arcsinh(x),
                    lambda x: 1. / np.sqrt(x**2 + 1.),
                    -5.0, 5.0],
        'arctan': [lambda x: mx.sym.arctan(x),
                   lambda x: np.arctan(x),
                   lambda x: 1. / (x ** 2. + 1.),
                   -5.0, 5.0],
        'arctanh': [lambda x: mx.sym.arctanh(x),
                    lambda x: np.arctanh(x),
                    lambda x: 1. / (1. - x ** 2),
                    -0.95, 0.95],
        'cbrt': [lambda x: mx.sym.cbrt(x),
                 lambda x: np.cbrt(x),
                 lambda x: 1. / (3. * np.cbrt(x) ** 2),
                 -10.0, 10.0],
        'cos': [lambda x: mx.sym.cos(x),
                lambda x: np.cos(x),
                lambda x: -np.sin(x),
                -5.0, 5.0],
        'cosh': [lambda x: mx.sym.cosh(x),
                 lambda x: np.cosh(x),
                 lambda x: np.sinh(x),
                 -2.0, 2.0],
        'exp': [lambda x: mx.sym.exp(x),
                lambda x: np.exp(x),
                lambda x: np.exp(x),
                -4.0, 4.0],
        'expm1': [lambda x: mx.sym.expm1(x),
                  lambda x: np.expm1(x),
                  lambda x: np.exp(x),
                  -0.1, 0.1],
        'log': [lambda x: mx.sym.log(x),
                lambda x: np.log(x),
                lambda x: 1. / x,
                0.01, 100.0],
        'log10': [lambda x: mx.sym.log10(x),
                lambda x: np.log10(x),
                lambda x: 1. / (x * np.log(10.)),
                0.01, 100.0],
        'log2': [lambda x: mx.sym.log2(x),
                lambda x: np.log2(x),
                lambda x: 1. / (x * np.log(2.)),
                0.01, 100.0],
        'log1p': [lambda x: mx.sym.log1p(x),
                  lambda x: np.log1p(x),
                  lambda x: 1. / (1. + x),
                  -0.1, 0.1],
        'rcbrt': [lambda x: mx.sym.rcbrt(x),
                  lambda x: 1. / np.cbrt(x),
                  lambda x: -1. / (3. * x * np.cbrt(x)),
                  0.01, 100.0],
        'reciprocal': [lambda x: mx.sym.reciprocal(x),
                       lambda x: 1. / x,
                       lambda x: -1. / (x ** 2),
                       0.01, 100.0],
        'relu': [lambda x: mx.sym.relu(x),
                 lambda x: np.maximum(x, 0.),
                 lambda x: 1. * (x > 0.),
                 -5.0, 5.0],
        'rsqrt': [lambda x: mx.sym.rsqrt(x),
                  lambda x: 1. / np.sqrt(x),
                  lambda x: -0.5 / (x * np.sqrt(x)),
                  0.01, 100.0],
        'sigmoid': [lambda x: mx.sym.sigmoid(x),
                    lambda x: 1. / (np.exp(-x) + 1.),
                    lambda x: 1. / (np.exp(-x) + 1.) / (np.exp(x) + 1.),
                    -3.0, 3.0],
        'sin': [lambda x: mx.sym.sin(x),
                lambda x: np.sin(x),
                lambda x: np.cos(x),
                -5.0, 5.0],
        'sinh': [lambda x: mx.sym.sinh(x),
                 lambda x: np.sinh(x),
                 lambda x: np.cosh(x),
                 -2.0, 2.0],
        'sqrt': [lambda x: mx.sym.sqrt(x),
                 lambda x: np.sqrt(x),
                 lambda x: 0.5 / np.sqrt(x),
                 0.01, 100.0],
        'tan': [lambda x: mx.sym.tan(x),
                lambda x: np.tan(x),
                lambda x: np.tan(x) ** 2 + 1.,
                -1.5, 1.5],
        'tanh': [lambda x: mx.sym.tanh(x),
                 lambda x: np.tanh(x),
                 lambda x: 1. - np.tanh(x) ** 2,
                 -4.0, 4.0],
        'smooth_l1_sig1': [lambda x: mx.sym.smooth_l1(x, scalar=1.),
                           lambda x: np_smooth_l1(x, 1.),
                           lambda x: np_smooth_l1_grad(x, 1.),
                           -2.0, 2.0],
        'smooth_l1_sig2': [lambda x: mx.sym.smooth_l1(x, scalar=2.),
                           lambda x: np_smooth_l1(x, 2.),
                           lambda x: np_smooth_l1_grad(x, 2.),
                           -1.0, 1.0]
    }
    if have_scipy:
        unary_ops['gamma'] = [lambda x: mx.sym.gamma(x),
                              lambda x: scipy_special.gamma(x),
                              lambda x: scipy_special.gamma(x) * scipy_special.psi(x),
                              0.01, 5.0]
        unary_ops['gammaln'] = [lambda x: mx.sym.gammaln(x),
                                lambda x: scipy_special.gammaln(x),
                                lambda x: scipy_special.psi(x),
                                0.01, 20.0]
    # Loop over operators
    for name, op in unary_ops.items():
        # Loop over dtype's
        for ind in range(len(dtype_l)):
            dtype = dtype_l[ind]
            if name == 'gammaln' or name == 'gamma':
                rtol = rtol_less_l[ind]
                atol = atol_less_l[ind]
            else:
                rtol = rtol_l[ind]
                atol = atol_l[ind]
            compare_forw_backw_unary_op(
                name, op[0], op[1], op[2], shape, op[3], op[4], rtol, atol,
                dtype)
        # Finite difference testing
        finite_diff_unary_op(
            name, op[0], shape, op[3], op[4], rtol_fd, atol_fd, num_eps)

def compare_forw_backw_binary_op(
        name, forward_mxnet_call, forward_numpy_call,
        backward1_numpy_call, backward2_numpy_call, shape, input1_low,
        input1_high, input2_low, input2_high, rtol, atol, dtype=np.float32):
    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol,
                               atol=atol, dtype=dtype)
    check_bw = lambda sym, location, out_grads, expected :\
        check_symbolic_backward(sym, location, out_grads, expected,
                                rtol=rtol, atol=atol, dtype=dtype)
    op_name = 'binary_op={}, dtype={}'.format(name, dtype)
    data1 = mx.symbol.Variable(op_name + '_data1', dtype=dtype)
    data2 = mx.symbol.Variable(op_name + '_data2', dtype=dtype)
    # Comparison: Forward expression
    data1_np = np.random.uniform(input1_low, input1_high, shape).astype(dtype)
    data2_np = np.random.uniform(input2_low, input2_high, shape).astype(dtype)
    res_np = forward_numpy_call(data1_np, data2_np)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data1, data2), mx.sym.zeros_like(data1),
        name=op_name)
    check_fw(op_ex, [data1_np, data2_np], [res_np])
    # Comparison: Backward expression
    res_grad = np.random.uniform(-2.0, 2.0, shape).astype(dtype)
    data1_grad = backward1_numpy_call(data1_np, data2_np) * res_grad
    data2_grad = backward2_numpy_call(data1_np, data2_np) * res_grad
    check_bw(op_ex, [data1_np, data2_np], [res_grad], [data1_grad, data2_grad])

def finite_diff_binary_op(
        name, forward_mxnet_call, shape, input1_low, input1_high, input2_low,
        input2_high, rtol, atol, num_eps):
    # Finite difference tests are done in float64
    dtype = np.float64
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol,
                               atol=atol, dtype=dtype)
    data1_np = np.random.uniform(input1_low, input1_high, shape).astype(dtype)
    data2_np = np.random.uniform(input2_low, input2_high, shape).astype(dtype)
    data1 = mx.symbol.Variable('data1', dtype=dtype)
    data2 = mx.symbol.Variable('data2', dtype=dtype)
    op_name = 'binary_op={}, dtype={}'.format(name, dtype)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data1, data2), mx.sym.zeros_like(data1),
        name=op_name)
    check_grad(op_ex, [data1_np, data2_np])

# Tests for unary operators (basic mathematical functions):
# - Forward: Comparison to NumPy (several dtype)
# - Backward: Comparison to NumPy (several dtype)
# - Finite difference tests (only dtype = float64)
def test_binary_math_operators():
    np.random.seed(192837465)
    shape=(9, 10)
    dtype_l = [np.float64, np.float32, np.float16]
    rtol_l = [1e-7, 1e-6, 1e-2]
    atol_l = [1e-7, 1e-6, 1e-2]
    rtol_fd = 1e-5
    atol_fd = 1e-6
    num_eps = 1e-6
    binary_ops = {
        'hypot' : [lambda x, y: mx.sym.hypot(x, y),
                   lambda x, y: np.hypot(x, y),
                   lambda x, y: x / np.hypot(x, y),
                   lambda x, y: y / np.hypot(x, y),
                    -5.0, 5.0, -5.0, 5.0],
        'pow': [lambda x, y: mx.sym.pow(x, y),
                lambda x, y: np.power(x, y),
                lambda x, y: np.power(x, y - 1.) * y,
                lambda x, y: np.power(x, y) * np.log(x),
                0.2, 5.0, -4.0, 4.0]
    }
    # Loop over operators
    for name, op in binary_ops.items():
        # Loop over dtype's
        for ind in range(len(dtype_l)):
            dtype = dtype_l[ind]
            compare_forw_backw_binary_op(
                name, op[0], op[1], op[2], op[3], shape, op[4], op[5], op[6],
                op[7], rtol_l[ind], atol_l[ind], dtype)
        # Finite difference testing
        finite_diff_binary_op(
            name, op[0], shape, op[4], op[5], op[6], op[7], rtol_fd, atol_fd,
            num_eps)


def test_softmax():
    check_softmax_with_shape((3, 4), default_context(), preserve_shape=False)
    check_softmax_with_shape((3, 4), default_context(), preserve_shape=True)
    check_softmax_with_shape((3, 4, 2), default_context(), preserve_shape=True)
    check_softmax_grad(default_context())
    check_smoothed_softmax_grad(default_context())


def test_slice():
    def test_slice_forward_backward(a, index):
        a_np = a.asnumpy()
        begin = []
        end = []
        step = []
        for slice_i in index:
            begin.append(slice_i.start)
            end.append(slice_i.stop)
            step.append(slice_i.step)
        b = mx.nd.slice(a, begin=begin, end=end, step=step)
        b_np = a_np[index]
        assert same(b.asnumpy(), b_np)

        data = mx.sym.Variable('data')
        slice_sym = mx.sym.slice(data, begin=begin, end=end, step=step)
        expected_in_grad = np.zeros_like(a_np)
        expected_in_grad[index] = b_np
        check_symbolic_backward(slice_sym, [a_np], [b_np], [expected_in_grad])

    shape = (16, 14, 17, 20)
    arr = mx.nd.arange(np.prod(shape)).reshape(shape=shape)
    index_list = [(slice(None),), (slice(None), slice(None)), (slice(1, 10),), (slice(1, 10), slice(3, 9)),
                  (slice(1, 10), slice(2, 5), slice(3, 6), slice(7, 10)),
                  (slice(1, 10, 2), slice(2, 9, 3), slice(3, 6, 5), slice(7, 10, 2)),
                  (slice(None, None, -1), slice(None, None, -1), slice(None, None, -1)),
                  (slice(10, 0, -2), slice(5, 2, -1), slice(7, None, 3), slice(None, 12, 4))]
    for index in index_list:
        test_slice_forward_backward(arr, index)

    # check numeric gradient
    in_data = np.arange(36).reshape(2, 2, 3, 3)
    data = mx.sym.Variable('data')
    slice_sym = mx.sym.slice(data, begin=[0, None], end=[1, None], step=[2, -1])
    check_numeric_gradient(slice_sym, [in_data])


def test_float16_min_max():
    """Test for issue: https://github.com/apache/incubator-mxnet/issues/9007"""
    a = mx.nd.array([np.finfo('float16').min, np.finfo('float16').max], dtype='float16')
    assert a.dtype == np.float16
    assert np.finfo('float16').min == mx.nd.min(a).asscalar()
    assert np.finfo('float16').max == mx.nd.max(a).asscalar()


def test_squeeze_op():
    def check_squeeze_op(shape, axis=None):
        data = mx.nd.random.uniform(low=-10.0, high=10.0, shape=shape)
        if axis is None:
            out = mx.nd.squeeze(data).asnumpy()
            out_expected = np.squeeze(data.asnumpy())
        else:
            out = mx.nd.squeeze(data, axis=axis).asnumpy()
            out_expected = np.squeeze(data.asnumpy(), axis=axis)
        if out.shape == (1,):  # as an exception (1, 1, 1) will be squeezed to (1,)
            out_expected = np.squeeze(data.asnumpy(), axis=tuple([i for i in range(1, len(shape))]))
        assert same(out, out_expected)

    # check forward
    check_squeeze_op((1, 5, 1, 3, 1), 0)
    check_squeeze_op((1, 5, 1, 3, 1), 2)
    check_squeeze_op((1, 5, 1, 3, 1), 4)
    check_squeeze_op((1, 5, 1, 3, 1), (0, 4))
    check_squeeze_op((1, 5, 1, 3, 1), (0, 2, 4))
    check_squeeze_op((1, 5, 1, 3, 1))
    check_squeeze_op((1, 1, 1, 1))

    # check gradient
    data = mx.symbol.Variable('data')
    shape = (1, 2, 1, 3, 1)
    data_tmp = np.ones(shape)
    test = mx.sym.squeeze(data)
    check_numeric_gradient(test, [data_tmp])
    test = mx.sym.squeeze(data, axis=2)
    check_numeric_gradient(test, [data_tmp])
    test = mx.sym.squeeze(data, axis=(2, 4))
    check_numeric_gradient(test, [data_tmp])


if __name__ == '__main__':
    import nose
    nose.runmodule()
