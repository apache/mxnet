# pylint: skip-file
import numpy as np
import mxnet as mx
import random
from numpy.testing import assert_allclose
from check_utils import (check_numeric_gradient, check_symbolic_backward,
                         check_symbolic_forward, reldiff, _np_reduce)


def same(a, b):
    return np.sum(a != b) == 0

def np_softmax(x):
    x = x - np.max(x, axis=1).reshape(x.shape[0], 1)
    x = np.exp(x)
    x /= np.sum(x, axis=1).reshape(x.shape[0], 1)
    return x


def check_elementwise_sum_with_shape(shape, n):
    # forward
    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.ElementWiseSum(*inputs, name='esum')
    arr = [mx.nd.empty(shape) for i in range(n)]
    arr_grad = [mx.nd.empty(shape) for i in range(n)]
    for i in range(n):
        arr[i][:] = np.random.uniform(-10, 10, shape)
    exec1 = out.bind(mx.Context('cpu'),
                     args=arr,
                     args_grad=arr_grad)
    out1 = exec1.outputs[0].asnumpy()
    exec1.forward()
    out1 = exec1.outputs[0].asnumpy()
    out = sum(a.asnumpy() for a  in arr)
    assert reldiff(out, out1) < 1e-6
    out_grad = mx.nd.empty(shape)
    out_grad[:] = np.random.uniform(-10, 10, shape)
    # backward
    exec1.backward([out_grad])
    for a in arr_grad:
        assert same(a.asnumpy(), out_grad.asnumpy())

def test_elementwise_sum():
    np.random.seed(0)
    nrepeat = 2
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
            check_elementwise_sum_with_shape(shape, np.random.randint(1, 8))

def check_slice_channel(dim, num):
    ins = []
    if dim == 2:
        shape = (2, 2)
    else:
        shape = (2, 2, 2 ,3)
    ins = [np.ones(shape) * i for i in range(num)]
    e = np.hstack(ins)

    e_nd = mx.nd.empty(e.shape)
    e_nd[:] = e
    data = mx.sym.Variable('data')
    op = mx.sym.SliceChannel(data=data, num_outputs=num)
    arg_shape, output_shape, aux_shape = op.infer_shape(data=e_nd.shape)
    grad_nd = [mx.nd.empty(shape) for shape in arg_shape]

    exe = op.bind(mx.cpu(), args=[e_nd], args_grad=grad_nd)
    assert len(exe.outputs) == num
    o_nd = [exe.outputs[i] for i in range(num)]
    # test forward
    exe.forward()
    for i in range(num):
        assert reldiff(o_nd[i].asnumpy(), ins[i]) < 1e-5
    # test backward
    for i in range(num):
        o_nd[i] += i
    exe.backward(o_nd)
    assert reldiff(grad_nd[0].asnumpy(), np.hstack([ins[i] + i for i in range(num)])) < 1e-5

    # test slice channel with squeeze_axis
    op = mx.sym.SliceChannel(data=data, num_outputs=shape[1], squeeze_axis=1)
    arg_shape, output_shape, aux_shape = op.infer_shape(data=shape)
    assert len(output_shape) == shape[1]
    for o_shape in output_shape:
        assert len(o_shape) == len(shape) - 1
        assert o_shape == tuple([shape[0]] + list(shape[2:]))

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
    exec1 = out.bind(mx.Context('cpu'),
                     args=arr,
                     args_grad=dict_grad)
    exec1.forward()
    out1 = exec1.outputs[0]
    ret = np.concatenate([narray.asnumpy() for narray in arr], axis=dimension)
    assert same(out1.asnumpy(), ret)
    # backward
    out1.copyto(out_grad)
    out_grad[:] += 1
    exec1.backward([out_grad])

    for i, name in enumerate(arg_names):
        if not skip_second or name != 'arg1':
            grad = dict_grad[name]
            np_grad = arr_np[i]
            assert same(grad.asnumpy(), np_grad + 1)

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

def test_slice_channel():
    check_slice_channel(2, 4)
    check_slice_channel(4, 4)
    check_slice_channel(2, 16)

def check_regression(symbol, forward, backward):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    out = symbol(data, label)
    shape = (3, 1)
    arr_data = mx.random.uniform(-1, 1, shape)
    arr_label = mx.random.uniform(0, 1, shape[0])
    arr_grad = mx.nd.empty(shape)
    exec1 = out.bind(mx.cpu(),
                     args=[arr_data, arr_label],
                     args_grad={"data" : arr_grad})
    exec1.forward()
    out1 = exec1.outputs[0].asnumpy()
    npout = forward(arr_data.asnumpy())
    assert reldiff(npout, out1) < 1e-6

    exec1.backward()
    npout = backward(npout,  arr_label.asnumpy().reshape(npout.shape))
    assert reldiff(npout, arr_grad.asnumpy()) < 1e-6

def test_regression():
    check_regression(mx.symbol.LogisticRegressionOutput,
                     lambda x: 1.0 / (1.0 + np.exp(-x)),
                     lambda x, y : x - y)
    check_regression(mx.symbol.LinearRegressionOutput,
                     lambda x: x,
                     lambda x, y : x - y)

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
    exec1.forward()
    exec1.backward()

    grad0 = grad.asnumpy()

    for i in range(int(shape[0]/2)):
        l_np[i] = 0
    l[:] = l_np

    exec1.forward()
    exec1.backward()
    grad1 = grad.asnumpy()

    assert(abs(np.sum(grad1[:int(shape[0]/2)])) < 1e-5)
    assert(reldiff(grad0[int(shape[0]/2):], grad1[int(shape[0]/2):]) < 1e-5)

def check_softmax_with_shape(shape, xpu):
    # bind with label
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SoftmaxOutput(data=X, label=L)
    x = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l[:] = np_softmax(l.asnumpy())
    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward()
    out = exec1.outputs[0].asnumpy()
    assert_allclose(out, np_softmax(x.asnumpy()))
    exec1.backward()
    assert_allclose(grad.asnumpy(), np_softmax(x.asnumpy()) - l.asnumpy())

def test_softmax():
    check_softmax_with_shape((3, 4), mx.cpu())

def check_multi_softmax_with_shape(shape, xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SoftmaxOutput(data=X, label=L, multi_output=True)
    x = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l = mx.nd.empty((shape[0], shape[2]), ctx = xpu)
    l[:] = np.random.randint(0, shape[1]-1, (shape[0], shape[2]))
    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward()
    print(exec1.outputs[0].asnumpy())
    exec1.backward()
    print(grad.asnumpy())

def test_python_op():
    X = mx.symbol.Variable('X')
    op = mx.operator.NumpyOp()
    s = op.get_symbol(X, name='numpy_op')

    x = mx.ndarray.ones((10))*10
    dx = mx.ndarray.zeros((10))
    dy = mx.ndarray.ones((10))
    exec1 = s.bind(mx.cpu(), args=[x], args_grad = {'X': dx})
    exec1.forward()
    assert reldiff(x.asnumpy(), exec1.outputs[0].asnumpy()) < 1e-5
    exec1.backward(dy)
    assert reldiff(dy.asnumpy(), dx.asnumpy()) < 1e-5

def test_swapaxes():
    data = mx.symbol.Variable('data')
    shape = (2, 3, 4)
    data_tmp = np.ones(shape)
    data_tmp[0] = 1
    data_tmp[1] = 2
    arr_data = mx.nd.array(data_tmp)
    swap0 = mx.symbol.SwapAxis(data=data, dim1=0, dim2=2)
    swap = mx.symbol.SwapAxis(data=swap0, dim1=1, dim2=2)
    exe_c = swap.bind(mx.cpu(), args=[arr_data])
    exe_c.forward()
    out = exe_c.outputs[0].asnumpy()

    swap0_ = np.swapaxes(data_tmp, 0, 2)
    swap_ = np.swapaxes(swap0_, 1, 2)

    assert reldiff(out, swap_) < 1e-6

def test_scalarop():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)*5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = 2 / (4-((1+data+1)*2/5)-0.2)

    npout_1 = (4-((1+data_tmp+1)*2/5)-0.2)
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
    check_numeric_gradient(y, [x])
    check_symbolic_forward(y, [x], [2**x])
    check_symbolic_backward(y, [x], [np.ones(shape)], [np.log(2) * 2**x])

def test_embedding():
    in_dim = 10
    out_dim = 4
    batch = 24

    data = mx.sym.Variable("data")
    embed = mx.sym.Embedding(data=data, input_dim=in_dim, output_dim=out_dim, name="embed")
    exe_test = embed.simple_bind(mx.cpu(), grad_req={'data': 'null', 'embed_weight': 'write'}, data=(batch,))
    arg_map = dict(zip(embed.list_arguments(), exe_test.arg_arrays))
    grad_map = dict(zip(embed.list_arguments(), exe_test.grad_arrays))
    np_data = np.random.randint(low=0, high=in_dim, size=batch)
    np_weight = np.random.uniform(-0.01, 0.01, arg_map["embed_weight"].shape)
    np_onehot = np.zeros((batch, in_dim))
    np_onehot[np.arange(batch), np_data] = 1.0
    # forward
    arg_map["data"][:] = np_data
    arg_map["embed_weight"][:] = np_weight
    exe_test.forward()
    assert reldiff(exe_test.outputs[0].asnumpy(), np.dot(np_onehot, np_weight)) < 1e-6
    # backward
    np_grad = np.random.uniform(-1, 1, exe_test.outputs[0].shape)
    grad = mx.nd.zeros(np_grad.shape)
    grad[:] = np_grad
    exe_test.backward([grad])
    assert reldiff(grad_map["embed_weight"].asnumpy(), np.dot(np_onehot.T, np_grad)) < 1e-6

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
    exe_square = square.bind(mx.cpu(), args=[arr_data], args_grad=[arr_grad])
    exe_square.forward()
    assert reldiff(exe_square.outputs[0].asnumpy(), data_tmp * data_tmp) < 1e-6
    exe_square.backward(out_grad)
    assert reldiff(arr_grad.asnumpy(), 2.0 * data_tmp) < 1e-6

def test_sign():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = mx.sym.sign(data)
    exe_test = test.bind(mx.cpu(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout = np.sign(data_tmp)
    assert reldiff(out, npout) < 1e-6

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = 0;
    exe_test.backward(out_grad)
    assert reldiff(arr_grad.asnumpy(), npout_grad) < 1e-6

def test_round_ceil_floor():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5.543
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]= 2

    test = mx.sym.round(data) + mx.sym.ceil(data) +  mx.sym.floor(data)
    exe_test = test.bind(mx.cpu(), args=[arr_data])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout = np.round(data_tmp) + np.ceil(data_tmp) + np.floor(data_tmp)
    assert reldiff(out, npout) < 1e-6

def test_rsqrt_cos_sin():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test =  mx.sym.rsqrt(data) + mx.sym.cos(data) + mx.sym.sin(data)
    exe_test = test.bind(mx.cpu(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout =  1/ np.sqrt(data_tmp) + np.cos(data_tmp) + np.sin(data_tmp)
    assert reldiff(out, npout) < 1e-6

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = npout_grad * -(1.0 / (2.0 * data_tmp * np.sqrt(data_tmp))) + npout_grad * -1 * np.sin(data_tmp) + npout_grad * np.cos(data_tmp)
    exe_test.backward(out_grad)
    assert reldiff(arr_grad.asnumpy(), npout_grad) < 1e-6

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
    exe_test = test.bind(mx.cpu(), args=[arr_data1,arr_data2], args_grad=[arr_grad1,arr_grad2])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout =  np.maximum(data_tmp1,data_tmp2) + np.minimum(data_tmp1,data_tmp2)
    assert reldiff(out, npout) < 1e-6

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2
    exe_test.backward(out_grad)

    npout_grad = np.ones(shape)
    npout_grad[:] = 2
    mask1 = (data_tmp1 > data_tmp2).astype('float')
    mask2 = (data_tmp1 < data_tmp2).astype('float')
    npout_grad1 = npout_grad * mask1 + npout_grad * mask2
    npout_grad2 = (npout_grad - npout_grad * mask1) + (npout_grad - npout_grad * mask2)

    assert reldiff(arr_grad1.asnumpy(), npout_grad1) < 1e-6
    assert reldiff(arr_grad2.asnumpy(), npout_grad2) < 1e-6

def test_maximum_minimum_scalar():
    data1 = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp1 = np.random.rand(3,4)
    data_tmp1[:] = 2

    arr_data1 = mx.nd.array(data_tmp1)
    arr_grad1 = mx.nd.empty(shape)

    test =  mx.sym.maximum(data1,3) + mx.sym.maximum(9,data1) + mx.sym.minimum(5,data1) + mx.sym.minimum(data1,4)
    exe_test = test.bind(mx.cpu(), args=[arr_data1], args_grad=[arr_grad1])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout =  np.maximum(data_tmp1,3) + np.maximum(9,data_tmp1) + np.minimum(5,data_tmp1) + np.minimum(data_tmp1,4)
    assert reldiff(out, npout) < 1e-6

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

    assert reldiff(arr_grad1.asnumpy(), npout_grad1) < 1e-6

def test_abs():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = mx.sym.abs(data)
    exe_test = test.bind(mx.cpu(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward()
    out = exe_test.outputs[0].asnumpy()
    npout = abs(data_tmp)
    assert reldiff(out, npout) < 1e-6

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = npout_grad * np.sign(data_tmp)
    exe_test.backward(out_grad)
    assert reldiff(arr_grad.asnumpy(), npout_grad) < 1e-6

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
    input_data = mx.random.uniform(-5, 5, input_shape)
    out_grad = input_data
    args = {}
    args["data"] = input_data
    args['conv_weight'] = args['deconv_weight'] = mx.random.normal(0, 1,
        (num_filter, input_shape[1]) + kernel)
    args_grad = [mx.nd.empty(s) for s in arg_shapes]

    exe = deconv.bind(mx.cpu(), args=args, args_grad=args_grad)
    exe.forward()
    out = exe.outputs[0].asnumpy()
    exe.backward(out_grad)
    assert reldiff(out, args_grad[0].asnumpy()) < 1e-6

def check_deconvolution_gradient(input_shape, num_filter, pad):
    """configure A: input --> conv --> output.
       configure B: input --> deconv --> output
       the convolution and deconvoluiton has similar parameter which ensure
       the input shape is the same as output;
       During backward(), if the input of A equals output of B, and the output
       of A equals input of B, then the grad of weight should be the same;
    """
    stride = (1, 1)
    kernel = (2*pad[0]+1, 2*pad[1]+1)
    data_conv = mx.sym.Variable(name="data_conv")
    conv = mx.sym.Convolution(
        data=data_conv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "conv")
    data_deconv = mx.sym.Variable(name="data_deconv")
    deconv = mx.sym.Deconvolution(
        data=data_deconv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "deconv")

    conv_data = mx.random.uniform(-5, 5, input_shape)
    conv_args = {}
    conv_args["data_conv"] = conv_data
    conv_args['conv_weight'] = \
        mx.random.normal(0, 1,(num_filter, input_shape[1]) + kernel)
    conv_args_grad = [mx.nd.zeros(conv_data.shape),
        mx.nd.zeros((num_filter, input_shape[1]) + kernel)]
    exe_conv = conv.bind(mx.cpu(), args=conv_args, args_grad=conv_args_grad)
    conv_out_grad = mx.random.normal(0, 2, exe_conv.outputs[0].shape)
    exe_conv.backward(conv_out_grad)

    deconv_data = conv_out_grad
    deconv_args = {}
    deconv_args['data_deconv'] = deconv_data
    deconv_args['deconv_weight'] = conv_args['conv_weight']
    deconv_args_grad = [mx.nd.zeros(deconv_data.shape),
        mx.nd.zeros((num_filter, input_shape[1]) + kernel)]
    exe_deconv = deconv.bind(mx.cpu(), args=deconv_args, args_grad=deconv_args_grad)
    deconv_out_grad = conv_data[:]
    exe_deconv.backward(deconv_out_grad)
    assert reldiff(conv_args_grad[1].asnumpy(), deconv_args_grad[1].asnumpy()) < 1e-6

def check_deconvolution_target_shape(input_shape, kernel, stride, pad, adj, target_shape=None):
    data = mx.sym.Variable(name="data")
    deconv = mx.sym.Deconvolution(
        data=data, kernel=kernel, stride=stride, pad=pad, adj=adj, num_filter=5,
        target_shape = target_shape if target_shape is not None else (0, 0))
    arg_names = deconv.list_arguments()
    arg_shapes, out_shapes, _ = deconv.infer_shape(data=input_shape)
    assert out_shapes[0] == (input_shape[0], 5, 8, 8)

def test_deconvolution():
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

def check_nearest_upsampling_with_shape(shapes, scale, root_scale):
    arr = {'arg_%d'%i: mx.random.uniform(-10.0, 10.0, shape) for i, shape in zip(range(len(shapes)), shapes)}
    arr_grad = {'arg_%d'%i: mx.nd.zeros(shape) for i, shape in zip(range(len(shapes)), shapes)}

    up = mx.sym.UpSampling(*[mx.sym.Variable('arg_%d'%i) for i in range(len(shapes))], sample_type='nearest', scale=root_scale)
    exe = up.bind(mx.cpu(), args=arr, args_grad=arr_grad)
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

def test_batchnorm_training():
    for shape in [(2, 3), (2, 3, 2, 2)]:
        data_tmp = np.random.normal(size=shape)
        s = shape[1],
        gamma = np.ones(s)
        beta = np.ones(s)
        gamma[1] = 3
        beta[0] = 3

        rolling_mean = np.random.uniform(size=s)
        rolling_std = np.random.uniform(size=s)

        data = mx.symbol.Variable('data')
        test = mx.symbol.BatchNorm(data, fix_gamma=False)

        check_numeric_gradient(test, [data_tmp, gamma, beta], [rolling_mean, rolling_std], numeric_eps=1e-3, check_eps=5e-2)

def test_convolution_grouping():
    num_filter = 4
    num_group = 2
    kernel = (3, 3)
    shape = (1, 4, 9, 9)

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

    exe1 = y1.simple_bind(mx.cpu(), x=shape)
    exe2 = y2.simple_bind(mx.cpu(), x=shape, w=(num_filter, shape[1]//num_group, kernel[0], kernel[1]), b=(num_filter,))
    for arr1, arr2 in zip(exe1.arg_arrays, exe2.arg_arrays):
        arr1[:] = np.random.normal(size=arr1.shape)
        arr2[:] = arr1
    exe1.forward(is_train=True)
    exe1.backward(exe1.outputs[0])
    exe2.forward(is_train=True)
    exe2.backward(exe2.outputs[0])

    for arr1, arr2 in zip(exe1.outputs + exe1.grad_arrays, exe2.outputs + exe2.grad_arrays):
        np.testing.assert_allclose(arr1.asnumpy(), arr2.asnumpy(), rtol=1e-3)

def _gen_broadcast_data():
    # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
    ndim = np.random.randint(1, 8)
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

def _check_broadcast_op_forward(symbol, baseline):
    sample_num = 200
    for i in range(sample_num):
        d = _gen_broadcast_data()
        x = baseline(d[0], d[1])
        y = symbol.bind(mx.cpu(), args={'a': mx.nd.array(d[0]), 'b' : mx.nd.array(d[1])})
        y.forward()
        err = np.sum(np.abs(x - y.outputs[0].asnumpy())) / np.sum(np.abs(x))
        assert err < 1e-4, 'error %f, shapes are %s, %s' % (
            err, d[0].shape, d[1].shape)

def _check_broadcast_op_backward(symbol, baseline):
    sample_num = 200
    for i in range(sample_num):
        d = _gen_broadcast_data()
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
        y = symbol.bind(mx.cpu(), args={'a': mx.nd.array(d[0]), 'b' : mx.nd.array(d[1])},
                        args_grad=[y_1, y_2])
        y.forward()
        y.backward([mx.nd.array(out)])
        err = lambda x, y: np.sum(np.abs(x-y)) / np.sum(np.abs(x))
        err_1 = err(x_1, y_1.asnumpy())
        err_2 = err(x_2, y_2.asnumpy())
        assert err_1 < 1e-5 and err_2 < 1e-5, 'lhs error %f, rhs error %f, shapes are %s %s' % (
            err_1, err_2, d[0].shape, d[1].shape)

def test_broadcast_binary_op():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')

    def test_bplus(a, b):
        c = mx.sym.broadcast_plus(a, b)
        _check_broadcast_op_forward(c, lambda a, b: a + b)
        _check_broadcast_op_backward(c, lambda g_out, a, b: (g_out, g_out))

    def test_bminus(a, b):
        c = mx.sym.broadcast_minus(a, b)
        _check_broadcast_op_forward(c, lambda a, b: a - b)
        _check_broadcast_op_backward(c, lambda g_out, a, b: (g_out, - g_out))

    def test_bmul(a, b):
        c = mx.sym.broadcast_mul(a, b)
        _check_broadcast_op_forward(c, lambda a, b: a * b)
        _check_broadcast_op_backward(c, lambda g_out, a, b: (g_out * b, g_out * a))

    def test_bdiv(a, b):
        c = mx.sym.broadcast_div(a, b)
        _check_broadcast_op_forward(c, lambda a, b: a / b)
        _check_broadcast_op_backward(c, lambda g_out, a, b: (g_out / b, - g_out * a / (b * b)))

    def test_bpow(a, b):
        c = mx.sym.broadcast_power(a, b)
        _check_broadcast_op_forward(c, lambda a, b: a ** b)
        _check_broadcast_op_backward(c, lambda g_out, a, b: (g_out * a **(b - 1) * b,
                                                             g_out * a ** b * np.log(a)))

    test_bplus(a, b)
    test_bminus(a, b)
    test_bmul(a, b)
    test_bdiv(a, b)
    test_bpow(a, b)

def test_run_convolution_dilated_impulse_response(dil=(1,1), kernel_shape=(3,3), verbose=False):
    # Input for spike response
    spike_imgs = np.zeros(shape=(1,1,33,33), dtype=np.float32)
    spike_imgs[0,0,16,16] = 1.0
    spike_img = mx.nd.array(spike_imgs)
    spike_img2 = mx.nd.array(spike_imgs)


    kernel_weights = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)
    kernel_weights2 = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)

    kernel = mx.symbol.Variable('kernel')
    in_img = mx.symbol.Variable('input')
    net = mx.symbol.Convolution(in_img, num_filter=1,kernel=kernel_shape, dilate=dil, no_bias="true", name='test_convolution')
    net.list_arguments()
    be = net.bind(mx.cpu(), args={ 'input' : spike_img, 'test_convolution_weight' : kernel_weights},
                args_grad={'input' : spike_img2, 'test_convolution_weight' : kernel_weights2 } )
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    ndo = be.outputs[0]

    out_grads = np.zeros(shape=be.outputs[0].shape, dtype=np.float32)
    out_grads[0,0, 16,16] = 1.0
    out_grad = mx.nd.array(out_grads)
    be.backward([out_grad])
    vgrad = be.grad_arrays[0].asnumpy()
    out = out_o.reshape((out_o.shape[2],out_o.shape[3]))
    nzx,nzy = np.nonzero(out)
    assert(np.sum(out)==np.prod(kernel_shape))
    assert(np.sum(vgrad)==np.prod(kernel_shape))

    # Now check whether the input gradient was computed correctly
    input_grad = mx.nd.array(vgrad)

    be = net.bind(mx.cpu(), args={ 'input' : input_grad, 'test_convolution_weight' : kernel_weights})
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    assert(out_o[0,0,16,16]==np.prod(kernel_shape))

    rnd_kernel_s = np.random.uniform(low=0.0, high=1.0, size=tuple([1,1]+list(kernel_shape))).astype(np.float32)
    impulse_error = mx.nd.array(out_o/np.sum(out_o)) # This should be 1.0 at [0,0,16,16]
    rnd_kernel = mx.nd.array(rnd_kernel_s)

    rnd_kernel2 = mx.nd.array(rnd_kernel_s)
    white_in = mx.nd.ones(shape=(1,1,33,33))
    white_in2 = mx.nd.ones(shape=(1,1,33,33))

    be = net.bind(mx.cpu(), args={ 'input' : white_in, 'test_convolution_weight' : rnd_kernel},
                args_grad={'input' : white_in2, 'test_convolution_weight' : rnd_kernel2 } )

    be.forward(True)
    be.backward([impulse_error])
    out_orig = be.outputs[0].asnumpy()
    kernel_gradient = be.grad_arrays[1].asnumpy()

    dkernel = mx.nd.array(rnd_kernel_s + kernel_gradient)

    be = net.bind(mx.cpu(), args={ 'input' : white_in, 'test_convolution_weight' : dkernel})

    be.forward(True)
    out = be.outputs[0].asnumpy()
    # Now do a simple check of the kernel gradient
    assert(out[0,0,16,16] - np.sum(kernel_gradient) - out_orig[0,0,16,16] < 0.001)


def test_convolution_dilated_impulse_response():
    for dil in [ (1,1), (2,2), (3,3) ]:
        for ks in [ (3,3), (4,4), (2,3), (3,2), (1,1) ]:
            test_run_convolution_dilated_impulse_response(dil=dil, kernel_shape=ks)

def test_reshape():

    def test_reshape_new(src_shape, shape_args, dst_shape):
        net = mx.sym.Variable("data")
        net = mx.sym.Reshape(net, shape=shape_args)
        js = net.tojson()
        net = mx.sym.load_json(js)
        _, output_shape, __ = net.infer_shape(data=src_shape)
        assert output_shape[0] == dst_shape, \
            'Src Shape = %s, Shape Arguments = %s, Dst Shape = %s, Output Shape = %s' \
            %(str(src_shape), str(shape_args), str(dst_shape), str(output_shape[0]))
        dat_npy = np.random.rand(*src_shape)
        grad_npy = np.random.rand(*dst_shape)
        exe = net.simple_bind(mx.cpu(), data=src_shape)
        exe.arg_dict['data'][:] = dat_npy
        exe.forward(is_train=True)
        assert np.square(exe.outputs[0].asnumpy() - dat_npy.reshape(dst_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Dst Shape = %s' %(str(src_shape),
                                                                     str(shape_args), str(dst_shape))
        exe.backward(out_grads=mx.nd.array(grad_npy))
        assert np.square(exe.grad_dict['data'].asnumpy() - grad_npy.reshape(src_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Dst Shape = %s' %(str(src_shape),
                                                                     str(shape_args), str(dst_shape))
    # Test new api (Using shape)
    test_cases = [[(2, 3, 5, 5), (0, -1), (2, 75)],
                  [(2, 3, 5, 5), (0, 0, -1), (2, 3, 25)],
                  [(5, 3, 4, 5), (0, -1, 0), (5, 15, 4)],
                  [(2, 3, 5, 4), (-1, 0, 0), (8, 3, 5),
                  [(2, 3, 4, 5), (3, -1, 0), (3, 10, 4)],
                  [(2, 3, 5, 5), (5, 3, 0, -1), (5, 3, 5, 2)],
                  [(2, 3, 5, 5), (0, 0, 0, 0), (2, 3, 5, 5)],
                  [(2, 4, 5, 3), (-1, 2, 2, 1), (30, 2, 2, 1)]]]
    for test_case in test_cases:
        test_reshape_new(test_case[0], test_case[1], test_case[2])
    # Test old api
    net = mx.sym.Variable("data")
    net = mx.sym.Reshape(net, target_shape=(2, 0))
    js = net.tojson()
    net = mx.sym.load_json(js)
    _, output_shape, __ = net.infer_shape(data=(2, 3, 5, 5))
    assert(output_shape[0] == (2, 75))

def test_reduce():
    sample_num = 200
    def test_reduce_inner(numpy_reduce_func, numpy_reduce_grad_func, mx_reduce_sym):
        for i in range(sample_num):
            # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
            ndim = np.random.randint(1, 8)
            shape = np.random.randint(1, 6, size=(ndim,))
            axis_num = np.random.randint(0, ndim, size=1)
            axis_flags = np.random.randint(0, 2, size=ndim)
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
            else:
                b = mx_reduce_sym(a, axis=axes, keepdims=keepdims)
            dat_npy = np.random.rand(*shape)
            sum_groundtruth = np.array(numpy_reduce_func(dat_npy, axis=axes, keepdims=keepdims))
            if sum_groundtruth.shape == ():
                sum_groundtruth = np.array([sum_groundtruth])
            grad_nd = mx.nd.empty(shape)
            outgrad_npy = np.array(np.random.rand(*sum_groundtruth.shape))
            grad_groundtruth = numpy_reduce_grad_func(outgrad=outgrad_npy, data=dat_npy,
                                                      axis=axes, keepdims=keepdims)
            net = b.bind(mx.cpu(), args={'a': mx.nd.array(dat_npy)},
                         args_grad={'a': grad_nd})
            net.forward(is_train=True)

            err_forward = reldiff(net.outputs[0].asnumpy(), sum_groundtruth)
            assert err_forward < 1E-4
            net.backward(out_grads=mx.nd.array(outgrad_npy))
            err_backward = reldiff(grad_nd.asnumpy(), grad_groundtruth)
            assert err_backward < 1E-4
    test_reduce_inner(lambda data, axis, keepdims:_np_reduce(data, axis, keepdims, np.sum),
                      lambda outgrad, data, axis, keepdims:
                        outgrad.reshape(_np_reduce(data, axis, 1, np.sum).shape),
                      mx.symbol.sum)

def test_broadcast():
    sample_num = 200
    for i in range(sample_num):
        # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
        ndim = np.random.randint(1, 8)
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
            grad_groundtruth = _np_reduce(outgrad_npy, axis=axis, keepdims=True,
                                          numpy_reduce_func=np.sum)
            net = sym_bcast.bind(mx.cpu(), args={'a': mx.nd.array(dat_npy)},
                                                 args_grad={'a': grad_nd})
            net.forward(is_train=True)
            assert (net.outputs[0].shape == target_shape).all()
            err_forward = reldiff(net.outputs[0].asnumpy(), groundtruth)
            assert err_forward < 1E-4
            net.backward(out_grads=mx.nd.array(outgrad_npy))
            err_backward = reldiff(grad_nd.asnumpy(), grad_groundtruth)
            assert err_backward < 1E-4
        test_broadcasting_ele(sym_bcast_axis)
        test_broadcasting_ele(sym_bcast_to)

def test_transpose():
    for ndim in range(1, 6):
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
        for t in range(5):
            dims = list(np.random.randint(1, 10, size=ndim))
            axis = np.random.randint(1, ndim+1)
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.expand_dims(x, axis=axis)
            assert_allclose(np.expand_dims(x.asnumpy(), axis=axis), y.asnumpy())


def test_crop():
    for ndim in range(1, 6):
        for t in range(5):
            dims = []
            begin = []
            end = []
            idx = []
            for i in range(ndim):
                d = random.randint(1, 10)
                b = random.randint(0, d-1)
                e = random.randint(b+1, d)
                dims.append(d)
                begin.append(b)
                end.append(e)
                idx.append(slice(b, e))
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.crop(x, begin=tuple(begin), end=tuple(end))
            assert_allclose(x.asnumpy()[idx], y.asnumpy())


def test_slice_axis():
    for ndim in range(1, 6):
        shape = np.random.randint(1, 11, size=(ndim,))
        for t in range(ndim):
            d = shape[t]
            b = random.randint(0, d-1)
            e = random.randint(b+1, d)
            idx = []
            for i in range(ndim):
                idx.append(slice(0, shape[i]))
            idx[t] = slice(b, e)

            X = mx.symbol.Variable('X')
            x = mx.nd.array(np.random.normal(size=shape))
            Y = mx.symbol.slice_axis(data=X, axis=t, begin=b, end=e)

            xgrad = mx.nd.empty(x.shape)
            exec1 = Y.bind(mx.cpu(), args = [x], args_grad = {'X': xgrad})
            exec1.forward()
            y = exec1.outputs[0]
            assert_allclose(x.asnumpy()[idx], y.asnumpy())
            exec1.backward([y])
            xx = x.asnumpy()
            xx[:] = 0.0
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(xx, xgrad.asnumpy())


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
    import pdb
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
                    dev = mx.cpu()
                    #dev = mx.gpu(0)
                    args = {}
                    args['data'] = mx.random.normal(0, 1, data_shape, dev)
                    args['loc_conv_weight'] = mx.nd.zeros((num_filter, data_shape[1], kernel[0], kernel[1]), ctx=dev)
                    args['loc_conv_bias'] = mx.nd.zeros((num_filter,), ctx=dev)
                    args['loc_fc_weight'] = mx.nd.zeros((6, num_filter*data_shape[2]*data_shape[3]), ctx=dev)
                    args['loc_fc_bias'] = mx.nd.array([0.5, 0, 0, 0, 0.5, 0], ctx=dev)
                    grad_grad = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
                    exe = stn.bind(dev, args=args, args_grad=grad_grad)
                    exe.forward(is_train=True)
                    out = exe.outputs[0].asnumpy()
                    # check forward
                    reldiff(out, args['data'].asnumpy()[:, :, h//4:h-h//4, w//4:w-w//4]) < 1e-6
                    out_grad = mx.nd.ones(out.shape, ctx=dev)
                    exe.backward([out_grad])
                    # check backward
                    reldiff(out_grad.asnumpy(), grad_grad[0].asnumpy()[:, :, h//4:h-h//4, w//4:w-w//4]) < 1e-6


def test_dot(ctx=mx.cpu()):
    for m in range(1, 5):
        for k in range(1, 5):
            for n in range(1, 5):
                a_npy = np.random.normal(0, 1, (m, k))
                b_npy = np.random.normal(0, 1, (k, n))
                c_npy = np.empty((m, n))
                ograd_npy = np.random.normal(0, 1, (m, n))
                agrad_npy = np.empty((m, k))
                bgrad_npy = np.empty((k, n))
                c_npy[:, :] = np.dot(a_npy[:, :], b_npy[:, :])
                bgrad_npy[:, :] = np.dot(a_npy[:, :].T, ograd_npy[:, :])
                agrad_npy[:, :] = np.dot(ograd_npy[:, :], b_npy[:, :].T)
                a = mx.sym.Variable('a')
                b = mx.sym.Variable('b')
                c = mx.sym.dot(a, b)
                exe = c.simple_bind(ctx=ctx, a=a_npy.shape, b=b_npy.shape)
                outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
                assert reldiff(outputs[0].asnumpy(), c_npy) < 1E-5
                exe.backward(out_grads=[mx.nd.array(ograd_npy, ctx=exe._ctx)])
                assert reldiff(exe.grad_dict['a'].asnumpy(), agrad_npy) < 1E-5
                assert reldiff(exe.grad_dict['b'].asnumpy(), bgrad_npy) < 1E-5


def test_batch_dot(ctx=mx.cpu()):
    for batch_size in range(1, 5):
        for m in range(1, 5):
            for k in range(1, 5):
                for n in range(1, 5):
                    a_npy = np.random.normal(0, 1, (batch_size, m, k))
                    b_npy = np.random.normal(0, 1, (batch_size, k, n))
                    c_npy = np.empty((batch_size, m, n))
                    ograd_npy = np.random.normal(0, 1, (batch_size, m, n))
                    agrad_npy = np.empty((batch_size, m, k))
                    bgrad_npy = np.empty((batch_size, k, n))
                    for i in range(batch_size):
                        c_npy[i, :, :] = np.dot(a_npy[i, :, :], b_npy[i, :, :])
                        bgrad_npy[i, :, :] = np.dot(a_npy[i, :, :].T, ograd_npy[i, :, :])
                        agrad_npy[i, :, :] = np.dot(ograd_npy[i, :, :], b_npy[i, :, :].T)
                    a = mx.sym.Variable('a')
                    b = mx.sym.Variable('b')
                    c = mx.sym.batch_dot(a, b)
                    exe = c.simple_bind(ctx=ctx, a=a_npy.shape, b=b_npy.shape)
                    outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
                    assert reldiff(outputs[0].asnumpy(), c_npy) < 1E-5
                    exe.backward(out_grads=[mx.nd.array(ograd_npy, ctx=exe._ctx)])
                    assert reldiff(exe.grad_dict['a'].asnumpy(), agrad_npy) < 1E-5
                    assert reldiff(exe.grad_dict['b'].asnumpy(), bgrad_npy) < 1E-5


if __name__ == '__main__':
    test_expand_dims()
    test_slice_axis()
    test_softmax()
    test_broadcast_binary_op()
    test_flip()
    test_crop()
    test_transpose()
    test_convolution_grouping()
    test_nearest_upsampling()
    test_binary_op_duplicate_input()
    test_elementwise_sum()
    test_concat()
    test_slice_channel()
    test_regression()
    test_python_op()
    test_swapaxes()
    test_scalarop()
    test_scalar_pow()
    test_symbol_pow()
    test_pow_fn()
    test_embedding()
    test_rsqrt_cos_sin()
    test_maximum_minimum()
    test_maximum_minimum_scalar()
    test_abs()
    test_round_ceil_floor()
    test_deconvolution()
    test_batchnorm_training()
    check_softmax_with_ignore_label(mx.cpu())
    test_convolution_dilated_impulse_response()
    test_reshape()
    test_reduce()
    test_broadcast()
    test_stn()
    test_dot()
    test_batch_dot()
