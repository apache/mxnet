# pylint: skip-file

import numpy as np
import mxnet as mx

def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    if diff == 0:
        return 0
    reldiff = diff  / norm
    return reldiff


def same(a, b):
    return np.sum(a != b) == 0


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
        shape = (2,2)
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

def check_concat_with_shape(shapes):
    n = len(shapes)
    # forward
    target_dim = 0
    for shape in shapes:
        target_dim += shape[1]

    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.Concat(*inputs, name='conc')
    arr = [mx.nd.empty(shape) for shape in shapes]
    for i in range(n):
        arr[i][:] = shapes[i][1]
    arr_np = [np.copy(narray.asnumpy()) for narray in arr]
    arr_grad = [mx.nd.empty(shape) for shape in shapes]
    args = out.list_arguments()
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(**dict(zip(args, shapes)))
    out_grad = mx.nd.empty(out_shapes[0])
    exec1 = out.bind(mx.Context('cpu'),
                     args=arr,
                     args_grad=arr_grad)
    exec1.forward()
    out1 = exec1.outputs[0]
    ret = np.concatenate([narray.asnumpy() for narray in arr], axis=1)
    assert same(out1.asnumpy(), ret)
    # backward
    out1.copyto(out_grad)
    out_grad[:] += 1
    exec1.backward([out_grad])
    for grad, np_grad in zip(arr_grad, arr_np):
        assert same(grad.asnumpy(), np_grad + 1)

def test_concat():
    n = 2
    batch = 2
    ch = [2, 3, 4, 5, 6]
    h = 3
    w = 4
    # test  2D
    for dim in range(2, 6):
        shapes = []
        for i in range(dim):
            shapes.append((batch, ch[i]))
        check_concat_with_shape(shapes)
    # test 4D
    for dim in range(2, 6):
        shapes = []
        for i in range(dim):
            shapes.append((batch, ch[i], h, w))
        check_concat_with_shape(shapes)

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

def check_softmax_with_shape(shape, xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.Softmax(data=X, label=L)
    x = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    l[:] = np.random.randint(0, shape[0]-1, (shape[0],))
    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    print('foward')
    exec1.forward()
    print(exec1.outputs[0].asnumpy())
    exec1.backward()
    print(grad.asnumpy())

def check_multi_softmax_with_shape(shape, xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.Softmax(data=X, label=L, multi_output=True)
    x = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l = mx.nd.empty((shape[0], shape[2]), ctx = xpu)
    l[:] = np.random.randint(0, shape[1]-1, (shape[0], shape[2]))
    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward()
    print(exec1.outputs[0].asnumpy())
    exec1.backward()
    print(grad.asnumpy())

if __name__ == '__main__':
    test_elementwise_sum()
    test_concat()
    test_slice_channel()
    test_regression()
    #check_softmax_with_shape((3,4), mx.cpu())
    #check_multi_softmax_with_shape((3,4,5), mx.cpu())
