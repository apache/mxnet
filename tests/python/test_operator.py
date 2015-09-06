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
    arr = [mx.narray.create(shape) for i in range(n)]
    arr_grad = [mx.narray.create(shape) for i in range(n)]
    for i in range(n):
        arr[i].numpy[:] = np.random.uniform(-10, 10, shape)
    exec1 = out.bind(mx.Context('cpu'),
                     args=arr,
                     args_grad=arr_grad)
    out1 = exec1.heads()[0].numpy
    exec1.forward()
    out1 = exec1.heads()[0].numpy
    out = sum(a.numpy for a  in arr)
    assert reldiff(out, out1) < 1e-6
    out_grad = mx.narray.create(shape)
    out_grad.numpy[:] = np.random.uniform(-10, 10, shape)
    # backward
    exec1.backward([out_grad])
    for a in arr_grad:
        assert same(a.numpy, out_grad.numpy)


def test_elementwise_sum():
    np.random.seed(0)
    nrepeat = 2
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
            check_elementwise_sum_with_shape(shape, np.random.randint(1, 8))


if __name__ == '__main__':
    test_elementwise_sum()
