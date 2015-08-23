import mxnet as mx
import numpy as np

def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    reldiff = diff  / norm
    return reldiff

def check_with_uniform(uf, arg_shapes, dim=None):
    """check function consistency with uniform random numbers"""
    if isinstance(arg_shapes, int):
        assert dim
        shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
        arg_shapes = [shape] * arg_shapes
    narray_arg = []
    numpy_arg = []
    for s in arg_shapes:
        narr = mx.narray.create(s)
        npy = np.random.uniform(-10, 10, s)
        narr.numpy[:] = npy
        narray_arg.append(narr)
        numpy_arg.append(npy)
    out1 = uf(*narray_arg)
    out2 = uf(*numpy_arg)
    assert out1.shape == out2.shape
    assert reldiff(out1.numpy, out2) < 1e-6


def test_narray_elementwise():
    np.random.seed(0)
    nrepeat = 10
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            check_with_uniform(lambda x, y: x + y, 2, dim)
            check_with_uniform(lambda x, y: x - y, 2, dim)
            check_with_uniform(lambda x, y: x * y, 2, dim)
            # check_with_uniform(lambda x, y: x / y, 2, dim)


def test_narray_copy():
    c = mx.narray.create((10,10))
    c.numpy[:] = np.random.uniform(-10, 10, c.shape)
    d = c.copyto(mx.Context('cpu', 0))
    assert np.sum(np.abs(c.numpy != d.numpy)) == 0.0
