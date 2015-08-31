import os
import mxnet as mx
import numpy as np
import pickle as pkl

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


def random_narray(dim):
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    data = mx.narray.create(shape)
    data.numpy[:] = np.random.uniform(-10, 10, data.shape)
    return data

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


def test_narray_pickle():
    np.random.seed(0)
    maxdim = 5
    nrepeat = 10
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            a = random_narray(dim)
            b = mx.narray.create(a.shape)
            a.numpy[:] = np.random.uniform(-10, 10, a.shape)
            b.numpy[:] = np.random.uniform(-10, 10, a.shape)
            a = a + b
            data = pkl.dumps(a)
            a2 = pkl.loads(data)
            assert np.sum(a.numpy != a2.numpy) == 0


def test_narray_saveload():
    np.random.seed(0)
    maxdim = 5
    nrepeat = 10
    fname = 'tmp_list.bin'
    for repeat in range(nrepeat):
        data = []
        for i in range(10):
            data.append(random_narray(np.random.randint(1, 5)))
        mx.narray.save(fname, data)
        data2 = mx.narray.load(fname)
        assert len(data) == len(data2)
        for x, y in zip(data, data2):
            assert np.sum(x.numpy != y.numpy) == 0
        dmap = {'narray xx %s' % i : x for i, x in enumerate(data)}
        mx.narray.save(fname, dmap)
        dmap2 = mx.narray.load(fname)
        assert len(dmap2) == len(dmap)
        for k, x in dmap.items():
            y = dmap2[k]
            assert np.sum(x.numpy != y.numpy) == 0
    os.remove(fname)

if __name__ == '__main__':
    test_narray_pickle()
    test_narray_saveload()
    test_narray_copy()
    test_narray_elementwise()
