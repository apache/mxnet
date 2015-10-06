import os
import mxnet as mx
import numpy as np
import pickle as pkl

def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    reldiff = diff  / norm
    return reldiff


def same(a, b):
    return np.sum(a != b) == 0


def check_with_uniform(uf, arg_shapes, dim=None):
    """check function consistency with uniform random numbers"""
    if isinstance(arg_shapes, int):
        assert dim
        shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
        arg_shapes = [shape] * arg_shapes
    ndarray_arg = []
    numpy_arg = []
    for s in arg_shapes:
        npy = np.random.uniform(-10, 10, s)
        narr = mx.nd.array(npy)
        ndarray_arg.append(narr)
        numpy_arg.append(npy)
    out1 = uf(*ndarray_arg)
    out2 = uf(*numpy_arg)
    assert out1.shape == out2.shape
    assert reldiff(out1.asnumpy(), out2) < 1e-6


def random_ndarray(dim):
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    data = mx.nd.array(np.random.uniform(-10, 10, shape))
    return data

def test_ndarray_elementwise():
    np.random.seed(0)
    nrepeat = 10
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            check_with_uniform(lambda x, y: x + y, 2, dim)
            check_with_uniform(lambda x, y: x - y, 2, dim)
            check_with_uniform(lambda x, y: x * y, 2, dim)
            check_with_uniform(lambda x, y: x / y, 2, dim)

def test_ndarray_negate():
    npy = np.random.uniform(-10, 10, (2,3,4))
    arr = mx.nd.array(npy)
    assert reldiff(npy, arr.asnumpy()) < 1e-6
    assert reldiff(-npy, (-arr).asnumpy()) < 1e-6

    # a final check to make sure the negation (-) is not implemented
    # as inplace operation, so the contents of arr does not change after
    # we compute (-arr)
    assert reldiff(npy, arr.asnumpy()) < 1e-6


def test_ndarray_copy():
    c = mx.nd.array(np.random.uniform(-10, 10, (10, 10)))
    d = c.copyto(mx.Context('cpu', 0))
    assert np.sum(np.abs(c.asnumpy() != d.asnumpy())) == 0.0


def test_ndarray_scalar():
    c = mx.nd.empty((10,10))
    d = mx.nd.empty((10,10))
    c[:] = 0.5
    d[:] = 1.0
    d -= c * 2 / 3 * 6.0
    c += 0.5
    assert(np.sum(c.asnumpy()) - 100 < 1e-5)
    assert(np.sum(d.asnumpy()) + 100 < 1e-5)
    c[:] = 2
    assert(np.sum(c.asnumpy()) - 200 < 1e-5)
    d = -c + 2
    assert(np.sum(d.asnumpy()) < 1e-5)

def test_ndarray_pickle():
    np.random.seed(0)
    maxdim = 5
    nrepeat = 10
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            a = random_ndarray(dim)
            b = mx.nd.empty(a.shape)
            a[:] = np.random.uniform(-10, 10, a.shape)
            b[:] = np.random.uniform(-10, 10, a.shape)
            a = a + b
            data = pkl.dumps(a)
            a2 = pkl.loads(data)
            assert np.sum(a.asnumpy() != a2.asnumpy()) == 0


def test_ndarray_saveload():
    np.random.seed(0)
    maxdim = 5
    nrepeat = 10
    fname = 'tmp_list.bin'
    for repeat in range(nrepeat):
        data = []
        for i in range(10):
            data.append(random_ndarray(np.random.randint(1, 5)))
        mx.nd.save(fname, data)
        data2 = mx.nd.load(fname)
        assert len(data) == len(data2)
        for x, y in zip(data, data2):
            assert np.sum(x.asnumpy() != y.asnumpy()) == 0
        dmap = {'ndarray xx %s' % i : x for i, x in enumerate(data)}
        mx.nd.save(fname, dmap)
        dmap2 = mx.nd.load(fname)
        assert len(dmap2) == len(dmap)
        for k, x in dmap.items():
            y = dmap2[k]
            assert np.sum(x.asnumpy() != y.asnumpy()) == 0
    os.remove(fname)


def test_ndarray_slice():
    shape = (10,)
    A = mx.nd.array(np.random.uniform(-10, 10, shape))
    A2 = A.asnumpy()
    assert same(A[3:8].asnumpy(), A2[3:8])
    A2[3:8] *= 10;
    A[3:8] = A2[3:8]
    assert same(A[3:8].asnumpy(), A2[3:8])

def test_clip():
    shape = (10,)
    A = mx.random.uniform(-10, 10, shape)
    B = mx.nd.clip(A, -2, 2)
    B1 = B.asnumpy()
    for i in range(shape[0]):
        assert B1[i] >= -2
        assert B1[i] <= 2

def test_dot():
    a = np.random.uniform(-3, 3, (3, 4))
    b = np.random.uniform(-3, 3, (4, 5))
    c = np.dot(a, b)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B)
    assert reldiff(c, C.asnumpy()) < 1e-5

if __name__ == '__main__':
    test_ndarray_slice()
    test_ndarray_pickle()
    test_ndarray_saveload()
    test_ndarray_copy()
    test_ndarray_elementwise()
    test_ndarray_negate()
    test_ndarray_scalar()
    test_clip()
    test_dot()
