import os
import mxnet as mx
import numpy as np
import pickle as pkl


def _np_reduce(dat, axis, keepdims, numpy_reduce_func):
    if isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis) if axis is not None else range(len(dat.shape))
    ret = dat
    for i in reversed(sorted(axis)):
        ret = numpy_reduce_func(ret, axis=i)
    if keepdims:
        keepdims_shape = list(dat.shape)
        for i in axis:
            keepdims_shape[i] = 1
        ret = ret.reshape(tuple(keepdims_shape))
    return ret


def reldiff(a, b):
    diff = np.abs(a - b)
    norm = np.abs(a)
    reldiff = np.max(diff  / (norm + 1e-7))
    return reldiff


def same(a, b):
    return np.sum(a != b) == 0


def check_with_uniform(uf, arg_shapes, dim=None, npuf=None, rmin=-10, type_list=[np.float32]):
    """check function consistency with uniform random numbers"""
    if isinstance(arg_shapes, int):
        assert dim
        shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
        arg_shapes = [shape] * arg_shapes
    for dtype in type_list:
        ndarray_arg = []
        numpy_arg = []
        for s in arg_shapes:
            npy = np.random.uniform(rmin, 10, s).astype(dtype)
            narr = mx.nd.array(npy, dtype=dtype)
            ndarray_arg.append(narr)
            numpy_arg.append(npy)
        out1 = uf(*ndarray_arg)
        if npuf is None:
            out2 = uf(*numpy_arg).astype(dtype)
        else:
            out2 = npuf(*numpy_arg).astype(dtype)

        assert out1.shape == out2.shape
        if isinstance(out1, mx.nd.NDArray):
            out1 = out1.asnumpy()
        if dtype == np.float16:
            assert reldiff(out1, out2) < 2e-3
        else:
            assert reldiff(out1, out2) < 1e-6


def random_ndarray(dim):
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    data = mx.nd.array(np.random.uniform(-10, 10, shape))
    return data

def test_ndarray_elementwise():
    np.random.seed(0)
    nrepeat = 10
    maxdim = 4
    all_type = [np.float32, np.float64, np.float16, np.uint8, np.int32]
    real_type = [np.float32, np.float64, np.float16]
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            check_with_uniform(lambda x, y: x + y, 2, dim, type_list=all_type)
            check_with_uniform(lambda x, y: x - y, 2, dim, type_list=all_type)
            check_with_uniform(lambda x, y: x * y, 2, dim, type_list=all_type)
            check_with_uniform(lambda x, y: x / y, 2, dim, type_list=real_type)
            check_with_uniform(lambda x, y: x / y, 2, dim, rmin=1, type_list=all_type)
            check_with_uniform(mx.nd.sqrt, 1, dim, np.sqrt, rmin=0)
            check_with_uniform(mx.nd.square, 1, dim, np.square, rmin=0)
            check_with_uniform(lambda x: mx.nd.norm(x).asscalar(), 1, dim, np.linalg.norm)

def test_ndarray_negate():
    npy = np.random.uniform(-10, 10, (2,3,4))
    arr = mx.nd.array(npy)
    assert reldiff(npy, arr.asnumpy()) < 1e-6
    assert reldiff(-npy, (-arr).asnumpy()) < 1e-6

    # a final check to make sure the negation (-) is not implemented
    # as inplace operation, so the contents of arr does not change after
    # we compute (-arr)
    assert reldiff(npy, arr.asnumpy()) < 1e-6


def test_ndarray_choose():
    shape = (100, 20)
    npy = np.arange(np.prod(shape)).reshape(shape)
    arr = mx.nd.array(npy)
    nrepeat = 3
    for repeat in range(nrepeat):
        indices = np.random.randint(shape[1], size=shape[0])
        assert same(npy[np.arange(shape[0]), indices],
                    mx.nd.choose_element_0index(arr, mx.nd.array(indices)).asnumpy())


def test_ndarray_fill():
    shape = (100, 20)
    npy = np.arange(np.prod(shape)).reshape(shape)
    arr = mx.nd.array(npy)
    new_npy = npy.copy()
    nrepeat = 3
    for repeat in range(nrepeat):
        indices = np.random.randint(shape[1], size=shape[0])
        val = np.random.randint(shape[1], size=shape[0])
        new_npy[:] = npy
        new_npy[np.arange(shape[0]), indices] = val
        assert same(new_npy,
                    mx.nd.fill_element_0index(arr, mx.nd.array(val), mx.nd.array(indices)).asnumpy())


def test_ndarray_onehot():
    shape = (100, 20)
    npy = np.arange(np.prod(shape)).reshape(shape)
    arr = mx.nd.array(npy)
    nrepeat = 3
    for repeat in range(nrepeat):
        indices = np.random.randint(shape[1], size=shape[0])
        npy[:] = 0.0
        npy[np.arange(shape[0]), indices] = 1.0
        mx.nd.onehot_encode(mx.nd.array(indices), out=arr)
        assert same(npy, arr.asnumpy())


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


def test_ndarray_slice_along_axis():
    arr = mx.nd.array(np.random.uniform(-10, 10, (3, 4, 2, 3)))
    sub_arr = mx.nd.zeros((3, 2, 2, 3))
    arr._copy_slice_to(1, 1, 3, sub_arr)

    # test we sliced correctly
    assert same(arr.asnumpy()[:, 1:3, :, :], sub_arr.asnumpy())

    # test that slice is copy, instead of shared memory
    sub_arr[:] = 0
    assert not same(arr.asnumpy()[:, 1:3, :, :], sub_arr.asnumpy())


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

def test_reduce():
    sample_num = 200
    def test_reduce_inner(numpy_reduce_func, nd_reduce_func):
        for i in range(sample_num):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 11, size=ndim)
            axis_flags = np.random.randint(0, 2, size=ndim)
            axes = []
            for (axis, flag) in enumerate(axis_flags):
                if flag:
                    axes.append(axis)
            keepdims = np.random.randint(0, 2)
            dat = np.random.rand(*shape) - 0.5
            if 0 == len(axes):
                axes = tuple(range(ndim))
            else:
                axes = tuple(axes)
            numpy_ret = numpy_reduce_func(dat, axis=axes, keepdims=keepdims)

            ndarray_ret = nd_reduce_func(mx.nd.array(dat), axis=axes, keepdims=keepdims)
            if type(ndarray_ret) is mx.ndarray.NDArray:
                ndarray_ret = ndarray_ret.asnumpy()
            assert (ndarray_ret.shape == numpy_ret.shape) or \
                   (ndarray_ret.shape == (1,) and numpy_ret.shape == ()), "nd:%s, numpy:%s" \
                                                         %(ndarray_ret.shape, numpy_ret.shape)
            err = np.square(ndarray_ret - numpy_ret).mean()
            assert err < 1E-4
    test_reduce_inner(lambda data, axis, keepdims:_np_reduce(data, axis, keepdims, np.sum),
                      mx.nd.sum)
    test_reduce_inner(lambda data, axis, keepdims:_np_reduce(data, axis, keepdims, np.max),
                      mx.nd.max)
    test_reduce_inner(lambda data, axis, keepdims:_np_reduce(data, axis, keepdims, np.min),
                      mx.nd.min)

def test_broadcast():
    sample_num = 1000
    def test_broadcast_to():
        for i in range(sample_num):
            ndim = np.random.randint(1, 6)
            target_shape = np.random.randint(1, 11, size=ndim)
            shape = target_shape.copy()
            axis_flags = np.random.randint(0, 2, size=ndim)
            axes = []
            for (axis, flag) in enumerate(axis_flags):
                if flag:
                    shape[axis] = 1
            dat = np.random.rand(*shape) - 0.5
            numpy_ret = dat
            ndarray_ret = mx.nd.array(dat).broadcast_to(shape=target_shape)
            if type(ndarray_ret) is mx.ndarray.NDArray:
                ndarray_ret = ndarray_ret.asnumpy()
            assert (ndarray_ret.shape == target_shape).all()
            err = np.square(ndarray_ret - numpy_ret).mean()
            assert err < 1E-8
    test_broadcast_to()

if __name__ == '__main__':
    mx.profiler.profiler_set_config(mode='all', filename='profile_ndarray.json')
    mx.profiler.profiler_set_state('run')
    test_ndarray_slice_along_axis()
    test_broadcast()
    test_ndarray_elementwise()
    test_ndarray_slice()
    test_ndarray_pickle()
    test_ndarray_saveload()
    test_ndarray_copy()
    test_ndarray_negate()
    test_ndarray_scalar()
    test_clip()
    test_dot()
    test_ndarray_choose()
    test_ndarray_onehot()
    test_ndarray_fill()
    test_reduce()
    mx.profiler.profiler_set_state('stop')
