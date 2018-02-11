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

import mxnet as mx
import numpy as np
import os
import pickle as pkl
import unittest
from nose.tools import raises
from mxnet.test_utils import almost_equal
from mxnet.test_utils import assert_almost_equal
from mxnet.test_utils import default_context
from mxnet.test_utils import np_reduce
from mxnet.test_utils import same
from numpy.testing import assert_allclose
import mxnet.autograd

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
            assert_almost_equal(out1, out2, rtol=2e-3)
        else:
            assert_almost_equal(out1, out2)


def random_ndarray(dim):
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    data = mx.nd.array(np.random.uniform(-10, 10, shape))
    return data


def test_ndarray_setitem():
    shape = (3, 4, 2)

    # scalar assignment
    x = mx.nd.zeros(shape)
    x[:] = 1
    x_np = np.ones(shape, dtype=x.dtype)
    assert same(x.asnumpy(), x_np)

    # ndarray assignment
    x = mx.nd.zeros(shape)
    x[:] = mx.nd.ones(shape)
    x_np = np.ones(shape, dtype=x.dtype)
    assert same(x.asnumpy(), x_np)

    # numpy assignment
    x = mx.nd.zeros(shape)
    x[:] = np.ones(shape)
    x_np = np.ones(shape, dtype=x.dtype)
    assert same(x.asnumpy(), x_np)

    # indexing sub-arrays
    x = mx.nd.zeros(shape)
    x[1] = 1
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[1] = 1
    assert same(x.asnumpy(), x_np)

    # short all-dim indexing
    x = mx.nd.zeros(shape)
    val = mx.nd.ones((3, 2))
    x[:, 1:3, 1] = val
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[:, 1:3, 1] = val.asnumpy()
    assert same(x.asnumpy(), x_np)

    x = mx.nd.zeros(shape)
    x[:, 1:3, 1] = 1
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[:, 1:3, 1:2] = 1
    assert same(x.asnumpy(), x_np)


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

def test_ndarray_elementwisesum():
    ones = mx.nd.ones((10,), dtype=np.int32)
    res = mx.nd.ElementWiseSum(ones, ones*2, ones*4, ones*8)
    assert same(res.asnumpy(), ones.asnumpy()*15)

def test_ndarray_negate():
    npy = np.random.uniform(-10, 10, (2,3,4))
    arr = mx.nd.array(npy)
    assert_almost_equal(npy, arr.asnumpy())
    assert_almost_equal(-npy, (-arr).asnumpy())

    # a final check to make sure the negation (-) is not implemented
    # as inplace operation, so the contents of arr does not change after
    # we compute (-arr)
    assert_almost_equal(npy, arr.asnumpy())

def test_ndarray_reshape():
    tensor  = mx.nd.array([[[1, 2], [3, 4]],
                           [[5, 6], [7, 8]]])
    true_res = mx.nd.arange(8) + 1
    assert same(tensor.reshape((-1, )).asnumpy(), true_res.asnumpy())
    true_res  = mx.nd.array([[1, 2, 3, 4],
                             [5, 6, 7, 8]])
    assert same(tensor.reshape((2, -1)).asnumpy(), true_res.asnumpy())
    assert same(tensor.reshape((0, -1)).asnumpy(), true_res.asnumpy())
    true_res  = mx.nd.array([[1, 2],
                             [3, 4],
                             [5, 6],
                             [7, 8]])
    assert same(tensor.reshape((-1, 2)).asnumpy(), true_res.asnumpy())


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
    nrepeat = 10
    fname = 'tmp_list.bin'
    for repeat in range(nrepeat):
        data = []
        # test save/load as list
        for i in range(10):
            data.append(random_ndarray(np.random.randint(1, 5)))
        mx.nd.save(fname, data)
        data2 = mx.nd.load(fname)
        assert len(data) == len(data2)
        for x, y in zip(data, data2):
            assert np.sum(x.asnumpy() != y.asnumpy()) == 0
        # test save/load as dict
        dmap = {'ndarray xx %s' % i : x for i, x in enumerate(data)}
        mx.nd.save(fname, dmap)
        dmap2 = mx.nd.load(fname)
        assert len(dmap2) == len(dmap)
        for k, x in dmap.items():
            y = dmap2[k]
            assert np.sum(x.asnumpy() != y.asnumpy()) == 0
        # test save/load as ndarray
        # we expect the single ndarray to be converted into a list containing the ndarray
        single_ndarray = data[0]
        mx.nd.save(fname, single_ndarray)
        single_ndarray_loaded = mx.nd.load(fname)
        assert len(single_ndarray_loaded) == 1
        single_ndarray_loaded = single_ndarray_loaded[0]
        assert np.sum(single_ndarray.asnumpy() != single_ndarray_loaded.asnumpy()) == 0
    os.remove(fname)

def test_ndarray_legacy_load():
    data = []
    for i in range(6):
        data.append(mx.nd.arange(128))
    path = os.path.dirname(os.path.realpath(__file__))
    legacy_data = mx.nd.load(os.path.join(path, 'legacy_ndarray.v0'))
    assert len(data) == len(legacy_data)
    for i in range(len(data)):
        assert same(data[i].asnumpy(), legacy_data[i].asnumpy())

def test_ndarray_slice():
    shape = (10,)
    A = mx.nd.array(np.random.uniform(-10, 10, shape))
    A2 = A.asnumpy()
    assert same(A[3:8].asnumpy(), A2[3:8])
    A2[3:8] *= 10;
    A[3:8] = A2[3:8]
    assert same(A[3:8].asnumpy(), A2[3:8])

    shape = (3,4,5,6,7)
    A = mx.nd.random.uniform(shape=shape)
    A2 = A.asnumpy()

    assert same(A[1,3:4,:,1:5].asnumpy(), A2[1,3:4,:,1:5])

    assert A[1,2,3,4,5].asscalar() == A2[1,2,3,4,5]

    a = mx.nd.array([[0, 1], [2, 3]])
    assert (a[[1, 1, 0], [0, 1, 0]].asnumpy() == [2, 3, 0]).all()
    assert (a[mx.nd.array([1, 1, 0]), mx.nd.array([0, 1, 0])].asnumpy() == [2, 3, 0]).all()


def test_ndarray_crop():
    # get crop
    x = mx.nd.ones((2, 3, 4))
    y = mx.nd.crop(x, begin=(0, 0, 0), end=(2, 1, 3))
    assert same(y.asnumpy(), np.ones((2, 1, 3), dtype=y.dtype))

    # crop assign
    z = mx.nd.zeros((2, 1, 3))
    mx.nd._internal._crop_assign(x, z, begin=(0, 0, 0),
                                 end=(2, 1, 3), out=x)
    np_x = np.ones(x.shape, dtype=x.dtype)
    np_x[0:2, 0:1, 0:3] = 0
    assert same(x.asnumpy(), np_x)

    # crop assign with scalar
    x = mx.nd.ones((2, 3, 4))
    mx.nd._internal._crop_assign_scalar(x, scalar=5,
                                        begin=(0, 0, 0),
                                        end=(2, 1, 3), out=x)
    np_x = np.ones(x.shape, dtype=x.dtype)
    np_x[0:2, 0:1, 0:3] = 5
    assert same(x.asnumpy(), np_x)


def test_ndarray_concatenate():
    axis = 1
    shapes = [(2, 3, 4, 2), (2, 2, 4, 2), (2, 1, 4, 2)]
    arrays_np = [np.random.uniform(-10, 10, s).astype(np.float32) for s in shapes]
    arrays_nd = [mx.nd.array(x) for x in arrays_np]

    array_nd = mx.nd.concatenate(arrays_nd, axis=axis)
    array_np = np.concatenate(arrays_np, axis=axis)

    assert same(array_np, array_nd.asnumpy())


def test_clip():
    shape = (10,)
    A = mx.random.uniform(-10, 10, shape)
    B = mx.nd.clip(A, -2, 2)
    B1 = B.asnumpy()
    for i in range(shape[0]):
        assert B1[i] >= -2
        assert B1[i] <= 2

def test_dot():
    # Test normal dot
    a = np.random.uniform(-3, 3, (3, 4))
    b = np.random.uniform(-3, 3, (4, 5))
    c = np.dot(a, b)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B)
    assert_almost_equal(c, C.asnumpy())
    # Test dot with transpose kargs
    a = np.random.uniform(-3, 3, (3, 4))
    b = np.random.uniform(-3, 3, (3, 5))
    c = np.dot(a.T, b)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B, transpose_a=True)
    assert_almost_equal(c, C.asnumpy())
    # Test dot with transpose kargs
    a = np.random.uniform(-3, 3, (3, 4))
    b = np.random.uniform(-3, 3, (5, 4))
    c = np.dot(a, b.T)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B, transpose_b=True)
    assert_almost_equal(c, C.asnumpy())
    # Test dot with transpose kargs
    a = np.random.uniform(-3, 3, (4, 3))
    b = np.random.uniform(-3, 3, (5, 4))
    c = np.dot(a.T, b.T)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B, transpose_a=True, transpose_b=True)
    assert_almost_equal(c, C.asnumpy())



def test_reduce():
    sample_num = 200
    def test_reduce_inner(numpy_reduce_func, nd_reduce_func, multi_axes):
        for i in range(sample_num):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 11, size=ndim)
            dat = np.random.rand(*shape) - 0.5
            keepdims = np.random.randint(0, 2)
            if multi_axes:
                axis_flags = np.random.randint(0, 2, size=ndim)
                axes = []
                for (axis, flag) in enumerate(axis_flags):
                    if flag:
                        axes.append(axis)
                if 0 == len(axes):
                    axes = tuple(range(ndim))
                else:
                    axes = tuple(axes)
            else:
                axes = np.random.randint(0, ndim)
            numpy_ret = numpy_reduce_func(dat, axis=axes, keepdims=keepdims)

            ndarray_ret = nd_reduce_func(mx.nd.array(dat), axis=axes, keepdims=keepdims)
            if type(ndarray_ret) is mx.ndarray.NDArray:
                ndarray_ret = ndarray_ret.asnumpy()
            assert (ndarray_ret.shape == numpy_ret.shape) or \
                   (ndarray_ret.shape == (1,) and numpy_ret.shape == ()), "nd:%s, numpy:%s" \
                                                         %(ndarray_ret.shape, numpy_ret.shape)
            err = np.square(ndarray_ret - numpy_ret).mean()
            assert err < 1E-4
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.sum),
                      mx.nd.sum, True)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.max),
                      mx.nd.max, True)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.min),
                      mx.nd.min, True)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.argmax),
                      mx.nd.argmax, False)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.argmin),
                      mx.nd.argmin, False)

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

def test_broadcast_binary():
    N = 100
    def check_broadcast_binary(fn):
        for _ in range(N):
            ndim = np.random.randint(1, 6)
            oshape = np.random.randint(1, 6, size=(ndim,))
            bdim = np.random.randint(1, ndim+1)
            lshape = list(oshape)
            rshape = list(oshape[ndim-bdim:])
            for i in range(bdim):
                sep = np.random.uniform(0, 1)
                if sep < 0.33:
                    lshape[ndim-i-1] = 1
                elif sep < 0.66:
                    rshape[bdim-i-1] = 1
            lhs = np.random.normal(0, 1, size=lshape)
            rhs = np.random.normal(0, 1, size=rshape)
            assert_allclose(fn(lhs, rhs),
                            fn(mx.nd.array(lhs), mx.nd.array(rhs)).asnumpy(),
                            rtol=1e-4, atol=1e-4)

    check_broadcast_binary(lambda x, y: x + y)
    check_broadcast_binary(lambda x, y: x - y)
    check_broadcast_binary(lambda x, y: x * y)
    check_broadcast_binary(lambda x, y: x / y)
    check_broadcast_binary(lambda x, y: x > y)
    check_broadcast_binary(lambda x, y: x < y)
    check_broadcast_binary(lambda x, y: x >= y)
    check_broadcast_binary(lambda x, y: x <= y)
    check_broadcast_binary(lambda x, y: x == y)

def test_moveaxis():
    X = mx.nd.array([[[1, 2, 3], [4, 5, 6]],
                     [[7, 8, 9], [10, 11, 12]]])
    res = mx.nd.moveaxis(X, 0, 3).asnumpy()
    true_res = mx.nd.array([[[  1.,   7.],
                             [  2.,   8.],
                             [  3.,   9.]],
                            [[  4.,  10.],
                             [  5.,  11.],
                             [  6.,  12.]]])
    assert same(res, true_res.asnumpy())
    assert mx.nd.moveaxis(X, 2, 0).shape == (3, 2, 2)

def test_arange():
    for i in range(5):
        start = np.random.rand() * 10
        stop = start + np.random.rand() * 100
        step = np.random.rand() * 4
        repeat = int(np.random.rand() * 5) + 1
        gt = np.arange(start=start, stop=stop, step=step)
        gt = np.broadcast_to(gt.reshape((gt.shape[0], 1)), shape=(gt.shape[0], repeat)).ravel()
        pred = mx.nd.arange(start=start, stop=stop, step=step, repeat=repeat).asnumpy()
        assert_almost_equal(pred, gt)
    gt = np.arange(start=0, stop=10000**2, step=10001, dtype=np.int32)
    pred = mx.nd.arange(start=0, stop=10000**2, step=10001,
                        dtype="int32").asnumpy()
    assert_almost_equal(pred, gt)

def test_order(ctx=default_context()):
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
            assert axis is None or axis ==1
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
    a_npy = np.random.normal(size=(5, 5, 5, 5))
    a_nd = mx.nd.array(a_npy, ctx=ctx)

    # test for ret_typ=indices
    nd_ret_topk = mx.nd.topk(a_nd, axis=1, ret_typ="indices", k=3, is_ascend=True).asnumpy()
    gt = gt_topk(a_npy, axis=1, ret_typ="indices", k=3, is_ascend=True)
    assert_almost_equal(nd_ret_topk, gt)
    nd_ret_topk = mx.nd.topk(a_nd, axis=3, ret_typ="indices", k=2, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=3, ret_typ="indices", k=2, is_ascend=False)
    assert_almost_equal(nd_ret_topk, gt)
    nd_ret_topk = mx.nd.topk(a_nd, axis=None, ret_typ="indices", k=21, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=None, ret_typ="indices", k=21, is_ascend=False)
    assert_almost_equal(nd_ret_topk, gt)

    # test for ret_typ=value
    nd_ret_topk = mx.nd.topk(a_nd, axis=1, ret_typ="value", k=3, is_ascend=True).asnumpy()
    gt = gt_topk(a_npy, axis=1, ret_typ="value", k=3, is_ascend=True)
    assert_almost_equal(nd_ret_topk, gt)
    nd_ret_topk = mx.nd.topk(a_nd, axis=3, ret_typ="value", k=2, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=3, ret_typ="value", k=2, is_ascend=False)
    assert_almost_equal(nd_ret_topk, gt)
    nd_ret_topk = mx.nd.topk(a_nd, axis=None, ret_typ="value", k=21, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=None, ret_typ="value", k=21, is_ascend=False)
    assert_almost_equal(nd_ret_topk, gt)

    # test for ret_typ=mask
    nd_ret_topk = mx.nd.topk(a_nd, axis=1, ret_typ="mask", k=3, is_ascend=True).asnumpy()
    gt = gt_topk(a_npy, axis=1, ret_typ="mask", k=3, is_ascend=True)
    assert_almost_equal(nd_ret_topk, gt)
    nd_ret_topk = mx.nd.topk(a_nd, axis=1, ret_typ="mask", k=2, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=1, ret_typ="mask", k=2, is_ascend=False)
    assert_almost_equal(nd_ret_topk, gt)
    nd_ret_topk = mx.nd.topk(a_nd, axis=None, ret_typ="mask", k=21, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=None, ret_typ="mask", k=21, is_ascend=False)
    assert_almost_equal(nd_ret_topk, gt)

    # test for ret_typ=both
    nd_ret_topk_val, nd_ret_topk_ind = mx.nd.topk(a_nd, axis=1, ret_typ="both", k=3, is_ascend=True)
    nd_ret_topk_val = nd_ret_topk_val.asnumpy()
    nd_ret_topk_ind = nd_ret_topk_ind.asnumpy()
    gt_val = gt_topk(a_npy, axis=1, ret_typ="value", k=3, is_ascend=True)
    gt_ind = gt_topk(a_npy, axis=1, ret_typ="indices", k=3, is_ascend=True)
    assert_almost_equal(nd_ret_topk_val, gt_val)
    assert_almost_equal(nd_ret_topk_ind, gt_ind)

    # test for sort
    nd_ret_sort = mx.nd.sort(a_nd, axis=1, is_ascend=True).asnumpy()
    gt = gt_topk(a_npy, axis=1, ret_typ="value", k=5, is_ascend=True)
    assert_almost_equal(nd_ret_sort, gt)
    nd_ret_sort = mx.nd.sort(a_nd, axis=None, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=None, ret_typ="value", k=5*5*5*5, is_ascend=False)
    assert_almost_equal(nd_ret_sort, gt)

    # test for argsort
    nd_ret_argsort = mx.nd.argsort(a_nd, axis=3, is_ascend=True).asnumpy()
    gt = gt_topk(a_npy, axis=3, ret_typ="indices", k=5, is_ascend=True)
    assert_almost_equal(nd_ret_argsort, gt)
    nd_ret_argsort = mx.nd.argsort(a_nd, axis=None, is_ascend=False).asnumpy()
    gt = gt_topk(a_npy, axis=None, ret_typ="indices", k=5*5*5*5, is_ascend=False)
    assert_almost_equal(nd_ret_argsort, gt)

    # test topk with a big shape
    a = mx.nd.arange(0, 54686454, step=1, repeat=1)
    assert_almost_equal(a.topk(k=54686454).asnumpy(), a.asnumpy()[::-1])

def test_ndarray_equal():
    x = mx.nd.zeros((2, 3))
    y = mx.nd.ones((2, 3))
    z = x == y
    assert (z.asnumpy() == np.zeros((2, 3))).all()
    z = 0 == x
    assert (z.asnumpy() == np.ones((2, 3))).all()

def test_ndarray_not_equal():
    x = mx.nd.zeros((2, 3))
    y = mx.nd.ones((2, 3))
    z = x != y
    assert (z.asnumpy() == np.ones((2, 3))).all()
    z = 0 != x
    assert (z.asnumpy() == np.zeros((2, 3))).all()

def test_ndarray_greater():
    x = mx.nd.zeros((2, 3))
    y = mx.nd.ones((2, 3))
    z = x > y
    assert (z.asnumpy() == np.zeros((2, 3))).all()
    z = y > 0
    assert (z.asnumpy() == np.ones((2, 3))).all()
    z = 0 > y
    assert (z.asnumpy() == np.zeros((2, 3))).all()

def test_ndarray_greater_equal():
    x = mx.nd.zeros((2, 3))
    y = mx.nd.ones((2, 3))
    z = x >= y
    assert (z.asnumpy() == np.zeros((2, 3))).all()
    z = y >= 0
    assert (z.asnumpy() == np.ones((2, 3))).all()
    z = 0 >= y
    assert (z.asnumpy() == np.zeros((2, 3))).all()
    z = y >= 1
    assert (z.asnumpy() == np.ones((2, 3))).all()

def test_ndarray_lesser():
    x = mx.nd.zeros((2, 3))
    y = mx.nd.ones((2, 3))
    z = y < x
    assert (z.asnumpy() == np.zeros((2, 3))).all()
    z = 0 < y
    assert (z.asnumpy() == np.ones((2, 3))).all()
    z = y < 0
    assert (z.asnumpy() == np.zeros((2, 3))).all()

def test_ndarray_lesser_equal():
    x = mx.nd.zeros((2, 3))
    y = mx.nd.ones((2, 3))
    z = y <= x
    assert (z.asnumpy() == np.zeros((2, 3))).all()
    z = 0 <= y
    assert (z.asnumpy() == np.ones((2, 3))).all()
    z = y <= 0
    assert (z.asnumpy() == np.zeros((2, 3))).all()
    z = 1 <= y
    assert (z.asnumpy() == np.ones((2, 3))).all()

def test_take():
    for data_ndim in range(2, 5):
        for idx_ndim in range(1, 4):
            data_shape = ()
            for _ in range(data_ndim):
                data_shape += (np.random.randint(low=3, high=6), )
            data_real = np.random.normal(size=data_shape).astype('float32')
            idx_shape = ()
            for _ in range(idx_ndim):
                idx_shape += (np.random.randint(low=3, high=5), )
            idx_real = np.random.randint(low=0, high=data_shape[0], size=idx_shape)
            data_real_mx = mx.nd.array(data_real)
            idx_real_mx = mx.nd.array(idx_real)
            result = mx.nd.take(data_real_mx, idx_real_mx)
            assert_almost_equal(result.asnumpy(), data_real[idx_real])


def test_iter():
    x = mx.nd.array([1, 2, 3])
    y = []
    for a in x:
        y.append(a)

    for i in range(x.size):
        assert same(y[i].asnumpy(), x[i].asnumpy())

@unittest.skip("test fails intermittently. temporarily disabled till it gets fixed. tracked at https://github.com/apache/incubator-mxnet/issues/8049")
def test_cached():
    sym = mx.sym.Convolution(kernel=(3, 3), num_filter=10) + 2
    op = mx.nd.CachedOp(sym)
    data = mx.nd.ones((3, 4, 10, 10))
    weight = mx.nd.ones((10, 4, 3, 3))
    bias = mx.nd.ones((10,))
    o1 = op(data, weight, bias)
    bias[:] = 2
    o2 = op(data, weight, bias)
    assert_almost_equal(o2.asnumpy(), o1.asnumpy()+1)
    o2[:] = 0
    op(data, weight, bias, out=o2)
    assert_almost_equal(o2.asnumpy(), o1.asnumpy()+1)

    weight.attach_grad()
    bias.attach_grad()
    with mx.autograd.record():
        bias = bias + 1
        o = op(data, weight, bias)
        o = o * 2
        o.backward()

    with mx.autograd.record():
        bias = bias + 1
        o = op(data, weight, bias)
        o = o * 2
        o.backward(retain_graph=True)
        o.backward()

    # try a different shape
    data = mx.nd.ones((5, 2, 10, 10))
    weight = mx.nd.ones((10, 2, 3, 3))
    bias = mx.nd.ones((10,))
    data.attach_grad()

    with mx.autograd.record():
        bias = bias + 1
        o = op(data, weight, bias)
        o = o * 2
        o.backward()


def test_output():
    shape = (2,2)
    ones = mx.nd.ones(shape)
    zeros = mx.nd.zeros(shape)
    out = mx.nd.zeros(shape)
    mx.nd.ones(shape, out=out)
    assert_almost_equal(out.asnumpy(), ones.asnumpy())
    mx.nd.zeros(shape, out=out)
    assert_almost_equal(out.asnumpy(), zeros.asnumpy())
    mx.nd.full(shape, 2, out=out)
    assert_almost_equal(out.asnumpy(), ones.asnumpy() * 2)
    arange_out = mx.nd.arange(0, 20, dtype='int64')
    assert_almost_equal(arange_out.asnumpy(), np.arange(0, 20))

def test_ndarray_fluent():
    has_grad = set(['flatten', 'expand_dims', 'flip', 'tile', 'transpose', 'sum', 'nansum', 'prod',
                    'nanprod', 'mean', 'max', 'min', 'reshape', 'broadcast_to', 'split',
                    'broadcast_axes', 'pad', 'swapaxes', 'slice', 'slice_axis', 'take',
                    'one_hot', 'pick', 'sort', 'topk', 'argsort', 'argmax', 'argmin',
                    'clip', 'abs', 'sign', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                    'degrees', 'radians', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                    'exp', 'expm1', 'log', 'log10', 'log2', 'log1p', 'sqrt', 'rsqrt', 'square',
                    'reshape_like', 'cbrt', 'rcbrt', 'relu', 'sigmoid', 'softmax', 'log_softmax',
                    'reciprocal'])
    def check_fluent_regular(func, kwargs, shape=(5, 17, 1), equal_nan=False):
        with mx.name.NameManager():
            data = mx.nd.random_uniform(shape=shape, ctx=default_context())
            regular = getattr(mx.ndarray, func)(data, **kwargs)
            fluent = getattr(data, func)(**kwargs)
            if isinstance(regular, list):
                for r, f in zip(regular, fluent):
                    assert almost_equal(r.asnumpy(), f.asnumpy(), equal_nan=equal_nan)
            else:
                assert almost_equal(regular.asnumpy(), fluent.asnumpy(), equal_nan=equal_nan)

    for func in ['flatten', 'norm', 'round', 'rint', 'fix', 'floor', 'ceil', 'trunc', 'zeros_like',
                 'ones_like', 'abs', 'sign', 'sin', 'cos', 'degrees', 'radians',
                 'exp', 'expm1', 'square', 'reciprocal', 'argmax_channel']:
        check_fluent_regular(func, {})

    for func in ['arccosh', 'arcsin', 'arccos', 'arctan', 'tan', 'sinh', 'cosh', 'tanh',
                 'arcsinh', 'arctanh', 'log', 'log10', 'log2', 'log1p', 'sqrt', 'rsqrt',
                 'cbrt', 'rcbrt', 'relu', 'sigmoid', 'softmax', 'log_softmax']:
        check_fluent_regular(func, {}, equal_nan=True)

    for func in ['expand_dims', 'flip', 'sort', 'topk', 'argsort', 'argmax', 'argmin']:
        check_fluent_regular(func, {'axis': 1})

    check_fluent_regular('one_hot', {'depth': 15})
    check_fluent_regular('tile', {'reps': (1,2)})
    check_fluent_regular('repeat', {'repeats': 3})
    check_fluent_regular('transpose', {'axes': (1,0,2)})
    check_fluent_regular('split', {'axis': 2, 'num_outputs': 3}, shape=(5, 17, 6))
    check_fluent_regular('slice', {'begin': (2, 5, 1), 'end': (4, 7, 6)}, shape=(5, 17, 6))
    check_fluent_regular('slice_axis', {'axis': 1, 'begin': 5, 'end': 7})
    check_fluent_regular('take', {'indices': mx.nd.array([2, 3])})
    check_fluent_regular('pick', {'axis': 1, 'index': mx.nd.array([[2], [3], [5], [6], [11]])})
    check_fluent_regular('clip', {'a_min': 0.25, 'a_max': 0.75})
    check_fluent_regular('broadcast_axes', {'axis': (2,), 'size': (5,)})
    check_fluent_regular('pad', {'mode': 'constant', 'pad_width': (0,0,0,0,3,0,0,4)}, shape=(5, 17, 2, 3))
    check_fluent_regular('reshape_like', {'rhs': mx.nd.ones((30, 17))}, shape=(5, 17, 2, 3))

    for func in ['sum', 'nansum', 'prod', 'nanprod', 'mean', 'max', 'min']:
        check_fluent_regular(func, {'axis': (1, 2)})

    check_fluent_regular('reshape', {'shape': (17, 1, 5)})
    check_fluent_regular('broadcast_to', {'shape': (5, 17, 47)})
    check_fluent_regular('squeeze', {'axis': (1, 3)}, shape=(2, 1, 3, 1, 4))

@raises(ValueError)
def test_bool_ambiguous():
    bool(mx.nd.ones((2,3,4)))

def test_bool():
    assert not bool(mx.nd.array([]))
    assert not bool(mx.nd.zeros((1,)))
    assert bool(mx.nd.ones((1,)))


def test_ndarray_indexing():
    def test_getitem(np_array, index, is_scalar=False):
        """`is_scalar` indicates whether we should expect a scalar for the result.
        If so, the indexed array of NDArray should call asscalar to compare
        with numpy's indexed array."""
        np_index = index
        if isinstance(index, mx.nd.NDArray):
            np_index = index.asnumpy()
        if isinstance(index, tuple):
            np_index = []
            for idx in index:
                if isinstance(idx, mx.nd.NDArray):
                    np_index.append(idx.asnumpy())
                else:
                    np_index.append(idx)
            np_index = tuple(np_index)

        np_indexed_array = np_array[np_index]
        mx_array = mx.nd.array(np_array, dtype=np_array.dtype)
        mx_indexed_array = mx_array[index]
        if is_scalar:
            mx_indexed_array = mx_indexed_array.asscalar()
        else:
            mx_indexed_array = mx_indexed_array.asnumpy()
        assert same(np_indexed_array, mx_indexed_array), 'Failed with index=%s' % str(index)

    def test_setitem(np_array, index, is_scalar):
        def assert_same(np_array, np_index, mx_array, mx_index, mx_value, np_value=None):
            if np_value is not None:
                np_array[np_index] = np_value
            else:
                np_array[np_index] = mx_value
            mx_array[mx_index] = mx_value
            assert same(np_array, mx_array.asnumpy())

        np_index = index
        if isinstance(index, mx.nd.NDArray):
            np_index = index.asnumpy()
        if isinstance(index, tuple):
            np_index = []
            for idx in index:
                if isinstance(idx, mx.nd.NDArray):
                    np_index.append(idx.asnumpy())
                else:
                    np_index.append(idx)
            np_index = tuple(np_index)

        mx_array = mx.nd.array(np_array, dtype=np_array.dtype)
        np_array = mx_array.asnumpy()
        if is_scalar:
            # test value is a numeric type
            assert_same(np_array, np_index, mx_array, index, np.random.randint(low=-10000, high=0))
            value_nd = [np.random.randint(low=-10000, high=0)]
            assert_same(np_array, np_index, mx_array, index, value_nd, value_nd[0])
        else:
            indexed_array_shape = np_array[np_index].shape
            np_indexed_array = np.random.randint(low=-10000, high=0, size=indexed_array_shape)
            # test value is a numpy array without broadcast
            assert_same(np_array, np_index, mx_array, index, np_indexed_array)
            # test value is an numeric_type
            assert_same(np_array, np_index, mx_array, index, np.random.randint(low=-10000, high=0))
            if len(indexed_array_shape) > 1:
                # test numpy array with broadcast
                assert_same(np_array, np_index, mx_array, index,
                            np.random.randint(low=-10000, high=0, size=(indexed_array_shape[-1],)))
                # test list with broadcast
                assert_same(np_array, np_index, mx_array, index,
                            [np.random.randint(low=-10000, high=0)] * indexed_array_shape[-1])

    def test_getitem_autograd(np_array, index):
        x = mx.nd.array(np_array, dtype=np_array.dtype)
        x.attach_grad()
        with mx.autograd.record():
            y = x[index]
        y.backward()
        value = mx.nd.ones_like(y)
        x_grad = mx.nd.zeros_like(x)
        x_grad[index] = value
        assert same(x_grad.asnumpy(), x.grad.asnumpy())

    shape = (8, 16, 9, 9)
    np_array = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    # index_list is a list of tuples. The tuple's first element is the index, the second one is a boolean value
    # indicating whether we should expect the result as a scalar compared to numpy.
    index_list = [(0, False), (5, False), (-1, False),
                  (slice(5), False), (slice(1, 5), False), (slice(1, 5, 2), False),
                  (slice(7, 0, -1), False), (slice(None, 6), False), (slice(None, 6, 3), False),
                  (slice(1, None), False), (slice(1, None, 3), False), (slice(None, None, 2), False),
                  (slice(None, None, -1), False), (slice(None, None, -2), False),
                  ((slice(None), slice(None), 1, 8), False),
                  ((slice(None), 2, slice(1, 5), 1), False),
                  ((1, 2, 3), False), ((1, 2, 3, 4), True),
                  ((slice(None, None, -1), 2, slice(1, 5), 1), False),
                  ((slice(None, None, -1), 2, slice(1, 7, 2), 1), False),
                  ((slice(1, 8, 2), slice(14, 2, -2), slice(3, 8), slice(0, 7, 3)), False),
                  ((slice(1, 8, 2), 1, slice(3, 8), 2), False),
                  ([1], False), ([1, 2], False), ([2, 1, 3], False), ([7, 5, 0, 3, 6, 2, 1], False),
                  (np.array([6, 3], dtype=np.int32), False),
                  (np.array([[3, 4], [0, 6]], dtype=np.int32), False),
                  (np.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int32), False),
                  (np.array([[2], [0], [1]], dtype=np.int32), False),
                  (mx.nd.array([4, 7], dtype=np.int32), False),
                  (mx.nd.array([[3, 6], [2, 1]], dtype=np.int32), False),
                  (mx.nd.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int32), False),
                  ((1, [2, 3]), False), ((1, [2, 3], np.array([[3], [0]], dtype=np.int32)), False),
                  ((1, [2], np.array([[5], [3]], dtype=np.int32), slice(None)), False),
                  ((1, [2, 3], np.array([[6], [0]], dtype=np.int32), slice(2, 5)), False),
                  ((1, [2, 3], np.array([[4], [7]], dtype=np.int32), slice(2, 5, 2)), False),
                  ((1, [2], np.array([[3]], dtype=np.int32), slice(None, None, -1)), False),
                  ((1, [2], np.array([[3]], dtype=np.int32), np.array([[5, 7], [2, 4]], dtype=np.int64)), False),
                  ((1, [2], mx.nd.array([[4]], dtype=np.int32), mx.nd.array([[1, 3], [5, 7]], dtype='int64')),
                   False),
                  ([0], False), ([0, 1], False), ([1, 2, 3], False), ([2, 0, 5, 6], False),
                  (([1, 1], [2, 3]), False), (([1], [4], [5]), False), (([1], [4], [5], [6]), False),
                  (([[1]], [[2]]), False), (([[1]], [[2]], [[3]], [[4]]), False),
                  ((slice(0, 2), [[1], [6]], slice(0, 2), slice(0, 5, 2)), False),
                  (([[[[1]]]], [[1]], slice(0, 3), [1, 5]), False),
                  (([[[[1]]]], 3, slice(0, 3), [1, 3]), False),
                  (([[[[1]]]], 3, slice(0, 3), 0), False),
                  (([[[[1]]]], [[2], [12]], slice(0, 3), slice(None)), False),
                  (([1, 2], slice(3, 5), [2, 3], [3, 4]), False),
                  (([1, 2], slice(3, 5), (2, 3), [3, 4]), False)]
    for index in index_list:
        test_getitem(np_array, index[0], index[1])
        test_setitem(np_array, index[0], index[1])
        test_getitem_autograd(np_array, index[0])


def test_assign_float_value_to_ndarray():
    """Test case from https://github.com/apache/incubator-mxnet/issues/8668"""
    a = np.array([47.844944], dtype=np.float32)
    b = mx.nd.zeros(1, dtype=np.float32)
    b[0] = a
    assert same(a, b.asnumpy())
    b[0] = a[0]
    assert same(a, b.asnumpy())


if __name__ == '__main__':
    import nose
    nose.runmodule()
