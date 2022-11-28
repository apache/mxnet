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
from distutils.version import LooseVersion
from itertools import permutations, combinations_with_replacement
import os
import pickle as pkl
import random
import functools
import pytest
from common import assertRaises, TemporaryDirectory
from mxnet.test_utils import almost_equal
from mxnet.test_utils import assert_almost_equal, assert_exception
from mxnet.test_utils import default_device
from mxnet.test_utils import np_reduce
from mxnet.test_utils import same
from mxnet.test_utils import random_sample, rand_shape_nd, random_arrays
from mxnet import runtime
from numpy.testing import assert_allclose, assert_array_equal, assert_array_almost_equal
import mxnet.autograd
from mxnet.base import integer_types
from mxnet.ndarray.ndarray import py_slice
from mxnet.amp.amp import bfloat16


def check_with_uniform(uf, arg_shapes, dim=None, npuf=None, rmin=-10, type_list=[np.float32]):
    """check function consistency with uniform random numbers"""
    if isinstance(arg_shapes, int):
        assert dim
        shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
        arg_shapes = [shape] * arg_shapes

    if npuf is None:
        npuf = uf

    for dtype in type_list:
        ndarray_arg = []
        numpy_arg = []
        for s in arg_shapes:
            npy = np.random.uniform(rmin, 10, s).astype(dtype)
            narr = mx.nd.array(npy, dtype=dtype)
            ndarray_arg.append(narr)
            numpy_arg.append(npy)
        out1 = uf(*ndarray_arg)
        out2 = npuf(*numpy_arg).astype(dtype)

        assert out1.shape == out2.shape
        if isinstance(out1, mx.nd.NDArray):
            out1 = out1.asnumpy()
        if dtype == np.float16:
            assert_almost_equal(out1, out2, rtol=2e-3, atol=1e-5)
        else:
            assert_almost_equal(out1, out2, atol=1e-5)


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
    x[-1] = 1
    x_np[-1] = 1
    assert same(x.asnumpy(), x_np)

    # Ellipsis
    x = mx.nd.zeros(shape)
    x[2, ...] = 1
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[2, ...] = 1
    assert same(x.asnumpy(), x_np)

    x = mx.nd.zeros(shape)
    x[..., 1] = 1
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[..., 1] = 1
    assert same(x.asnumpy(), x_np)

    # `None` should be ignored
    x = mx.nd.zeros(shape)
    x[None, 0, None, None, 0, 0, None] = 1
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[None, 0, None, None, 0, 0, None] = 1
    assert same(x.asnumpy(), x_np)

    # short all-dim indexing
    x = mx.nd.zeros(shape)
    val = mx.nd.ones((3, 2))
    x[:, 1:3, 1] = val
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[:, 1:3, 1] = val.asnumpy()
    assert same(x.asnumpy(), x_np)
    x[:, 1:3, -1] = val
    x_np[:, 1:3, -1] = val.asnumpy()
    assert same(x.asnumpy(), x_np)

    x = mx.nd.zeros(shape)
    x[:, 1:3, 1:2] = 1
    x_np = np.zeros(shape, dtype=x.dtype)
    x_np[:, 1:3, 1:2] = 1
    assert same(x.asnumpy(), x_np)
    x[:, -3:-1, -2:-1] = 1
    x_np[:, -3:-1, -2:-1] = 1
    assert same(x.asnumpy(), x_np)

    # Assignments for empty axes
    for trivial_shape in [(1,), (1, 1), (1, 1, 1)]:
        x = mx.nd.zeros(trivial_shape)
        x[:] = np.ones(trivial_shape)
        x_np = np.ones(trivial_shape, dtype=x.dtype)
        assert x.shape == trivial_shape
        assert same(x.asnumpy(), x_np)

    # test https://github.com/apache/mxnet/issues/16647
    dst = mx.nd.zeros((1, 3, 1))  # destination array
    src = [1, 2, 3]
    dst[0, :len(src), 0] = src
    assert same(dst.asnumpy(), np.array([1, 2, 3], dtype=dst.dtype).reshape(dst.shape))

    dst = mx.nd.zeros((1, 3, 1))  # destination array
    src = [1, 2, 3]
    dst[0, :len(src), 0] = mx.nd.array(src)
    assert same(dst.asnumpy(), np.array([1, 2, 3], dtype=dst.dtype).reshape(dst.shape))

    dst = mx.nd.zeros((1, 3, 1))  # destination array
    src = [1, 2]
    dst[0, :len(src), 0] = src
    assert same(dst.asnumpy(), np.array([1, 2, 0], dtype=dst.dtype).reshape(dst.shape))


def test_ndarray_elementwise():
    nrepeat = 10
    maxdim = 4
    all_type = [np.float32, np.float64, np.float16, np.uint8, np.int8, np.int32, np.int64]
    real_type = [np.float32, np.float64, np.float16]
    for _ in range(nrepeat):
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


def test_ndarray_magic_abs():
    for dim in range(1, 7):
        shape = rand_shape_nd(dim)
        npy = np.random.uniform(-10, 10, shape)
        arr = mx.nd.array(npy)
        assert_almost_equal(abs(arr).asnumpy(), arr.abs().asnumpy())


def test_ndarray_reshape():
    tensor = (mx.nd.arange(30) + 1).reshape(2, 3, 5)
    true_res = mx.nd.arange(30) + 1
    assert same(tensor.reshape((-1,)).asnumpy(), true_res.asnumpy())
    assert same(tensor.reshape((2, -1)).asnumpy(), true_res.reshape(2, 15).asnumpy())
    assert same(tensor.reshape((0, -1)).asnumpy(), true_res.reshape(2, 15).asnumpy())
    assert same(tensor.reshape((-1, 2)).asnumpy(), true_res.reshape(15, 2).asnumpy())
    assert same(tensor.reshape(6, 5).asnumpy(), true_res.reshape(6, 5).asnumpy())
    assert same(tensor.reshape(-1, 2).asnumpy(), true_res.reshape(15, 2).asnumpy())
    assert same(tensor.reshape(-1).asnumpy(), true_res.asnumpy())
    assert same(tensor.reshape(30).asnumpy(), true_res.asnumpy())
    assert same(tensor.reshape(0, -1).asnumpy(), true_res.reshape(2, 15).asnumpy())
    assert same(tensor.reshape(-1, 6).asnumpy(), true_res.reshape(5, 6).asnumpy())
    assert same(tensor.reshape(-2,).asnumpy(), true_res.reshape(2, 3, 5).asnumpy())
    assert same(tensor.reshape(-3, -1).asnumpy(), true_res.reshape(6, 5).asnumpy())
    assert same(tensor.reshape(-1, 15).reshape(0, -4, 3, -1).asnumpy(), true_res.reshape(2, 3, 5).asnumpy())
    assert same(tensor.reshape(-1, 0).asnumpy(), true_res.reshape(10, 3).asnumpy())
    assert same(tensor.reshape(-1, 0, reverse=True).asnumpy(), true_res.reshape(6, 5).asnumpy())
    # https://github.com/apache/mxnet/issues/18886
    assertRaises(ValueError, tensor.reshape, (2, 3))

def test_ndarray_flatten():
    tensor = (mx.nd.arange(30) + 1).reshape(2, 3, 5)
    copy = tensor.flatten()
    ref = tensor.flatten(inplace=True)
    assert same(copy.asnumpy(), tensor.reshape(2, 15).asnumpy())
    assert same(ref.asnumpy(), tensor.reshape(2, 15).asnumpy())

    tensor[0] = -1
    assert not same(copy.asnumpy(), tensor.reshape(2, 15).asnumpy())
    assert same(ref.asnumpy(), tensor.reshape(2, 15).asnumpy())


def test_ndarray_squeeze():
    def check_squeeze(shape, axis=None):
        data = mx.random.uniform(low=-10.0, high=10.0, shape=shape)
        copy = data.squeeze(axis=axis)
        ref = data.squeeze(axis=axis, inplace=True)
        out_expected = np.squeeze(data.asnumpy(), axis=axis)
        if copy.shape == (1,):  # as an exception (1, 1, 1) will be squeezed to (1,)
            out_expected = np.squeeze(data.asnumpy(), axis=tuple([i for i in range(1, len(shape))]))
        assert same(copy.asnumpy(), out_expected)
        assert same(ref.asnumpy(), out_expected)
        data[0][0] = -1
        assert same(copy.asnumpy(), out_expected)
        assert not same(ref.asnumpy(), out_expected)

    # check forward
    check_squeeze((1, 5, 1, 3, 1), 0)
    check_squeeze((1, 5, 1, 3, 1), 2)
    check_squeeze((1, 5, 1, 3, 1), 4)
    check_squeeze((1, 5, 1, 3, 1), (0, 4))
    check_squeeze((1, 5, 1, 3, 1), (0, 2, 4))
    check_squeeze((1, 5, 1, 3, 1), -5)
    check_squeeze((1, 5, 1, 3, 1), -3)
    check_squeeze((1, 5, 1, 3, 1), -1)
    check_squeeze((1, 5, 1, 3, 1), (0, 4))
    check_squeeze((1, 5, 1, 3, 1), (0, 2, 4))
    check_squeeze((1, 5, 1, 3, 1))
    check_squeeze((1, 1, 1, 1))


def test_ndarray_expand_dims():
    for ndim in range(1, 6):
        for axis in range(-ndim-1, ndim+1):
            shape = list(np.random.randint(1, 10, size=ndim))
            data = mx.random.normal(shape=shape)
            copy = data.expand_dims(axis=axis)
            ref = data.expand_dims(axis=axis, inplace=True)
            out_expected = np.expand_dims(data.asnumpy(), axis=axis)
            assert same(copy.asnumpy(), out_expected)
            assert same(ref.asnumpy(), out_expected), (shape, axis, ref.asnumpy().shape, out_expected.shape)
            data[0] = -1
            assert same(copy.asnumpy(), out_expected)
            assert not same(ref.asnumpy(), out_expected)


def test_ndarray_choose():
    shape = (100, 20)
    npy = np.arange(np.prod(shape)).reshape(shape)
    arr = mx.nd.array(npy)
    nrepeat = 3
    for _ in range(nrepeat):
        indices = np.random.randint(shape[1], size=shape[0])
        assert same(npy[np.arange(shape[0]), indices],
                    mx.nd.choose_element_0index(arr, mx.nd.array(indices)).asnumpy())


def test_ndarray_fill():
    shape = (100, 20)
    npy = np.arange(np.prod(shape)).reshape(shape)
    arr = mx.nd.array(npy)
    new_npy = npy.copy()
    nrepeat = 3
    for _ in range(nrepeat):
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
    for _ in range(nrepeat):
        indices = np.random.randint(shape[1], size=shape[0])
        npy[:] = 0.0
        npy[np.arange(shape[0]), indices] = 1.0
        mx.nd.onehot_encode(mx.nd.array(indices), out=arr)
        assert same(npy, arr.asnumpy())


def test_init_from_scalar():
    npy = np.ones([])
    arr = mx.nd.array(npy)
    assert arr.shape == ()
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
    maxdim = 5
    for dim in range(1, maxdim):
        a = random_ndarray(dim)
        b = mx.nd.empty(a.shape)
        a[:] = np.random.uniform(-10, 10, a.shape)
        b[:] = np.random.uniform(-10, 10, a.shape)
        a = a + b
        data = pkl.dumps(a)
        a2 = pkl.loads(data)
        assert np.sum(a.asnumpy() != a2.asnumpy()) == 0


@pytest.mark.parametrize('save_fn', [mx.nd.save, mx.npx.savez])
def test_ndarray_saveload(save_fn):
    nrepeat = 10
    fname = 'tmp_list'
    for _ in range(nrepeat):
        data = []
        # test save/load as list
        for _ in range(10):
            data.append(random_ndarray(np.random.randint(1, 5)))
        if save_fn is mx.nd.save:
            save_fn(fname, data)
        else:
            save_fn(fname, *data)
        data2 = mx.nd.load(fname)
        assert len(data) == len(data2)
        for x, y in zip(data, data2 if save_fn is mx.nd.save else data2.values()):
            assert np.sum(x.asnumpy() != y.asnumpy()) == 0
        # test save/load as dict
        dmap = {f'ndarray xx {i}' : x for i, x in enumerate(data)}
        if save_fn is mx.nd.save:
            save_fn(fname, dmap)
        else:
            save_fn(fname, **dmap)
        dmap2 = mx.nd.load(fname)
        assert len(dmap2) == len(dmap)
        for k, x in dmap.items():
            y = dmap2[k]
            assert np.sum(x.asnumpy() != y.asnumpy()) == 0

        # test save/load as ndarray
        # we expect the single ndarray to be converted into a list containing the ndarray
        single_ndarray = data[0]
        save_fn(fname, single_ndarray)

        # Test loading with numpy
        if save_fn is mx.npx.savez:
            with np.load(fname) as fname_np_loaded:
                single_ndarray_loaded = fname_np_loaded['arr_0']
            assert np.sum(single_ndarray.asnumpy() != single_ndarray_loaded) == 0

            mx.npx.save(fname, single_ndarray)
            single_ndarray_loaded = np.load(fname)
            assert np.sum(single_ndarray.asnumpy() != single_ndarray_loaded) == 0

        # Test loading with mxnet backend
        single_ndarray_loaded = mx.nd.load(fname)
        assert len(single_ndarray_loaded) == 1
        single_ndarray_loaded = single_ndarray_loaded[0]
        assert np.sum(single_ndarray.asnumpy() != single_ndarray_loaded.asnumpy()) == 0

    os.remove(fname)


@mx.util.use_np
def test_ndarray_load_fortran_order(tmp_path):
    arr = np.arange(20).reshape((2, 10)).T
    assert np.isfortran(arr)
    np.save(tmp_path / 'fortran_order.npy', arr)

    mx_arr = mx.npx.load(str(tmp_path / 'fortran_order.npy'))
    np_mx_arr = mx_arr.asnumpy()
    assert not np.isfortran(np_mx_arr)
    assert np.sum(np_mx_arr != arr) == 0


def test_ndarray_legacy_load():
    data = []
    for _ in range(6):
        data.append(mx.nd.arange(128))
    path = os.path.dirname(os.path.realpath(__file__))
    legacy_data = mx.nd.load(os.path.join(path, 'legacy_ndarray.v0'))
    assert len(data) == len(legacy_data)
    for i in range(len(data)):
        assert same(data[i].asnumpy(), legacy_data[i].asnumpy())


def test_buffer_load():
    nrepeat = 10
    with TemporaryDirectory(prefix='test_buffer_load_') as tmpdir:
        for repeat in range(nrepeat):
            # test load_buffer as list
            data = []
            for _ in range(10):
                data.append(random_ndarray(np.random.randint(1, 5)))
            fname = os.path.join(tmpdir, 'list_{0}.param'.format(repeat))
            mx.nd.save(fname, data)
            with open(fname, 'rb') as dfile:
                buf_data = dfile.read()
                data2 = mx.nd.load_frombuffer(buf_data)
                assert len(data) == len(data2)
                for x, y in zip(data, data2):
                    assert np.sum(x.asnumpy() != y.asnumpy()) == 0
                # test garbage values
                assertRaises(mx.base.MXNetError,  mx.nd.load_frombuffer, buf_data[:-10])
            # test load_buffer as dict
            dmap = {f'ndarray xx {i}' : x for i, x in enumerate(data)}
            fname = os.path.join(tmpdir, 'dict_{0}.param'.format(repeat))
            mx.nd.save(fname, dmap)
            with open(fname, 'rb') as dfile:
                buf_dmap = dfile.read()
                dmap2 = mx.nd.load_frombuffer(buf_dmap)
                assert len(dmap2) == len(dmap)
                for k, x in dmap.items():
                    y = dmap2[k]
                    assert np.sum(x.asnumpy() != y.asnumpy()) == 0
                # test garbage values
                assertRaises(mx.base.MXNetError,  mx.nd.load_frombuffer, buf_dmap[:-10])

            # we expect the single ndarray to be converted into a list containing the ndarray
            single_ndarray = data[0]
            fname = os.path.join(tmpdir, 'single_{0}.param'.format(repeat))
            mx.nd.save(fname, single_ndarray)
            with open(fname, 'rb') as dfile:
                buf_single_ndarray = dfile.read()
                single_ndarray_loaded = mx.nd.load_frombuffer(buf_single_ndarray)
                assert len(single_ndarray_loaded) == 1
                single_ndarray_loaded = single_ndarray_loaded[0]
                assert np.sum(single_ndarray.asnumpy() != single_ndarray_loaded.asnumpy()) == 0
                # test garbage values
                assertRaises(mx.base.MXNetError,  mx.nd.load_frombuffer, buf_single_ndarray[:-10])


@pytest.mark.serial
def test_ndarray_slice():
    shape = (10,)
    A = mx.nd.array(np.random.uniform(-10, 10, shape))
    A2 = A.asnumpy()
    assert same(A[3:8].asnumpy(), A2[3:8])
    A2[3:8] *= 10
    A[3:8] = A2[3:8]
    assert same(A[3:8].asnumpy(), A2[3:8])

    shape = (3,4,5,6,7)
    A = mx.nd.random.uniform(shape=shape)
    A2 = A.asnumpy()

    assert same(A[1,3:4,:,1:5].asnumpy(), A2[1,3:4,:,1:5])

    assert A[1,2,3,4,5].asscalar() == A2[1,2,3,4,5]
    assert A[-1,-2,-3,-4,-5].asscalar() == A2[-1,-2,-3,-4,-5]

    a = mx.nd.array([[0, 1], [2, 3]])
    assert (a[[1, 1, 0], [0, 1, 0]].asnumpy() == [2, 3, 0]).all()
    assert (a[mx.nd.array([1, 1, 0]), mx.nd.array([0, 1, 0])].asnumpy() == [2, 3, 0]).all()

    shape = (4, 4)
    A = mx.nd.random.uniform(shape=shape)
    A2 = A.asnumpy()
    for i in range(-4, 0):
        assert A[i, i].asscalar() == A2[i, i]
        assert same(A[:, i].asnumpy(), A2[:, i])
        assert same(A[i, :].asnumpy(), A2[i, :])


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


@pytest.mark.serial
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
    # Non-zero atol required, as exposed by seed 828791701
    atol = 1e-5
    # Test normal dot
    a = np.random.uniform(-3, 3, (3, 4))
    b = np.random.uniform(-3, 3, (4, 5))
    c = np.dot(a, b)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B)
    assert_almost_equal(c, C.asnumpy(), atol=atol)
    # Test dot with transpose kargs
    a = np.random.uniform(-3, 3, (3, 4))
    b = np.random.uniform(-3, 3, (3, 5))
    c = np.dot(a.T, b)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B, transpose_a=True)
    assert_almost_equal(c, C.asnumpy(), atol=atol)
    # Test dot with transpose kargs
    a = np.random.uniform(-3, 3, (3, 4))
    b = np.random.uniform(-3, 3, (5, 4))
    c = np.dot(a, b.T)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B, transpose_b=True)
    assert_almost_equal(c, C.asnumpy(), atol=atol)
    # Test dot with transpose kargs
    a = np.random.uniform(-3, 3, (4, 3))
    b = np.random.uniform(-3, 3, (5, 4))
    c = np.dot(a.T, b.T)
    A = mx.nd.array(a)
    B = mx.nd.array(b)
    C = mx.nd.dot(A, B, transpose_a=True, transpose_b=True)
    assert_almost_equal(c, C.asnumpy(), atol=atol)


@pytest.mark.serial
def test_reduce():
    sample_num = 300
    def test_reduce_inner(numpy_reduce_func, nd_reduce_func, multi_axes,
                          allow_almost_equal=False, check_dtype=True):
        dtypes = [(np.float16, 1),
                  (np.float32, 4),
                  (np.double, 6)]
        for _ in range(sample_num):
            dtype, decimal = random.choice(dtypes)
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 11, size=ndim)
            dat = (np.random.rand(*shape) - 0.5).astype(dtype)
            keepdims = np.random.randint(0, 2)

            allow_nan = np.random.randint(0, 2)
            if allow_nan:
                total_nans = np.random.randint(0, dat.size//10+1)
                dat.ravel()[np.random.choice(
                    dat.size, total_nans, replace=False)] = np.nan

            allow_inf = np.random.randint(0, 2)
            if allow_inf:
                r = np.random.randint(0, 3)
                total_infs = np.random.randint(0, dat.size//20+1)
                if r == 0:
                    total_pos_infs, total_neg_infs = total_infs, 0
                elif r == 1:
                    total_pos_infs, total_neg_infs = 0, total_infs
                else:
                    total_pos_infs = total_neg_infs = total_infs // 2
                dat.ravel()[np.random.choice(
                    dat.size, total_pos_infs, replace=False)] = np.inf
                dat.ravel()[np.random.choice(
                    dat.size, total_neg_infs, replace=False)] = -np.inf

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

            mx_arr = mx.nd.array(dat, dtype=dtype)
            ndarray_ret = nd_reduce_func(mx_arr, axis=axes, keepdims=keepdims)
            if type(ndarray_ret) is mx.ndarray.NDArray:
                ndarray_ret = ndarray_ret.asnumpy()
            assert (ndarray_ret.shape == numpy_ret.shape) or \
                   (ndarray_ret.shape == (1,) and numpy_ret.shape == ()), \
                   f"nd:{ndarray_ret.shape}, numpy:{numpy_ret.shape}"
            if check_dtype:
                assert ndarray_ret.dtype == numpy_ret.dtype,\
                        (ndarray_ret.dtype, numpy_ret.dtype)
            if allow_almost_equal:
                assert_array_almost_equal(ndarray_ret, numpy_ret, decimal=decimal)
            else:
                assert_array_equal(ndarray_ret, numpy_ret)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.sum),
                      mx.nd.sum, True, allow_almost_equal=True)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.max),
                      mx.nd.max, True)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.min),
                      mx.nd.min, True)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.argmax),
                      mx.nd.argmax, False, check_dtype=False)
    test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.argmin),
                      mx.nd.argmin, False, check_dtype=False)


@pytest.mark.serial
def test_broadcast():
    sample_num = 1000
    def test_broadcast_to():
        for _ in range(sample_num):
            ndim = np.random.randint(1, 6)
            target_shape = np.random.randint(1, 11, size=ndim)
            shape = target_shape.copy()
            axis_flags = np.random.randint(0, 2, size=ndim)
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

    def test_broadcast_like():
        for _ in range(sample_num):
            ndim = np.random.randint(1, 6)
            target_shape = np.random.randint(1, 11, size=ndim)
            target = mx.nd.ones(shape=tuple(target_shape))
            shape = target_shape.copy()
            axis_flags = np.random.randint(0, 2, size=ndim)
            for (axis, flag) in enumerate(axis_flags):
                if flag:
                    shape[axis] = 1
            dat = np.random.rand(*shape) - 0.5
            numpy_ret = dat
            ndarray_ret = mx.nd.array(dat).broadcast_like(target)
            if type(ndarray_ret) is mx.ndarray.NDArray:
                ndarray_ret = ndarray_ret.asnumpy()
            assert (ndarray_ret.shape == target_shape).all()
            err = np.square(ndarray_ret - numpy_ret).mean()
            assert err < 1E-8

    def test_broadcast_like_axis():
        testcases = [
            # Lhs shape, rhs shape, lhs axis, rhs axis, result
            [(1, 2, 1, 3), (5, 6, 7, 8), (0,2), (1,3), (6, 2, 8, 3)],
            [(1,), (5,), (0,), (-1,), (5,)],
            [(1, 7, 9, 1, 1), (9,), (-2,), (0,), (1, 7, 9, 9, 1)],
            [(1, 7, 9, 1, 1), (9, 1), (-2, -1), (-2, -1), (1, 7, 9, 9, 1)],
            [(2, 1), (1, 7, 9, 1, 1), (1,), (-3,), (2, 9)]
        ]

        for test_data in testcases:
            lhs = mx.nd.random.uniform(shape=test_data[0])
            rhs = mx.nd.random.uniform(shape=test_data[1])
            output = mx.nd.broadcast_like(lhs, rhs, lhs_axes=test_data[2], rhs_axes=test_data[3])

            assert_exception(mx.nd.broadcast_like, mx.base.MXNetError, lhs, rhs, lhs_axes=(), rhs_axes=())
            assert output.shape == test_data[4]

    test_broadcast_to()
    test_broadcast_like()
    test_broadcast_like_axis()


@pytest.mark.serial
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
    # The following ops are sensitive to the precision of the calculation.
    # Force numpy to match mxnet's float32.
    check_broadcast_binary(lambda x, y: x.astype(np.float32) > y.astype(np.float32))
    check_broadcast_binary(lambda x, y: x.astype(np.float32) < y.astype(np.float32))
    check_broadcast_binary(lambda x, y: x.astype(np.float32) >= y.astype(np.float32))
    check_broadcast_binary(lambda x, y: x.astype(np.float32) <= y.astype(np.float32))
    check_broadcast_binary(lambda x, y: x.astype(np.float32) == y.astype(np.float32))


def test_moveaxis():
    X = mx.nd.array([[[1, 2, 3], [4, 5, 6]],
                     [[7, 8, 9], [10, 11, 12]]])
    res = mx.nd.moveaxis(X, 0, 2).asnumpy()
    true_res = mx.nd.array([[[  1.,   7.],
                             [  2.,   8.],
                             [  3.,   9.]],
                            [[  4.,  10.],
                             [  5.,  11.],
                             [  6.,  12.]]])
    assert same(res, true_res.asnumpy())
    assert mx.nd.moveaxis(X, 2, 0).shape == (3, 2, 2)

    def test_move_to_end():
        x = mx.nd.random.normal(0, 1, (5, 6, 7))
        for source, expected in [(0, (6, 7, 5)),
                                 (1, (5, 7, 6)),
                                 (2, (5, 6, 7)),
                                 (-1, (5, 6, 7))]:
            actual = mx.nd.moveaxis(x, source, -1).shape
            assert actual == expected

    def test_move_new_position():
        x = mx.nd.random.normal(0, 1, (1, 2, 3, 4))
        for source, destination, expected in [
            (0, 1, (2, 1, 3, 4)),
            (1, 2, (1, 3, 2, 4)),
            (1, -1, (1, 3, 4, 2)),
        ]:
            actual = mx.nd.moveaxis(x, source, destination).shape
            assert actual == expected

    def test_preserve_order():
        x = mx.nd.zeros((1, 2, 3, 4))
        for source, destination in [
            (0, 0),
            (3, -1),
            (-1, 3),
            ([0, -1], [0, -1]),
            ([2, 0], [2, 0]),
            (range(4), range(4)),
        ]:
            actual = mx.nd.moveaxis(x, source, destination).shape
            assert actual == (1, 2, 3, 4)

    def test_move_multiples():
        x = mx.nd.zeros((4, 1, 2, 3))
        for source, destination, expected in [
            ([0, 1], [2, 3], (2, 3, 4, 1)),
            ([2, 3], [0, 1], (2, 3, 4, 1)),
            ([0, 1, 2], [2, 3, 0], (2, 3, 4, 1)),
            ([3, 0], [1, 0], (4, 3, 1, 2)),
            ([0, 3], [0, 1], (4, 3, 1, 2)),
        ]:
            actual = mx.nd.moveaxis(x, source, destination).shape
            assert actual == expected

    def test_errors():
        x = mx.nd.random.normal(0, 1, (1, 2, 3))
        assert_exception(mx.nd.moveaxis, ValueError, x, 3, 0)
        assert_exception(mx.nd.moveaxis, ValueError, x, -4, 0)
        assert_exception(mx.nd.moveaxis, ValueError, x, 0, 5)
        assert_exception(mx.nd.moveaxis, ValueError, x, [0, 0], [0, 1])
        assert_exception(mx.nd.moveaxis, ValueError, x, [0, 1], [1, 1])
        assert_exception(mx.nd.moveaxis, ValueError, x, 0, [0, 1])
        assert_exception(mx.nd.moveaxis, ValueError, x, [0, 1], [0])

    test_move_to_end()
    test_move_new_position()
    test_preserve_order()
    test_move_multiples()
    test_errors()


def test_arange():
    for _ in range(5):
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


def test_linspace():
    for _ in range(5):
        start = np.random.rand() * 100
        stop = np.random.rand() * 100
        num = np.random.randint(20)
        gt = np.linspace(start, stop, num)
        pred = mx.nd.linspace(start, stop, num).asnumpy()
        assert_almost_equal(pred, gt)
        gt = np.linspace(start, stop, num, endpoint=False)
        pred = mx.nd.linspace(start, stop, num, endpoint=False).asnumpy()
        assert_almost_equal(pred, gt)
        gt = np.linspace(start, stop, num, dtype="int32")
        pred = mx.nd.linspace(start, stop, num, dtype="int32").asnumpy()
        assert_almost_equal(pred, gt)


@pytest.mark.serial
def test_order():
    ctx = default_device()
    dat_size = 5
    is_large_tensor_enabled = runtime.Features().is_enabled('INT64_TENSOR_SIZE')
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
            assert dat.shape == (dat_size, dat_size, dat_size, dat_size)
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
                for i in range(dat_size):
                    for j in range(dat_size):
                        for k in range(dat_size):
                            ret[i, gt_argsort[i, :, j, k], j, k] = 1
        return ret

    # Produce input data for the tests, including ensuring unique values if desired.
    # Numpy's argsort does not consistently return lowest-index-first for matching
    # values, making it hard to generate a numpy 'golden copy' to compare against
    # the mxnet operator.  The 'mask' function is particularly hard to test given that
    # equal values might span the 'k' boundary.  Issue exposed with seed 1405838964.
    def get_values(ensure_unique, dtype):
        if dtype == np.int16 or dtype == np.int32 or dtype == np.int64:
            return np.arange(dat_size ** 4, dtype=dtype).reshape((dat_size, dat_size, dat_size, dat_size))
        elif dtype == np.float32 or dtype == np.float64:
            while True:
                data = np.random.normal(size=(dat_size, dat_size, dat_size, dat_size)).astype(dtype)
                if not ensure_unique:
                    return data
                num_unique_values = len(set(data.flatten()))
                if data.size == num_unique_values:
                    return data
        else:
            raise NotImplementedError

    # Produce a large matrix (256, 300096) as the input data, to cover the case which
    # has a large size of matrix (exceed the express range by float precisly), but
    # the number of elements in each dimension could be expressed by float precisly.
    def get_large_matrix():
        data = np.array([np.arange(300096).astype(np.float32)])
        data = np.repeat(data, 100, axis=0)
        np.apply_along_axis(np.random.shuffle, 1, data)
        return data

    large_matrix_npy = get_large_matrix()
    large_matrix_nd = mx.nd.array(large_matrix_npy, ctx=ctx, dtype=large_matrix_npy.dtype)

    nd_ret_topk = mx.nd.topk(large_matrix_nd, axis=1, ret_typ="indices", k=5, is_ascend=False).asnumpy()
    gt = gt_topk(large_matrix_npy, axis=1, ret_typ="indices", k=5, is_ascend=False)
    assert_almost_equal(nd_ret_topk, gt)

    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        a_npy = get_values(ensure_unique=True, dtype=dtype)
        a_nd = mx.nd.array(a_npy, ctx=ctx, dtype=dtype)

        # test for ret_typ=indices
        nd_ret_topk = mx.nd.topk(a_nd, axis=1, ret_typ="indices", k=3, is_ascend=True).asnumpy()
        # Test the default dtype
        assert nd_ret_topk.dtype == np.float32
        gt = gt_topk(a_npy, axis=1, ret_typ="indices", k=3, is_ascend=True)
        assert_almost_equal(nd_ret_topk, gt)
        nd_ret_topk = mx.nd.topk(a_nd, axis=3, ret_typ="indices", k=2, is_ascend=False, dtype=np.float64).asnumpy()
        assert nd_ret_topk.dtype == np.float64
        gt = gt_topk(a_npy, axis=3, ret_typ="indices", k=2, is_ascend=False)
        assert_almost_equal(nd_ret_topk, gt)
        nd_ret_topk = mx.nd.topk(a_nd, axis=None, ret_typ="indices", k=21, is_ascend=False, dtype=np.int32).asnumpy()
        assert nd_ret_topk.dtype == np.int32
        gt = gt_topk(a_npy, axis=None, ret_typ="indices", k=21, is_ascend=False)
        assert_almost_equal(nd_ret_topk, gt)

        # test for ret_typ=value
        nd_ret_topk = mx.nd.topk(a_nd, axis=1, ret_typ="value", k=3, is_ascend=True).asnumpy()
        assert nd_ret_topk.dtype == dtype
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
        assert nd_ret_topk.dtype == dtype
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
        assert nd_ret_topk_val.dtype == dtype
        assert nd_ret_topk_ind.dtype == np.float32
        gt_val = gt_topk(a_npy, axis=1, ret_typ="value", k=3, is_ascend=True)
        gt_ind = gt_topk(a_npy, axis=1, ret_typ="indices", k=3, is_ascend=True)
        assert_almost_equal(nd_ret_topk_val, gt_val)
        assert_almost_equal(nd_ret_topk_ind, gt_ind)
        # test for kNullOp
        _, nd_ret_topk_ind = mx.nd.topk(a_nd, axis=1, ret_typ="both", k=3, is_ascend=True, dtype=np.float64)
        assert nd_ret_topk_ind.dtype == np.float64
        nd_ret_topk_ind = nd_ret_topk_ind.asnumpy()
        assert_almost_equal(nd_ret_topk_ind, gt_ind)
        # test for kNullOp
        nd_ret_topk_val, _ = mx.nd.topk(a_nd, axis=1, ret_typ="both", k=3, is_ascend=True)
        nd_ret_topk_val = nd_ret_topk_val.asnumpy()
        assert_almost_equal(nd_ret_topk_val, gt_val)

        # test for sort
        nd_ret_sort = mx.nd.sort(a_nd, axis=1, is_ascend=True).asnumpy()
        gt = gt_topk(a_npy, axis=1, ret_typ="value", k=dat_size, is_ascend=True)
        assert_almost_equal(nd_ret_sort, gt)
        nd_ret_sort = mx.nd.sort(a_nd, axis=None, is_ascend=False).asnumpy()
        gt = gt_topk(a_npy, axis=None, ret_typ="value",
                     k=dat_size*dat_size*dat_size*dat_size, is_ascend=False)
        assert_almost_equal(nd_ret_sort, gt)

        # test for argsort
        for idtype in [np.int32, np.float16, np.float32, np.float64]:
            nd_ret_argsort = mx.nd.argsort(a_nd, axis=3, is_ascend=True, dtype=idtype).asnumpy()
            assert nd_ret_argsort.dtype == idtype
            gt = gt_topk(a_npy, axis=3, ret_typ="indices", k=dat_size, is_ascend=True)
            assert_almost_equal(nd_ret_argsort, gt)
            nd_ret_argsort = mx.nd.argsort(a_nd, axis=None, is_ascend=False, dtype=idtype).asnumpy()
            assert nd_ret_argsort.dtype == idtype
            gt = gt_topk(a_npy, axis=None, ret_typ="indices",
                         k=dat_size*dat_size*dat_size*dat_size, is_ascend=False)
            assert_almost_equal(nd_ret_argsort, gt)

        # Repeat those tests that don't involve indices.  These should pass even with
        # duplicated input data values (over many repeated runs with different random seeds,
        # this will be tested).
        a_npy = get_values(ensure_unique=False, dtype=dtype)
        a_nd = mx.nd.array(a_npy, ctx=ctx, dtype=dtype)

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

        # test for sort
        nd_ret_sort = mx.nd.sort(a_nd, axis=1, is_ascend=True).asnumpy()
        gt = gt_topk(a_npy, axis=1, ret_typ="value", k=dat_size, is_ascend=True)
        assert_almost_equal(nd_ret_sort, gt)
        nd_ret_sort = mx.nd.sort(a_nd, axis=None, is_ascend=False).asnumpy()
        gt = gt_topk(a_npy, axis=None, ret_typ="value",
                     k=dat_size*dat_size*dat_size*dat_size, is_ascend=False)
        assert_almost_equal(nd_ret_sort, gt)

    a = mx.nd.arange(0, 1024, step=1, repeat=1, dtype=np.int32)
    assert_almost_equal(a.topk(k=1024, dtype=np.int32).asnumpy(), a.asnumpy()[::-1])
    a.attach_grad()

    k = 10
    with mx.autograd.record():
        b = mx.nd.topk(a, k=k, ret_typ='value')
        b.backward(mx.nd.ones((k,), dtype=np.int32))
    a_grad = a.grad.asnumpy()
    for i in range(-1, - k - 1, -1):
        assert a_grad[i] == 1

    # test topk gradient with a small shape
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        a = mx.nd.arange(0, 1000, step=1, repeat=1, dtype=dtype)
        a.attach_grad()
        k = 10
        ograd = mx.nd.arange(0, k, dtype=dtype)
        with mx.autograd.record():
            b = mx.nd.topk(a, k=k, ret_typ='value')
            b.backward(ograd)
        a_grad = a.grad.asnumpy()
        ograd_npy = ograd.asnumpy()
        for i in range(-1, - k - 1, -1):
            assert a_grad[i] == ograd_npy[-i - 1]

    # Repeat those tests that don't involve indices.  These should pass even with
    # duplicated input data values (over many repeated runs with different random seeds,
    # this will be tested).
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        a_npy = get_values(ensure_unique=False, dtype=dtype)
        a_nd = mx.nd.array(a_npy, ctx=ctx, dtype=dtype)

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

        # test for sort
        nd_ret_sort = mx.nd.sort(a_nd, axis=1, is_ascend=True).asnumpy()
        gt = gt_topk(a_npy, axis=1, ret_typ="value", k=dat_size, is_ascend=True)
        assert_almost_equal(nd_ret_sort, gt)
        nd_ret_sort = mx.nd.sort(a_nd, axis=None, is_ascend=False).asnumpy()
        gt = gt_topk(a_npy, axis=None, ret_typ="value",
                     k=dat_size*dat_size*dat_size*dat_size, is_ascend=False)
        assert_almost_equal(nd_ret_sort, gt)


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


def test_ndarray_take():
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

@pytest.mark.serial
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
    N_array = np.random.randint(1, high=8, size=10)
    M_array = np.random.randint(1, high=8, size=10)
    k_array = np.random.randint(-10, high=10, size=10)
    for i in range(10):
        N = N_array[i]
        M = M_array[i]
        k = k_array[i]
        assert_almost_equal(np.eye(N, M, k), mx.nd.eye(N, M, k).asnumpy())
        assert_almost_equal(np.eye(N, k=k), mx.nd.eye(N, k=k).asnumpy())


@pytest.mark.serial
def test_ndarray_fluent():
    has_grad = set(['flatten', 'expand_dims', 'flip', 'tile', 'transpose', 'sum', 'nansum', 'prod',
                    'nanprod', 'mean', 'max', 'min', 'reshape', 'broadcast_to', 'split', 'split_v2',
                    'broadcast_axes', 'pad', 'swapaxes', 'slice', 'slice_axis', 'slice_like',
                    'take', 'one_hot', 'pick', 'sort', 'topk', 'argsort', 'argmax', 'argmin',
                    'clip', 'abs', 'sign', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                    'degrees', 'radians', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                    'exp', 'expm1', 'log', 'log10', 'log2', 'log1p', 'sqrt', 'rsqrt', 'square',
                    'reshape_like', 'cbrt', 'rcbrt', 'relu', 'sigmoid', 'softmax', 'log_softmax',
                    'softmin', 'reciprocal'])
    def check_fluent_regular(func, kwargs, shape=(5, 17, 1), equal_nan=False):
        with mx.name.NameManager():
            data = mx.nd.random_uniform(shape=shape, ctx=default_device())
            regular = getattr(mx.ndarray, func)(data, **kwargs)
            fluent = getattr(data, func)(**kwargs)
            if isinstance(regular, list):
                for r, f in zip(regular, fluent):
                    assert almost_equal(r.asnumpy(), f.asnumpy(), equal_nan=equal_nan)
            else:
                assert almost_equal(regular.asnumpy(), fluent.asnumpy(), equal_nan=equal_nan)

    for func in ['flatten', 'norm', 'round', 'rint', 'fix', 'floor', 'ceil', 'trunc', 'zeros_like',
                 'ones_like', 'abs', 'sign', 'sin', 'cos', 'degrees', 'radians', 'exp', 'expm1',
                 'square', 'reciprocal', 'argmax_channel', 'shape_array', 'size_array']:
        check_fluent_regular(func, {})

    for func in ['arccosh', 'arcsin', 'arccos', 'arctan', 'tan', 'sinh', 'cosh', 'tanh',
                 'arcsinh', 'arctanh', 'log', 'log10', 'log2', 'log1p', 'sqrt', 'rsqrt',
                 'cbrt', 'rcbrt', 'relu', 'sigmoid', 'softmax', 'log_softmax', 'softmin']:
        check_fluent_regular(func, {}, equal_nan=True)

    for func in ['expand_dims', 'flip', 'sort', 'topk', 'argsort', 'argmax', 'argmin']:
        check_fluent_regular(func, {'axis': 1})

    check_fluent_regular('one_hot', {'depth': 15})
    check_fluent_regular('tile', {'reps': (1,2)})
    check_fluent_regular('repeat', {'repeats': 3})
    check_fluent_regular('transpose', {'axes': (1,0,2)})
    check_fluent_regular('split', {'axis': 2, 'num_outputs': 3}, shape=(5, 17, 6))
    check_fluent_regular('split_v2', {'axis': 2, 'indices_or_sections': 3}, shape=(5, 17, 6))
    check_fluent_regular('split_v2', {'axis': 2, 'indices_or_sections': (1, 3, 5)}, shape=(5, 17, 6))
    check_fluent_regular('slice', {'begin': (2, 5, 1), 'end': (4, 7, 6)}, shape=(5, 17, 6))
    check_fluent_regular('slice_axis', {'axis': 1, 'begin': 5, 'end': 7})
    check_fluent_regular('slice_like', {'axes': (0, -2), 'shape_like': mx.nd.zeros((3, 3))})
    check_fluent_regular('take', {'indices': mx.nd.array([2, 3])})
    check_fluent_regular('pick', {'axis': 1, 'index': mx.nd.array([[2], [3], [5], [6], [11]])})
    check_fluent_regular('clip', {'a_min': 0.25, 'a_max': 0.75})
    check_fluent_regular('broadcast_axes', {'axis': (2,), 'size': (5,)})
    check_fluent_regular('pad', {'mode': 'constant', 'pad_width': (0,0,0,0,3,0,0,4)}, shape=(5, 17, 2, 3))
    check_fluent_regular('reshape_like', {'rhs': mx.nd.ones((30, 17))}, shape=(5, 17, 2, 3))

    for func in ['sum', 'nansum', 'prod', 'nanprod', 'mean', 'max', 'min', 'norm']:
        check_fluent_regular(func, {'axis': (1, 2)})

    check_fluent_regular('reshape', {'shape': (17, 1, 5)})
    check_fluent_regular('broadcast_to', {'shape': (5, 17, 47)})
    check_fluent_regular('squeeze', {'axis': (1, 3)}, shape=(2, 1, 3, 1, 4))


def test_bool_ambiguous():
    with pytest.raises(ValueError):
        bool(mx.nd.ones((2,3,4)))


def test_bool():
    assert not bool(mx.nd.array([]))
    assert not bool(mx.nd.zeros((1,)))
    assert bool(mx.nd.ones((1,)))


@pytest.mark.serial
def test_basic_indexing_is_contiguous():
    x_np = np.arange(np.prod((6, 7, 8, 9))).reshape((6, 7, 8, 9))
    x_mx = mx.nd.array(x_np)

    slices = [
        slice(None),
        slice(2),
        slice(20),
        slice(1, 4),
        slice(None, None, 2),
        slice(None, None, 20),
        slice(0, 1),
        slice(None, None, -1),
        slice(3, None, -2),
    ]

    is_contiguous = mx.nd.NDArray._basic_indexing_slice_is_contiguous

    for idx in combinations_with_replacement(slices, 4):
        for slc in permutations(idx):
            # Check helper function
            contig_pred = is_contiguous(slc, x_np.shape)
            contig_true = x_np[slc].flags.contiguous
            assert contig_pred == contig_true, (
                "failed with slc={}, pred ({}) != actual ({})"
                "".format(slc, contig_pred, contig_true)
            )

            if contig_pred:
                # Check mutation behavior
                y_mx = x_mx.copy()
                y_mx_slc = y_mx[slc]
                y_mx_slc[:] = 0
                assert (y_mx[slc].asnumpy() == 0).all()


@pytest.mark.serial
def test_ndarray_indexing():
    def test_getitem(np_array, index, is_scalar=False):
        """`is_scalar` indicates whether we should expect a scalar for the result.
        If so, the indexed array of NDArray should call asscalar to compare
        with numpy's indexed array."""
        np_index = index
        if isinstance(index, mx.nd.NDArray):
            np_index = index.asnumpy()
        if isinstance(index, tuple):
            np_index = tuple(
                idx.asnumpy() if isinstance(idx, mx.nd.NDArray) else idx
                for idx in index
            )

        np_indexed_array = np_array[np_index]
        mx_array = mx.nd.array(np_array, dtype=np_array.dtype)
        try:
            mx_indexed_array = mx_array[index]
        except Exception as e:
            print('Failed with index = {}'.format(index))
            raise e
        if is_scalar:
            mx_indexed_array = mx_indexed_array.asscalar()
        else:
            mx_indexed_array = mx_indexed_array.asnumpy()

        assert same(np_indexed_array, mx_indexed_array), 'Failed with index = {}'.format(index)

    def test_setitem(np_array, index, is_scalar):
        def assert_same(np_array, np_index, mx_array, mx_index, mx_value, np_value=None):
            if np_value is not None:
                np_array[np_index] = np_value
            elif isinstance(mx_value, mx.nd.NDArray):
                np_array[np_index] = mx_value.asnumpy()
            else:
                np_array[np_index] = mx_value

            try:
                mx_array[mx_index] = mx_value
            except Exception as e:
                print('Failed with index = {}, value.shape = {}'.format(mx_index, mx_value.shape))
                raise e
            assert same(np_array, mx_array.asnumpy())

        def _is_basic_index(index):
            if isinstance(index, (integer_types, py_slice)):
                return True
            if isinstance(index, tuple) and all(isinstance(i, (integer_types, py_slice)) for i in index):
                return True
            return False

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
                np_value = np.random.randint(low=-10000, high=0, size=(indexed_array_shape[-1],))
                # test NDArray with broadcast
                assert_same(np_array, np_index, mx_array, index, mx.nd.array(np_value))
                # test numpy array with broadcast
                assert_same(np_array, np_index, mx_array, index, np_value)

                # test value shape are expanded to be longer than index array's shape
                # this is currently only supported in basic indexing
                if _is_basic_index(index):
                    expanded_value_shape = (1, 1, 1) + np_value.shape
                    assert_same(np_array, np_index, mx_array, index, np.array(np_value.reshape(expanded_value_shape)))
                    assert_same(np_array, np_index, mx_array, index, np_value.reshape(expanded_value_shape))

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

    def test_setitem_autograd(np_array, index):
        x = mx.nd.array(np_array, dtype=np_array.dtype)
        out_shape = x[index].shape
        y = mx.nd.random.uniform(shape=out_shape)
        y.attach_grad()
        try:
            with mx.autograd.record():
                x[index] = y
                # `a[None] = v` is equivalent to `a[...] = v` which doesn't raise
                if index is not None:
                    assert False, 'failed with index = {}'.format(index)  # should not reach here
        except mx.base.MXNetError as err:
            assert str(err).find('Inplace operations (+=, -=, x[:]=, etc) are not supported when recording with') != -1

    def np_int(index, int_type=np.int32):
        def convert(num):
            if num is None:
                return num
            else:
                return int_type(num)

        if isinstance(index, slice):
            return slice(convert(index.start), convert(index.stop), convert(index.step))
        elif isinstance(index, tuple):  # tuple of slices and integers
            ret = []
            for elem in index:
                if isinstance(elem, slice):
                    ret.append(slice(convert(elem.start), convert(elem.stop), convert(elem.step)))
                else:
                    ret.append(convert(elem))
            return tuple(ret)
        else:
            assert False

    shape = (8, 16, 9, 9)
    np_array = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    # index_list is a list of tuples. The tuple's first element is the index, the second one is a boolean value
    # indicating whether we should expect the result as a scalar compared to numpy.
    index_list = [# Basic indexing
                  # Single int as index
                  (0, False), (np.int32(0), False), (np.int64(0), False),
                  (5, False), (np.int32(5), False), (np.int64(5), False),
                  (-1, False), (np.int32(-1), False), (np.int64(-1), False),
                  # Slicing as index
                  (slice(5), False), (np_int(slice(5), np.int32), False), (np_int(slice(5), np.int64), False),
                  (slice(1, 5), False), (np_int(slice(1, 5), np.int32), False), (np_int(slice(1, 5), np.int64), False),
                  (slice(1, 5, 2), False), (np_int(slice(1, 5, 2), np.int32), False),
                  (np_int(slice(1, 5, 2), np.int64), False),
                  (slice(7, 0, -1), False), (np_int(slice(7, 0, -1)), False),
                  (np_int(slice(7, 0, -1), np.int64), False),
                  (slice(None, 6), False), (np_int(slice(None, 6)), False),
                  (np_int(slice(None, 6), np.int64), False),
                  (slice(None, 6, 3), False), (np_int(slice(None, 6, 3)), False),
                  (np_int(slice(None, 6, 3), np.int64), False),
                  (slice(1, None), False), (np_int(slice(1, None)), False),
                  (np_int(slice(1, None), np.int64), False),
                  (slice(1, None, 3), False), (np_int(slice(1, None, 3)), False),
                  (np_int(slice(1, None, 3), np.int64), False),
                  (slice(None, None, 2), False), (np_int(slice(None, None, 2)), False),
                  (np_int(slice(None, None, 2), np.int64), False),
                  (slice(None, None, -1), False),
                  (np_int(slice(None, None, -1)), False), (np_int(slice(None, None, -1), np.int64), False),
                  (slice(None, None, -2), False),
                  (np_int(slice(None, None, -2), np.int32), False), (np_int(slice(None, None, -2), np.int64), False),
                  # slice(None) as indices
                  ((slice(None), slice(None), 1, 8), False),
                  ((slice(None), slice(None), -1, 8), False),
                  ((slice(None), slice(None), 1, -8), False),
                  ((slice(None), slice(None), -1, -8), False),
                  (np_int((slice(None), slice(None), 1, 8)), False),
                  (np_int((slice(None), slice(None), 1, 8), np.int64), False),
                  ((slice(None), slice(None), 1, 8), False),
                  (np_int((slice(None), slice(None), -1, -8)), False),
                  (np_int((slice(None), slice(None), -1, -8), np.int64), False),
                  ((slice(None), 2, slice(1, 5), 1), False),
                  (np_int((slice(None), 2, slice(1, 5), 1)), False),
                  (np_int((slice(None), 2, slice(1, 5), 1), np.int64), False),
                  # Multiple ints as indices
                  ((1, 2, 3), False),
                  (np_int((1, 2, 3)), False),
                  (np_int((1, 2, 3), np.int64), False),
                  ((-1, -2, -3), False),
                  (np_int((-1, -2, -3)), False),
                  (np_int((-1, -2, -3), np.int64), False),
                  ((1, 2, 3, 4), True),
                  (np_int((1, 2, 3, 4)), True),
                  (np_int((1, 2, 3, 4), np.int64), True),
                  ((-4, -3, -2, -1), True),
                  (np_int((-4, -3, -2, -1)), True),
                  (np_int((-4, -3, -2, -1), np.int64), True),
                  ((slice(None, None, -1), 2, slice(1, 5), 1), False),
                  (np_int((slice(None, None, -1), 2, slice(1, 5), 1)), False),
                  (np_int((slice(None, None, -1), 2, slice(1, 5), 1), np.int64), False),
                  ((slice(None, None, -1), 2, slice(1, 7, 2), 1), False),
                  (np_int((slice(None, None, -1), 2, slice(1, 7, 2), 1)), False),
                  (np_int((slice(None, None, -1), 2, slice(1, 7, 2), 1), np.int64), False),
                  ((slice(1, 8, 2), slice(14, 2, -2), slice(3, 8), slice(0, 7, 3)), False),
                  (np_int((slice(1, 8, 2), slice(14, 2, -2), slice(3, 8), slice(0, 7, 3))), False),
                  (np_int((slice(1, 8, 2), slice(14, 2, -2), slice(3, 8), slice(0, 7, 3)), np.int64), False),
                  ((slice(1, 8, 2), 1, slice(3, 8), 2), False),
                  (np_int((slice(1, 8, 2), 1, slice(3, 8), 2)), False),
                  (np_int((slice(1, 8, 2), 1, slice(3, 8), 2), np.int64), False),
                  # Test Ellipsis ('...')
                  ((1, Ellipsis, -1), False),
                  ((slice(2), Ellipsis, None, 0), False),
                  # Test basic indexing with newaxis
                  (None, False),
                  ((1, None, -2, 3, -4), False),
                  ((1, slice(2, 5), None), False),
                  ((slice(None), slice(1, 4), None, slice(2, 3)), False),
                  ((slice(1, 3), slice(1, 3), slice(1, 3), slice(1, 3), None), False),
                  ((slice(1, 3), slice(1, 3), None, slice(1, 3), slice(1, 3)), False),
                  ((None, slice(1, 2), 3, None), False),
                  ((1, None, 2, 3, None, None, 4), False),
                  # Advanced indexing
                  ([1], False), ([1, 2], False), ([2, 1, 3], False), ([7, 5, 0, 3, 6, 2, 1], False),
                  (np.array([6, 3], dtype=np.int32), False),
                  (np.array([[3, 4], [0, 6]], dtype=np.int32), False),
                  (np.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int32), False),
                  (np.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int64), False),
                  (np.array([[2], [0], [1]], dtype=np.int32), False),
                  (np.array([[2], [0], [1]], dtype=np.int64), False),
                  (mx.nd.array([4, 7], dtype=np.int32), False),
                  (mx.nd.array([4, 7], dtype=np.int64), False),
                  (mx.nd.array([[3, 6], [2, 1]], dtype=np.int32), False),
                  (mx.nd.array([[3, 6], [2, 1]], dtype=np.int64), False),
                  (mx.nd.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int32), False),
                  (mx.nd.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int64), False),
                  ((1, [2, 3]), False), ((1, [2, 3], np.array([[3], [0]], dtype=np.int32)), False),
                  ((1, [2, 3]), False), ((1, [2, 3], np.array([[3], [0]], dtype=np.int64)), False),
                  ((1, [2], np.array([[5], [3]], dtype=np.int32), slice(None)), False),
                  ((1, [2], np.array([[5], [3]], dtype=np.int64), slice(None)), False),
                  ((1, [2, 3], np.array([[6], [0]], dtype=np.int32), slice(2, 5)), False),
                  ((1, [2, 3], np.array([[6], [0]], dtype=np.int64), slice(2, 5)), False),
                  ((1, [2, 3], np.array([[4], [7]], dtype=np.int32), slice(2, 5, 2)), False),
                  ((1, [2, 3], np.array([[4], [7]], dtype=np.int64), slice(2, 5, 2)), False),
                  ((1, [2], np.array([[3]], dtype=np.int32), slice(None, None, -1)), False),
                  ((1, [2], np.array([[3]], dtype=np.int64), slice(None, None, -1)), False),
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
                  (([1, 2], slice(3, 5), (2, 3), [3, 4]), False),
                  # Advanced indexing with None
                  (([1, 2], slice(3, 5), None, None, [3, 4]), False),
                  ((slice(None), slice(3, 5), None, None, [2, 3], [3, 4]), False),
                  ((slice(None), slice(3, 5), None, [2, 3], None, [3, 4]), False),
                  ((None, slice(None), slice(3, 5), [2, 3], None, [3, 4]), False),
                  ((None, slice(None), None, slice(3, 5), [2, 3], None, [3, 4]), False),
                  (([2, 3, 4], None, [3, 4, 6], None, slice(1, 2), None, [1, 2, 3]), False),
    ]

    for index in index_list:
        test_getitem(np_array, index[0], index[1])
        test_setitem(np_array, index[0], index[1])
        test_getitem_autograd(np_array, index[0])
        test_setitem_autograd(np_array, index[0])


def test_assign_float_value_to_ndarray():
    """Test case from https://github.com/apache/mxnet/issues/8668"""
    a = np.array([47.844944], dtype=np.float32)
    b = mx.nd.zeros(1, dtype=np.float32)
    b[0] = a
    assert same(a, b.asnumpy())
    b[0] = a[0]
    assert same(a, b.asnumpy())

def test_assign_large_int_to_ndarray():
    """Test case from https://github.com/apache/mxnet/issues/11639"""
    a = mx.nd.zeros((4, 1), dtype=np.int32)
    a[1,0] = int(16800001)
    a[2,0] = int(16800002)
    b = a.asnumpy()
    assert same(b[1,0], 16800001)
    a = a-1
    b = a.asnumpy()
    assert same(b[1,0], 16800000)

def test_assign_a_row_to_ndarray():
    """Test case from https://github.com/apache/mxnet/issues/9976"""
    H, W = 10, 10
    dtype = np.float32
    a_np = np.random.random((H, W)).astype(dtype)
    a_nd = mx.nd.array(a_np)

    # assign directly
    a_np[0] = a_np[1]
    a_nd[0] = a_nd[1]
    assert same(a_np, a_nd.asnumpy())

    # assign a list
    v = np.random.random(W).astype(dtype).tolist()
    a_np[1] = v
    a_nd[1] = v
    assert same(a_np, a_nd.asnumpy())

    # assign a np.ndarray
    v = np.random.random(W).astype(dtype)
    a_np[2] = v
    a_nd[2] = v
    assert same(a_np, a_nd.asnumpy())

    # assign by slice
    a_np[0, :] = a_np[1]
    a_nd[0, :] = a_nd[1]
    assert same(a_np, a_nd.asnumpy())

def test_ndarray_astype():
    x = mx.nd.zeros((2, 3), dtype='int32')
    y = x.astype('float32')
    assert (y.dtype==np.float32)
    # Test that a new ndarray has been allocated
    assert (id(x) != id(y))

    x = mx.nd.zeros((2, 3), dtype='int32')
    y = x.astype('float32', copy=False)
    assert (y.dtype==np.float32)
    # Test that a new ndarray has been allocated
    assert (id(x) != id(y))

    x = mx.nd.zeros((2, 3), dtype='int32')
    y = x.astype('int32')
    assert (y.dtype==np.int32)
    # Test that a new ndarray has been allocated
    # even though they have same dtype
    assert (id(x) != id(y))

    # Test that a new ndarray has not been allocated
    x = mx.nd.zeros((2, 3), dtype='int32')
    y = x.astype('int32', copy=False)
    assert (id(x) == id(y))

    # Test the string version 'int32'
    # has the same behaviour as the np.int32
    x = mx.nd.zeros((2, 3), dtype='int32')
    y = x.astype(np.int32, copy=False)
    assert (id(x) == id(y))


@pytest.mark.serial
def test_norm(ctx=default_device()):
    try:
        import scipy
        assert LooseVersion(scipy.__version__) >= LooseVersion('0.1')
        from scipy.linalg import norm as sp_norm
    except (AssertionError, ImportError):
        print("Could not import scipy.linalg.norm or scipy is too old. "
              "Falling back to numpy.linalg.norm which is not numerically stable.")
        from numpy.linalg import norm as sp_norm

    def l1norm(input_data, axis=0, keepdims=False):
        return np.sum(abs(input_data), axis=axis, keepdims=keepdims)
    def l2norm(input_data, axis=0, keepdims=False):
        return sp_norm(input_data, axis=axis, keepdims=keepdims)

    in_data_dim = random_sample([4,5,6], 1)[0]
    for force_reduce_dim1 in [True, False]:
        in_data_shape = rand_shape_nd(in_data_dim)
        if force_reduce_dim1:
            in_data_shape = in_data_shape[:3] + (1, ) + in_data_shape[4:]
        np_arr = np.random.uniform(-1, 1, in_data_shape).astype(np.float32)
        mx_arr = mx.nd.array(np_arr, ctx=ctx)
        for ord in [1, 2]:
            for keep_dims in [True, False]:
                for i in range(4):
                    npy_out = l1norm(np_arr, i, keep_dims) if ord == 1 else l2norm(
                        np_arr, i, keep_dims)
                    mx_out = mx.nd.norm(mx_arr, ord=ord, axis=i, keepdims=keep_dims)
                    assert npy_out.shape == mx_out.shape
                    assert_almost_equal(npy_out, mx_out)
                    if (i < 3):
                        npy_out = l1norm(np_arr, (i, i + 1), keep_dims) if ord == 1 else l2norm(
                            np_arr, (i, i + 1), keep_dims)
                        mx_out = mx.nd.norm(mx_arr, ord=ord, axis=(i, i + 1), keepdims=keep_dims)
                        assert npy_out.shape == mx_out.shape
                        assert_almost_equal(npy_out, mx_out)


def test_ndarray_cpu_shared_ctx():
    ctx = mx.Context('cpu_shared', 0)
    res = mx.nd.zeros((1, 2, 3), ctx=ctx)
    assert(res.context == ctx)

@pytest.mark.serial
def test_dlpack():
    for _ in [np.float32, np.int32]:
        for shape in [(3, 4, 5, 6), (2, 10), (15,)]:
            a = mx.nd.random.uniform(shape = shape)
            a_np = a.copy()

            pack = a.to_dlpack_for_read()
            b = mx.nd.from_dlpack(pack)

            a_copy = a.copy()
            pack2 = a_copy.to_dlpack_for_write()
            c = mx.nd.from_dlpack(pack2)

            pack3 = mx.nd.to_dlpack_for_read(a)
            d = mx.nd.from_dlpack(pack3)

            a_copy = a.copy()
            pack4 = mx.nd.to_dlpack_for_write(a_copy)
            e = mx.nd.from_dlpack(pack4)

            del a, pack, pack2, pack3, pack4

            assert_almost_equal(a_np, b)
            assert_almost_equal(a_np, c)
            assert_almost_equal(a_np, d)
            assert_almost_equal(a_np, e)

def test_ndarray_is_inf():
    random_dimensions = np.random.randint(2, 5)
    random_shape = [np.random.randint(2, 5) for i in range(random_dimensions)]
    data = mxnet.test_utils.rand_ndarray(random_shape,'default')
    data[0][0] = np.inf
    data[0][1] = -np.inf
    data[1][0] = np.nan
    data[1][1] = 5
    output = mx.nd.contrib.isinf(data)
    expected_output = np.isinf(data.asnumpy())
    np.testing.assert_equal(output.asnumpy(), expected_output.astype(int))
    # astype since numpy functions default return type is boolean array instead of int

def test_ndarray_is_finite():
    random_dimensions = np.random.randint(2, 5)
    random_shape = [np.random.randint(2, 5) for i in range(random_dimensions)]
    data = mxnet.test_utils.rand_ndarray(random_shape,'default')
    data[0][0] = np.inf
    data[0][1] = -np.inf
    data[1][0] = np.nan
    data[1][1] = 5
    output = mx.nd.contrib.isfinite(data)
    expected_output = np.isfinite(data.asnumpy())
    np.testing.assert_equal(output.asnumpy(), expected_output.astype(int))
    # astype since numpy functions default return type is boolean array instead of int

def test_ndarray_is_nan():
    random_dimensions = np.random.randint(2, 5)
    random_shape = [np.random.randint(2, 5) for i in range(random_dimensions)]
    data = mxnet.test_utils.rand_ndarray(random_shape,'default')
    data[0][0] = np.inf
    data[0][1] = -np.inf
    data[1][0] = np.nan
    data[1][1] = 5
    output = mx.nd.contrib.isnan(data)
    expected_output = np.isnan(data.asnumpy())
    np.testing.assert_equal(output.asnumpy(), expected_output.astype(int))
    # astype since numpy functions default return type is boolean array instead of int

def test_ndarray_nan_comparison():
    random_dimensions = np.random.randint(2, 5)
    random_shape = [np.random.randint(2, 5) for i in range(random_dimensions)]
    data1 = mxnet.test_utils.rand_ndarray(random_shape,'default')
    data2 = mxnet.test_utils.rand_ndarray(random_shape,'default')
    data1[1][0] = np.NaN
    data2[0][0] = np.NaN

    nd_max = mx.nd.maximum(data1, data2)
    np_max = np.maximum(data1.asnumpy(), data2.asnumpy())
    np.testing.assert_equal(nd_max.asnumpy(), np_max)

    nd_min = mx.nd.minimum(data1, data2)
    np_min = np.minimum(data1.asnumpy(), data2.asnumpy())
    np.testing.assert_equal(nd_min.asnumpy(), np_min)

    nd_relu = mx.nd.relu(data1)
    np_relu = np.maximum(data1.asnumpy(), 0)
    np.testing.assert_equal(nd_relu.asnumpy(), np_relu)

    data1.attach_grad()
    with mx.autograd.record():
        y = mx.nd.relu(data1)
    y.backward()
    data1_grad = data1.grad.asnumpy()
    for i in (np.isnan(data1_grad))[1][0].flatten():
        assert i == True


def test_zero_from_numpy():
    # Test zero_copy
    arrays = [
        # ordinary numpy array
        np.array([[1, 2], [3, 4], [5, 6]], dtype="float32"),
        # 0-dim
        np.array((1, )).reshape(()),
        # 0-size
        np.array(()).reshape((1, 0, 2)),
    ]
    for zero_copy in [False, True]:
        for np_array in arrays:
            mx_array = mx.nd.from_numpy(np_array, zero_copy=zero_copy)
            mx.test_utils.assert_almost_equal(np_array, mx_array.asnumpy())
    np_array = arrays[0]
    mx_array = mx.nd.from_numpy(np_array)
    assertRaises(ValueError, np_array.__setitem__, (2, 1), 0)

    mx_array[2, 1] = 100
    mx.test_utils.assert_almost_equal(np_array, mx_array.asnumpy())
    np_array = np.array([[1, 2], [3, 4], [5, 6]]).transpose()
    assert not np_array.flags["C_CONTIGUOUS"]
    try:
        mx_array = mx.nd.from_numpy(np_array)
    except ValueError:
        pass
    else:
        assert False


def test_save_load_scalar_zero_size_ndarrays():
    def check_save_load(save_is_np_shape, load_is_np_shape, shapes, save_throw_exception, load_throw_exception):
        with mx.np_shape(save_is_np_shape):
            array_list = [np.random.randint(0, 10, size=shape) for shape in shapes]
            array_list = [mx.nd.array(arr) for arr in array_list]
            with TemporaryDirectory() as work_dir:
                fname = os.path.join(work_dir, 'dataset')
                if save_throw_exception:
                    assert_exception(mx.nd.save, mx.MXNetError, fname, array_list)
                else:
                    mx.nd.save(fname, array_list)
                with mx.np_shape(load_is_np_shape):
                    if load_throw_exception:
                        assert_exception(mx.nd.load, mx.MXNetError, fname)
                    else:
                        array_list_loaded = mx.nd.load(fname)
                        assert len(array_list) == len(array_list_loaded)
                        for a1, a2 in zip(array_list, array_list_loaded):
                            assert np.array_equal(a1.asnumpy(), a2.asnumpy())

    check_save_load(False, False, [(2, 0, 1), (0,), (0, 4), (3, 0, 0, 0), (2, 1), (0, 5, 0)], False, False)
    check_save_load(True, False, [(2, 0, 1), (0,), (0, 4), (3, 0, 0, 0), (2, 1), (0, 5, 0)], False, True)
    check_save_load(False, True, [(2, 0, 1), (0,), (0, 4), (3, 0, 0, 0), (2, 1), (0, 5, 0)], False, True)
    check_save_load(True, True, [(2, 0, 1), (0,), (), (), (0, 4), (), (3, 0, 0, 0), (2, 1), (0, 5, 0)], False, False)


def _test_update_ops_mutation_impl():
    assert_allclose = functools.partial(
                np.testing.assert_allclose, rtol=1e-10)

    def assert_mutate(x, y):
            np.testing.assert_raises(
                AssertionError, assert_allclose, x, y)

    def assert_unchanged(x, y):
            assert_allclose(x, y)

    def test_op(op, num_inputs, mutated_inputs, **kwargs):
        for dim in range(1, 7):
            shape = rand_shape_nd(dim)
            shapes = (shape,) * num_inputs

            # Generate Arrays
            arrays = tuple(map(mx.nd.array, random_arrays(*shapes)))

            # Arrays before update
            pre_arrays = tuple(map(
                lambda x: x.asnumpy(), arrays))

            # Operate
            # weight -> arrays[0]
            op(*arrays, out=arrays[0], **kwargs)

            # Arrays post update
            post_arrays = tuple(map(
                lambda x: x.asnumpy(), arrays))

            for idx, (pre_array, post_array) in \
                    enumerate(zip(pre_arrays, post_arrays)):
                if idx in mutated_inputs:
                    assert_mutate(pre_array, post_array)
                else:
                    assert_unchanged(pre_array, post_array)

    test_op(mx.nd.signsgd_update, 2, [0], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3,
             'clip_gradient': 1e-3})
    test_op(mx.nd.signum_update, 3, [0, 2], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3,
             'momentum': 1e-3, 'clip_gradient': 1e-3,
             'wd_lh': 1e-3})
    test_op(mx.nd.sgd_update, 2, [0], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3,
             'clip_gradient': 1e-3})
    test_op(mx.nd.sgd_mom_update, 3, [0, 2], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3,
             'momentum': 0.01, 'clip_gradient': 1e-3})
    test_op(mx.nd.nag_mom_update, 3, [0, 2], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3,
             'momentum': 0.01, 'clip_gradient': 1e-3})
    test_op(mx.nd.ftml_update, 5, [0, 2, 3, 4], **
            {'t': 3, 'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3,
             'clip_grad': 1e-3})
    test_op(mx.nd.ftrl_update, 4, [0, 2, 3], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3})
    test_op(mx.nd.adam_update, 4, [0, 2, 3], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3})
    test_op(mx.nd.rmspropalex_update, 5, [0, 2, 3, 4], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3})
    test_op(mx.nd.rmsprop_update, 3, [0, 2], **
            {'rescale_grad': 0.1, 'lr': 0.01, 'wd': 1e-3})


@pytest.mark.serial
def test_update_ops_mutation():
    _test_update_ops_mutation_impl()


# Problem :
# https://github.com/apache/mxnet/pull/15768#issuecomment-532046408
@pytest.mark.seed(412298777)
@pytest.mark.serial
def test_update_ops_mutation_failed_seed():
    # The difference was -5.9604645e-08 which was
    # lower than then `rtol` of 1e-07
    _test_update_ops_mutation_impl()


def test_large_int_rounding():
    large_integer = 50000001

    a = mx.nd.array([large_integer], dtype='int32')
    assert np.all(a == large_integer)

    a = mx.nd.array([large_integer], dtype='int32').floor()
    assert np.all(a == large_integer)

    a = mx.nd.array([large_integer], dtype='int32').round()
    assert np.all(a == large_integer)

    a = mx.nd.array([large_integer], dtype='int32').ceil()
    assert np.all(a == large_integer)

    a = mx.nd.array([large_integer], dtype='int32').trunc()
    assert np.all(a == large_integer)


def test_load_saved_gpu_array_when_no_gpus_are_present():
    # State obtained with mx.nd.arange(1, ctx=mx.gpu()).__getstate__()
    # State needs to be exported manually, as running above command will only
    # work if a gpu is present.
    ndarray_state = {
        'handle':
        bytearray(
            b'\xc9\xfa\x93\xf9\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        )
    }
    array = mx.nd.arange(1)
    # Test that MXNDArrayLoadFromRawBytes works even if we have built with Cuda
    # but there are no GPUs
    array.__setstate__(ndarray_state)

def test_readable_bfloat16_print():
    arr_bfloat16 = mx.nd.linspace(0, 1, 16).reshape((2, 2, 2, 2)).astype(bfloat16)
    arr_uint16 = arr_bfloat16.asnumpy()
    arr_float = arr_bfloat16.astype(float)
    assert (arr_bfloat16.__str__() == arr_float.__str__())
    assert (arr_bfloat16.__repr__().find(arr_uint16.__str__()) != -1)
