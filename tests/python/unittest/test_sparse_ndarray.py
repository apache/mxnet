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

import pickle as pkl

from mxnet.ndarray import NDArray
from mxnet.test_utils import *
from common import setup_module, with_seed, random_seed, teardown
from mxnet.base import mx_real_t
from numpy.testing import assert_allclose
import numpy.random as rnd
import numpy as np
from common import assertRaises
from mxnet.ndarray.sparse import RowSparseNDArray, CSRNDArray


def sparse_nd_ones(shape, stype):
    return mx.nd.ones(shape).tostype(stype)


@with_seed()
def test_sparse_nd_elemwise_add():
    def check_sparse_nd_elemwise_binary(shapes, stypes, f, g):
        # generate inputs
        nds = []
        for i, stype in enumerate(stypes):
            if stype == 'row_sparse':
                nd, _ = rand_sparse_ndarray(shapes[i], stype)
            elif stype == 'default':
                nd = mx.nd.array(random_arrays(shapes[i]), dtype = np.float32)
            else:
                assert(False)
            nds.append(nd)
        # check result
        test = f(nds[0], nds[1])
        assert_almost_equal(test.asnumpy(), g(nds[0].asnumpy(), nds[1].asnumpy()))

    num_repeats = 3
    g = lambda x,y: x + y
    op = mx.nd.elemwise_add
    for i in range(num_repeats):
        shape = [rand_shape_2d()] * 2
        check_sparse_nd_elemwise_binary(shape, ['default'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)


@with_seed()
def test_sparse_nd_copy():
    def check_sparse_nd_copy(from_stype, to_stype, shape):
        from_nd = rand_ndarray(shape, from_stype)
        # copy to ctx
        to_ctx = from_nd.copyto(default_context())
        # copy to stype
        to_nd = rand_ndarray(shape, to_stype)
        to_nd = from_nd.copyto(to_nd)
        assert np.sum(np.abs(from_nd.asnumpy() != to_ctx.asnumpy())) == 0.0
        assert np.sum(np.abs(from_nd.asnumpy() != to_nd.asnumpy())) == 0.0

    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    stypes = ['row_sparse', 'csr']
    for stype in stypes:
        check_sparse_nd_copy(stype, 'default', shape)
        check_sparse_nd_copy('default', stype, shape)
    check_sparse_nd_copy('row_sparse', 'row_sparse', shape_3d)
    check_sparse_nd_copy('row_sparse', 'default', shape_3d)
    check_sparse_nd_copy('default', 'row_sparse', shape_3d)

@with_seed()
def test_sparse_nd_basic():
    def check_sparse_nd_basic_rsp():
        storage_type = 'row_sparse'
        shape = rand_shape_2d()
        nd, (v, idx) = rand_sparse_ndarray(shape, storage_type)
        assert(nd._num_aux == 1)
        assert(nd.indices.dtype == np.int64)
        assert(nd.stype == 'row_sparse')

    check_sparse_nd_basic_rsp()


@with_seed()
def test_sparse_nd_setitem():
    def check_sparse_nd_setitem(stype, shape, dst):
        x = mx.nd.zeros(shape=shape, stype=stype)
        x[:] = dst
        dst_nd = mx.nd.array(dst) if isinstance(dst, (np.ndarray, np.generic)) else dst
        assert np.all(x.asnumpy() == dst_nd.asnumpy() if isinstance(dst_nd, NDArray) else dst)

    shape = rand_shape_2d()
    for stype in ['row_sparse', 'csr']:
        # ndarray assignment
        check_sparse_nd_setitem(stype, shape, rand_ndarray(shape, 'default'))
        check_sparse_nd_setitem(stype, shape, rand_ndarray(shape, stype))
        # numpy assignment
        check_sparse_nd_setitem(stype, shape, np.ones(shape))
    # scalar assigned to row_sparse NDArray
    check_sparse_nd_setitem('row_sparse', shape, 2)

@with_seed()
def test_sparse_nd_slice():
    shape = (rnd.randint(2, 10), rnd.randint(2, 10))
    stype = 'csr'
    A, _ = rand_sparse_ndarray(shape, stype)
    A2 = A.asnumpy()
    start = rnd.randint(0, shape[0] - 1)
    end = rnd.randint(start + 1, shape[0])
    assert same(A[start:end].asnumpy(), A2[start:end])
    assert same(A[start - shape[0]:end].asnumpy(), A2[start:end])
    assert same(A[start:].asnumpy(), A2[start:])
    assert same(A[:end].asnumpy(), A2[:end])
    ind = rnd.randint(-shape[0], shape[0] - 1)
    assert same(A[ind].asnumpy(), A2[ind][np.newaxis, :])

    start_col = rnd.randint(0, shape[1] - 1)
    end_col = rnd.randint(start_col + 1, shape[1])
    result = mx.nd.slice(A, begin=(start, start_col), end=(end, end_col))
    result_dense = mx.nd.slice(mx.nd.array(A2), begin=(start, start_col), end=(end, end_col))
    assert same(result_dense.asnumpy(), result.asnumpy())

    A = mx.nd.sparse.zeros('csr', shape)
    A2 = A.asnumpy()
    assert same(A[start:end].asnumpy(), A2[start:end])
    result = mx.nd.slice(A, begin=(start, start_col), end=(end, end_col))
    result_dense = mx.nd.slice(mx.nd.array(A2), begin=(start, start_col), end=(end, end_col))
    assert same(result_dense.asnumpy(), result.asnumpy())

    def check_slice_nd_csr_fallback(shape):
        stype = 'csr'
        A, _ = rand_sparse_ndarray(shape, stype)
        A2 = A.asnumpy()
        start = rnd.randint(0, shape[0] - 1)
        end = rnd.randint(start + 1, shape[0])

        # non-trivial step should fallback to dense slice op
        result = mx.nd.sparse.slice(A, begin=(start,), end=(end + 1,), step=(2,))
        result_dense = mx.nd.slice(mx.nd.array(A2), begin=(start,), end=(end + 1,), step=(2,))
        assert same(result_dense.asnumpy(), result.asnumpy())

    shape = (rnd.randint(2, 10), rnd.randint(1, 10))
    check_slice_nd_csr_fallback(shape)


@with_seed()
def test_sparse_nd_concat():
    def check_concat(arrays):
        ret = np.concatenate([arr.asnumpy() for arr in arrays], axis=0)
        same(mx.nd.concat(*arrays, dim=0).asnumpy(), ret)
    nds = []
    zero_nds = []
    ncols = rnd.randint(2, 10)
    for i in range(3):
        shape = (rnd.randint(2, 10), ncols)
        A, _ = rand_sparse_ndarray(shape, 'csr')
        nds.append(A)
        zero_nds.append(mx.nd.zeros(shape).tostype('csr'))
    check_concat(nds)
    check_concat(zero_nds)


@with_seed()
def test_sparse_nd_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.nd.zeros(shape=shape, stype=stype)
        y = sparse_nd_ones(shape, stype)
        z = x == y
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = 0 == y
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == 'default'
        z = 1 == y
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == stype


@with_seed()
def test_sparse_nd_not_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.nd.zeros(shape=shape, stype=stype)
        y = sparse_nd_ones(shape, stype)
        z = x != y
        assert (z.asnumpy() == np.ones(shape)).all()
        z = 0 != y
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == stype
        z = 1 != y
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == 'default'


@with_seed()
def test_sparse_nd_greater():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.nd.zeros(shape=shape, stype=stype)
        y = sparse_nd_ones(shape, stype)
        z = x > y
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = y > 0
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == stype
        z = 0 > y
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == stype
        z = y > 1
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == stype


@with_seed()
def test_sparse_nd_greater_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.nd.zeros(shape=shape, stype=stype)
        y = sparse_nd_ones(shape, stype)
        z = x >= y
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = y >= 0
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == 'default'
        z = 0 >= y
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == 'default'
        z = y >= 1
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == stype


@with_seed()
def test_sparse_nd_lesser():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.nd.zeros(shape=shape, stype=stype)
        y = sparse_nd_ones(shape, stype)
        z = y < x
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = 0 < y
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == stype
        z = y < 0
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == stype
        z = y < 1
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == 'default'


@with_seed()
def test_sparse_nd_lesser_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.nd.zeros(shape=shape, stype=stype)
        y = sparse_nd_ones(shape, stype)
        z = y <= x
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = 0 <= y
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == 'default'
        z = y <= 0
        assert (z.asnumpy() == np.zeros(shape)).all()
        assert z.stype == 'default'
        z = 1 <= y
        assert (z.asnumpy() == np.ones(shape)).all()
        assert z.stype == stype


@with_seed()
def test_sparse_nd_binary():
    N = 3
    def check_binary(fn, stype):
        for _ in range(N):
            ndim = 2
            oshape = np.random.randint(1, 6, size=(ndim,))
            bdim = 2
            lshape = list(oshape)
            # one for broadcast op, another for elemwise op
            rshape = list(oshape[ndim-bdim:])
            for i in range(bdim):
                sep = np.random.uniform(0, 1)
                if sep < 0.33:
                    lshape[ndim-i-1] = 1
                elif sep < 0.66:
                    rshape[bdim-i-1] = 1
            lhs = np.random.uniform(0, 1, size=lshape)
            rhs = np.random.uniform(0, 1, size=rshape)
            lhs_nd = mx.nd.array(lhs).tostype(stype)
            rhs_nd = mx.nd.array(rhs).tostype(stype)
            assert_allclose(fn(lhs, rhs), fn(lhs_nd, rhs_nd).asnumpy(), rtol=1e-4, atol=1e-4)
            assert_allclose(fn(lhs, lhs), fn(lhs_nd, lhs_nd).asnumpy(), rtol=1e-4, atol=1e-4)

    stypes = ['row_sparse', 'csr']
    for stype in stypes:
        check_binary(lambda x, y: x + y, stype)
        check_binary(lambda x, y: x - y, stype)
        check_binary(lambda x, y: x * y, stype)
        check_binary(lambda x, y: x / y, stype)
        check_binary(lambda x, y: x ** y, stype)
        check_binary(lambda x, y: x > y, stype)
        check_binary(lambda x, y: x < y, stype)
        check_binary(lambda x, y: x >= y, stype)
        check_binary(lambda x, y: x <= y, stype)
        check_binary(lambda x, y: x == y, stype)


@with_seed()
def test_sparse_nd_binary_scalar_op():
    N = 3
    def check(fn, stype, out_stype=None):
        for _ in range(N):
            ndim = 2
            shape = np.random.randint(1, 6, size=(ndim,))
            npy = np.random.normal(0, 1, size=shape)
            nd = mx.nd.array(npy).tostype(stype)
            if out_stype is not None:
                assert(nd.stype == out_stype)
            assert_allclose(fn(npy), fn(nd).asnumpy(), rtol=1e-4, atol=1e-4)

    stypes = ['row_sparse', 'csr']
    for stype in stypes:
        check(lambda x: 1 + x, stype)
        check(lambda x: 1 - x, stype)
        check(lambda x: 1 * x, stype)
        check(lambda x: 1 / x, stype)
        check(lambda x: 2 ** x, stype)
        check(lambda x: 1 > x, stype)
        check(lambda x: 0.5 > x, stype)
        check(lambda x: 0.5 < x, stype)
        check(lambda x: 0.5 >= x, stype)
        check(lambda x: 0.5 <= x, stype)
        check(lambda x: 0.5 == x, stype)
        check(lambda x: x / 2, stype, out_stype=stype)
        check(lambda x: x + 0, stype, out_stype=stype)
        check(lambda x: x - 0, stype, out_stype=stype)


@with_seed()
def test_sparse_nd_binary_iop():
    N = 3
    def check_binary(fn, stype):
        for _ in range(N):
            ndim = 2
            oshape = np.random.randint(1, 6, size=(ndim,))
            lshape = list(oshape)
            rshape = list(oshape)
            lhs = np.random.uniform(0, 1, size=lshape)
            rhs = np.random.uniform(0, 1, size=rshape)
            lhs_nd = mx.nd.array(lhs).tostype(stype)
            rhs_nd = mx.nd.array(rhs).tostype(stype)
            assert_allclose(fn(lhs, rhs),
                            fn(lhs_nd, rhs_nd).asnumpy(),
                            rtol=1e-4, atol=1e-4)

    def inplace_add(x, y):
        x += y
        return x
    def inplace_mul(x, y):
        x *= y
        return x
    stypes = ['csr', 'row_sparse']
    fns = [inplace_add, inplace_mul]
    for stype in stypes:
        for fn in fns:
            check_binary(fn, stype)


@with_seed()
def test_sparse_nd_negate():
    def check_sparse_nd_negate(shape, stype):
        npy = np.random.uniform(-10, 10, rand_shape_2d())
        arr = mx.nd.array(npy).tostype(stype)
        assert_almost_equal(npy, arr.asnumpy())
        assert_almost_equal(-npy, (-arr).asnumpy())

        # a final check to make sure the negation (-) is not implemented
        # as inplace operation, so the contents of arr does not change after
        # we compute (-arr)
        assert_almost_equal(npy, arr.asnumpy())

    shape = rand_shape_2d()
    stypes = ['csr', 'row_sparse']
    for stype in stypes:
        check_sparse_nd_negate(shape, stype)


@with_seed()
def test_sparse_nd_broadcast():
    sample_num = 1000
    # TODO(haibin) test with more than 2 dimensions
    def test_broadcast_to(stype):
        for _ in range(sample_num):
            ndim = 2
            target_shape = np.random.randint(1, 11, size=ndim)
            shape = target_shape.copy()
            axis_flags = np.random.randint(0, 2, size=ndim)
            for (axis, flag) in enumerate(axis_flags):
                if flag:
                    shape[axis] = 1
            dat = np.random.rand(*shape) - 0.5
            numpy_ret = dat
            ndarray = mx.nd.array(dat).tostype(stype)
            ndarray_ret = ndarray.broadcast_to(shape=target_shape)
            if type(ndarray_ret) is mx.ndarray.NDArray:
                ndarray_ret = ndarray_ret.asnumpy()
            assert (ndarray_ret.shape == target_shape).all()
            err = np.square(ndarray_ret - numpy_ret).mean()
            assert err < 1E-8

    def test_broadcast_like(stype):
        for _ in range(sample_num):
            ndim = 2
            target_shape = np.random.randint(1, 11, size=ndim)
            target = mx.nd.ones(shape=tuple(target_shape))
            shape = target_shape.copy()
            axis_flags = np.random.randint(0, 2, size=ndim)
            for (axis, flag) in enumerate(axis_flags):
                if flag:
                    shape[axis] = 1
            dat = np.random.rand(*shape) - 0.5
            numpy_ret = dat
            ndarray = mx.nd.array(dat).tostype(stype)
            ndarray_ret = ndarray.broadcast_like(target)
            if type(ndarray_ret) is mx.ndarray.NDArray:
                ndarray_ret = ndarray_ret.asnumpy()
            assert (ndarray_ret.shape == target_shape).all()
            err = np.square(ndarray_ret - numpy_ret).mean()
            assert err < 1E-8

    stypes = ['csr', 'row_sparse']
    for stype in stypes:
        test_broadcast_to(stype)
        test_broadcast_like(stype)


@with_seed()
def test_sparse_nd_transpose():
    npy = np.random.uniform(-10, 10, rand_shape_2d())
    stypes = ['csr', 'row_sparse']
    for stype in stypes:
        nd = mx.nd.array(npy).tostype(stype)
        assert_almost_equal(npy.T, (nd.T).asnumpy())


@with_seed()
def test_sparse_nd_storage_fallback():
    def check_output_fallback(shape):
        ones = mx.nd.ones(shape)
        out = mx.nd.zeros(shape=shape, stype='csr')
        mx.nd.broadcast_add(ones, ones * 2, out=out)
        assert(np.sum(out.asnumpy() - 3) == 0)

    def check_input_fallback(shape):
        ones = mx.nd.ones(shape)
        out = mx.nd.broadcast_add(ones.tostype('csr'), ones.tostype('row_sparse'))
        assert(np.sum(out.asnumpy() - 2) == 0)

    def check_fallback_with_temp_resource(shape):
        ones = mx.nd.ones(shape)
        out = mx.nd.sum(ones)
        assert(out.asscalar() == np.prod(shape))

    shape = rand_shape_2d()
    check_output_fallback(shape)
    check_input_fallback(shape)
    check_fallback_with_temp_resource(shape)


@with_seed()
def test_sparse_nd_random():
    """ test sparse random operator on cpu """
    # gpu random operator doesn't use fixed seed
    if default_context().device_type is 'gpu':
        return
    shape = (100, 100)
    fns = [mx.nd.random.uniform, mx.nd.random.normal, mx.nd.random.gamma]
    for fn in fns:
        rsp_out = mx.nd.zeros(shape=shape, stype='row_sparse')
        dns_out = mx.nd.zeros(shape=shape, stype='default')
        with random_seed(0):
            fn(shape=shape, out=dns_out)
        with random_seed(0):
            fn(shape=shape, out=rsp_out)
        assert_almost_equal(dns_out.asnumpy(), rsp_out.asnumpy())


@with_seed()
def test_sparse_nd_astype():
    stypes = ['row_sparse', 'csr']
    for stype in stypes:
        x = mx.nd.zeros(shape=rand_shape_2d(), stype=stype, dtype='float32')
        y = x.astype('int32')
        assert(y.dtype == np.int32), y.dtype


@with_seed()
def test_sparse_nd_astype_copy():
    stypes = ['row_sparse', 'csr']
    for stype in stypes:
        x = mx.nd.zeros(shape=rand_shape_2d(), stype=stype, dtype='int32')
        y = x.astype('float32')
        assert (y.dtype == np.float32)
        # Test that a new ndarray has been allocated
        assert (id(x) != id(y))

        y = x.astype('float32', copy=False)
        assert (y.dtype == np.float32)
        # Test that a new ndarray has been allocated
        assert (id(x) != id(y))

        y = x.astype('int32')
        assert (y.dtype == np.int32)
        # Test that a new ndarray has been allocated
        # even though they have same dtype
        assert (id(x) != id(y))

        # Test that a new ndarray has not been allocated
        y = x.astype('int32', copy=False)
        assert (id(x) == id(y))

        # Test the string version 'int32'
        # has the same behaviour as the np.int32
        y = x.astype(np.int32, copy=False)
        assert (id(x) == id(y))


@with_seed()
def test_sparse_nd_pickle():
    dim0 = 40
    dim1 = 40
    stypes = ['row_sparse', 'csr']
    densities = [0, 0.5]
    stype_dict = {'row_sparse': RowSparseNDArray, 'csr': CSRNDArray}
    shape = rand_shape_2d(dim0, dim1)
    for stype in stypes:
        for density in densities:
            a, _ = rand_sparse_ndarray(shape, stype, density)
            assert isinstance(a, stype_dict[stype])
            data = pkl.dumps(a)
            b = pkl.loads(data)
            assert isinstance(b, stype_dict[stype])
            assert same(a.asnumpy(), b.asnumpy())


# @kalyc: Getting rid of fixed seed as flakiness could not be reproduced
# tracked at https://github.com/apache/incubator-mxnet/issues/11741
@with_seed()
def test_sparse_nd_save_load():
    repeat = 1
    stypes = ['default', 'row_sparse', 'csr']
    stype_dict = {'default': NDArray, 'row_sparse': RowSparseNDArray, 'csr': CSRNDArray}
    num_data = 20
    densities = [0, 0.5]
    fname = 'tmp_list.bin'
    for _ in range(repeat):
        data_list1 = []
        for i in range(num_data):
            stype = stypes[np.random.randint(0, len(stypes))]
            shape = rand_shape_2d(dim0=40, dim1=40)
            density = densities[np.random.randint(0, len(densities))]
            data_list1.append(rand_ndarray(shape, stype, density))
            assert isinstance(data_list1[-1], stype_dict[stype])
        mx.nd.save(fname, data_list1)

        data_list2 = mx.nd.load(fname)
        assert len(data_list1) == len(data_list2)
        for x, y in zip(data_list1, data_list2):
            assert same(x.asnumpy(), y.asnumpy())

        data_map1 = {'ndarray xx %s' % i: x for i, x in enumerate(data_list1)}
        mx.nd.save(fname, data_map1)
        data_map2 = mx.nd.load(fname)
        assert len(data_map1) == len(data_map2)
        for k, x in data_map1.items():
            y = data_map2[k]
            assert same(x.asnumpy(), y.asnumpy())
    os.remove(fname)


@with_seed()
def test_sparse_nd_unsupported():
    nd = mx.nd.zeros((2,2), stype='row_sparse')
    fn_slice = lambda x: x._slice(None, None)
    fn_at = lambda x: x._at(None)
    fn_reshape = lambda x: x.reshape(None)
    fns = [fn_slice, fn_at, fn_reshape]
    for fn in fns:
        try:
            fn(nd)
            assert(False)
        except:
            pass


@with_seed()
def test_create_csr():
    def check_create_csr_from_nd(shape, density, dtype):
        matrix = rand_ndarray(shape, 'csr', density)
        # create data array with provided dtype and ctx
        data = mx.nd.array(matrix.data.asnumpy(), dtype=dtype)
        indptr = matrix.indptr
        indices = matrix.indices
        csr_created = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
        assert csr_created.stype == 'csr'
        assert same(csr_created.data.asnumpy(), data.asnumpy())
        assert same(csr_created.indptr.asnumpy(), indptr.asnumpy())
        assert same(csr_created.indices.asnumpy(), indices.asnumpy())
        # verify csr matrix dtype and ctx is consistent from the ones provided
        assert csr_created.dtype == dtype, (csr_created, dtype)
        assert csr_created.data.dtype == dtype, (csr_created.data.dtype, dtype)
        assert csr_created.context == Context.default_ctx, (csr_created.context, Context.default_ctx)
        csr_copy = mx.nd.array(csr_created)
        assert(same(csr_copy.asnumpy(), csr_created.asnumpy()))

    def check_create_csr_from_coo(shape, density, dtype):
        matrix = rand_ndarray(shape, 'csr', density)
        sp_csr = matrix.asscipy()
        sp_coo = sp_csr.tocoo()
        csr_created = mx.nd.sparse.csr_matrix((sp_coo.data, (sp_coo.row, sp_coo.col)), shape=shape, dtype=dtype)
        assert csr_created.stype == 'csr'
        assert same(csr_created.data.asnumpy(), sp_csr.data)
        assert same(csr_created.indptr.asnumpy(), sp_csr.indptr)
        assert same(csr_created.indices.asnumpy(), sp_csr.indices)
        csr_copy = mx.nd.array(csr_created)
        assert(same(csr_copy.asnumpy(), csr_created.asnumpy()))
        # verify csr matrix dtype and ctx is consistent
        assert csr_created.dtype == dtype, (csr_created.dtype, dtype)
        assert csr_created.data.dtype == dtype, (csr_created.data.dtype, dtype)
        assert csr_created.context == Context.default_ctx, (csr_created.context, Context.default_ctx)

    def check_create_csr_from_scipy(shape, density, f):
        def assert_csr_almost_equal(nd, sp):
            assert_almost_equal(nd.data.asnumpy(), sp.data)
            assert_almost_equal(nd.indptr.asnumpy(), sp.indptr)
            assert_almost_equal(nd.indices.asnumpy(), sp.indices)
            sp_csr = nd.asscipy()
            assert_almost_equal(sp_csr.data, sp.data)
            assert_almost_equal(sp_csr.indptr, sp.indptr)
            assert_almost_equal(sp_csr.indices, sp.indices)
            assert(sp.dtype == sp_csr.dtype), (sp.dtype, sp_csr.dtype)

        try:
            import scipy.sparse as spsp
            # random canonical csr
            csr_sp = spsp.rand(shape[0], shape[1], density, format="csr")
            csr_nd = f(csr_sp)
            assert_csr_almost_equal(csr_nd, csr_sp)
            # non-canonical csr which contains duplicates and unsorted indices
            indptr = np.array([0, 2, 3, 7])
            indices = np.array([0, 2, 2, 0, 1, 2, 1])
            data = np.array([1, 2, 3, 4, 5, 6, 1])
            non_canonical_csr = spsp.csr_matrix((data, indices, indptr), shape=(3, 3), dtype=csr_nd.dtype)
            canonical_csr_nd = f(non_canonical_csr, dtype=csr_nd.dtype)
            canonical_csr_sp = non_canonical_csr.copy()
            canonical_csr_sp.sum_duplicates()
            canonical_csr_sp.sort_indices()
            assert_csr_almost_equal(canonical_csr_nd, canonical_csr_sp)
        except ImportError:
            print("Could not import scipy.sparse. Skipping unit tests for scipy csr creation")

    dim0 = 20
    dim1 = 20
    densities = [0, 0.5]
    dtype = np.float64
    for density in densities:
        shape = rand_shape_2d(dim0, dim1)
        check_create_csr_from_nd(shape, density, dtype)
        check_create_csr_from_coo(shape, density, dtype)
        check_create_csr_from_scipy(shape, density, mx.nd.sparse.array)
        check_create_csr_from_scipy(shape, density, mx.nd.array)


@with_seed()
def test_create_row_sparse():
    dim0 = 50
    dim1 = 50
    densities = [0, 0.5, 1]
    for density in densities:
        shape = rand_shape_2d(dim0, dim1)
        matrix = rand_ndarray(shape, 'row_sparse', density)
        data = matrix.data
        indices = matrix.indices
        rsp_created = mx.nd.sparse.row_sparse_array((data, indices), shape=shape)
        assert rsp_created.stype == 'row_sparse'
        assert same(rsp_created.data.asnumpy(), data.asnumpy())
        assert same(rsp_created.indices.asnumpy(), indices.asnumpy())
        rsp_copy = mx.nd.array(rsp_created)
        assert(same(rsp_copy.asnumpy(), rsp_created.asnumpy()))

        # add this test since we added np.int32 and np.int64 to integer_types
        if len(shape) == 2:
            for np_int_type in (np.int32, np.int64):
                shape = list(shape)
                shape = [np_int_type(x) for x in shape]
                arg1 = tuple(shape)
                mx.nd.sparse.row_sparse_array(arg1, tuple(shape))
                shape[0] += 1
                assert_exception(mx.nd.sparse.row_sparse_array, ValueError, arg1, tuple(shape))



@with_seed()
def test_create_sparse_nd_infer_shape():
    def check_create_csr_infer_shape(shape, density, dtype):
        try:
            matrix = rand_ndarray(shape, 'csr', density=density)
            data = matrix.data
            indptr = matrix.indptr
            indices = matrix.indices
            nd = mx.nd.sparse.csr_matrix((data, indices, indptr), dtype=dtype)
            num_rows, num_cols = nd.shape
            assert(num_rows == len(indptr) - 1)
            assert(indices.shape[0] > 0), indices
            assert(np.sum((num_cols <= indices).asnumpy()) == 0)
            assert(nd.dtype == dtype), (nd.dtype, dtype)
        # cannot infer on invalid shape
        except ValueError:
            pass

    def check_create_rsp_infer_shape(shape, density, dtype):
        try:
            array = rand_ndarray(shape, 'row_sparse', density=density)
            data = array.data
            indices = array.indices
            nd = mx.nd.sparse.row_sparse_array((data, indices), dtype=dtype)
            inferred_shape = nd.shape
            assert(inferred_shape[1:] == data.shape[1:])
            assert(indices.ndim > 0)
            assert(nd.dtype == dtype)
            if indices.shape[0] > 0:
                assert(np.sum((inferred_shape[0] <= indices).asnumpy()) == 0)
        # cannot infer on invalid shape
        except ValueError:
            pass

    dtype = np.int32
    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    densities = [0, 0.5, 1]
    for density in densities:
        check_create_csr_infer_shape(shape, density, dtype)
        check_create_rsp_infer_shape(shape, density, dtype)
        check_create_rsp_infer_shape(shape_3d, density, dtype)


@with_seed()
def test_create_sparse_nd_from_dense():
    def check_create_from_dns(shape, f, dense_arr, dtype, default_dtype, ctx):
        arr = f(dense_arr, dtype=dtype, ctx=ctx)
        assert(same(arr.asnumpy(), np.ones(shape)))
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        # verify the default dtype inferred from dense arr
        arr2 = f(dense_arr)
        assert(arr2.dtype == default_dtype)
        assert(arr2.context == Context.default_ctx)
    shape = rand_shape_2d()
    dtype = np.int32
    src_dtype = np.float64
    ctx = mx.cpu(1)
    dense_arrs = [mx.nd.ones(shape, dtype=src_dtype), np.ones(shape, dtype=src_dtype), \
                  np.ones(shape, dtype=src_dtype).tolist()]
    for f in [mx.nd.sparse.csr_matrix, mx.nd.sparse.row_sparse_array]:
        for dense_arr in dense_arrs:
            default_dtype = dense_arr.dtype if isinstance(dense_arr, (NDArray, np.ndarray)) \
                            else np.float32
            check_create_from_dns(shape, f, dense_arr, dtype, default_dtype, ctx)

@with_seed()
def test_create_sparse_nd_from_sparse():
    def check_create_from_sp(shape, f, sp_arr, dtype, src_dtype, ctx):
        arr = f(sp_arr, dtype=dtype, ctx=ctx)
        assert(same(arr.asnumpy(), np.ones(shape)))
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        # verify the default dtype inferred from dense arr
        arr2 = f(sp_arr)
        assert(arr2.dtype == src_dtype)
        assert(arr2.context == Context.default_ctx)

    shape = rand_shape_2d()
    src_dtype = np.float64
    dtype = np.int32
    ctx = mx.cpu(1)
    ones = mx.nd.ones(shape, dtype=src_dtype)
    csr_arrs = [ones.tostype('csr')]
    rsp_arrs = [ones.tostype('row_sparse')]
    try:
        import scipy.sparse as spsp
        csr_sp = spsp.csr_matrix(np.ones(shape, dtype=src_dtype))
        csr_arrs.append(csr_sp)
    except ImportError:
        print("Could not import scipy.sparse. Skipping unit tests for scipy csr creation")
    f_csr = mx.nd.sparse.csr_matrix
    f_rsp = mx.nd.sparse.row_sparse_array
    for sp_arr in csr_arrs:
        check_create_from_sp(shape, f_csr, sp_arr, dtype, src_dtype, ctx)
    for sp_arr in rsp_arrs:
        check_create_from_sp(shape, f_rsp, sp_arr, dtype, src_dtype, ctx)


@with_seed()
def test_create_sparse_nd_empty():
    def check_empty(shape, stype):
        arr = mx.nd.empty(shape, stype=stype)
        assert(arr.stype == stype)
        assert same(arr.asnumpy(), np.zeros(shape))

    def check_csr_empty(shape, dtype, ctx):
        arr = mx.nd.sparse.csr_matrix(shape, dtype=dtype, ctx=ctx)
        assert(arr.stype == 'csr')
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        assert same(arr.asnumpy(), np.zeros(shape))
        # check the default value for dtype and ctx
        arr = mx.nd.sparse.csr_matrix(shape)
        assert(arr.dtype == np.float32)
        assert(arr.context == Context.default_ctx)

    def check_rsp_empty(shape, dtype, ctx):
        arr = mx.nd.sparse.row_sparse_array(shape, dtype=dtype, ctx=ctx)
        assert(arr.stype == 'row_sparse')
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        assert same(arr.asnumpy(), np.zeros(shape))
        # check the default value for dtype and ctx
        arr = mx.nd.sparse.row_sparse_array(shape)
        assert(arr.dtype == np.float32)
        assert(arr.context == Context.default_ctx)

    stypes = ['csr', 'row_sparse']
    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    dtype = np.int32
    ctx = mx.cpu(1)
    for stype in stypes:
        check_empty(shape, stype)
    check_csr_empty(shape, dtype, ctx)
    check_rsp_empty(shape, dtype, ctx)
    check_rsp_empty(shape_3d, dtype, ctx)


@with_seed()
def test_synthetic_dataset_generator():
    def test_powerlaw_generator(csr_arr, final_row=1):
        """Test power law distribution
        Total Elements: 32000, Number of zeros: 3200
        Every row has 2 * non zero elements of the previous row.
        Also since (2047 < 3200 < 4095) this will be true till 10th row"""
        indices = csr_arr.indices.asnumpy()
        indptr = csr_arr.indptr.asnumpy()
        for row in range(1, final_row + 1):
            nextrow = row + 1
            current_row_nnz = indices[indptr[row] - 1] + 1
            next_row_nnz = indices[indptr[nextrow] - 1] + 1
            assert next_row_nnz == 2 * current_row_nnz

    # Test if density is preserved
    csr_arr_cols, _ = rand_sparse_ndarray(shape=(32, 10000), stype="csr",
                                          density=0.01, distribution="powerlaw")

    csr_arr_small, _ = rand_sparse_ndarray(shape=(5, 5), stype="csr",
                                           density=0.5, distribution="powerlaw")

    csr_arr_big, _ = rand_sparse_ndarray(shape=(32, 1000000), stype="csr",
                                         density=0.4, distribution="powerlaw")

    csr_arr_square, _ = rand_sparse_ndarray(shape=(1600, 1600), stype="csr",
                                            density=0.5, distribution="powerlaw")
    assert len(csr_arr_cols.data) == 3200
    test_powerlaw_generator(csr_arr_cols, final_row=9)
    test_powerlaw_generator(csr_arr_small, final_row=1)
    test_powerlaw_generator(csr_arr_big, final_row=4)
    test_powerlaw_generator(csr_arr_square, final_row=6)


@with_seed()
def test_sparse_nd_fluent():
    def check_fluent_regular(stype, func, kwargs, shape=(5, 17), equal_nan=False):
        with mx.name.NameManager():
            data = mx.nd.random_uniform(shape=shape, ctx=default_context()).tostype(stype)
            regular = getattr(mx.ndarray, func)(data, **kwargs)
            fluent = getattr(data, func)(**kwargs)
            if isinstance(regular, list):
                for r, f in zip(regular, fluent):
                    assert almost_equal(r.asnumpy(), f.asnumpy(), equal_nan=equal_nan)
            else:
                assert almost_equal(regular.asnumpy(), fluent.asnumpy(), equal_nan=equal_nan)

    all_funcs = ['zeros_like', 'square', 'round', 'rint', 'fix', 'floor', 'ceil', 'trunc',
                 'abs', 'sign', 'sin', 'degrees', 'radians', 'expm1']
    for func in all_funcs:
        check_fluent_regular('csr', func, {})
        check_fluent_regular('row_sparse', func, {})

    all_funcs = ['arcsin', 'arctan', 'tan', 'sinh', 'tanh',
                'arcsinh', 'arctanh', 'log1p', 'sqrt', 'relu']
    for func in all_funcs:
        check_fluent_regular('csr', func, {}, equal_nan=True)
        check_fluent_regular('row_sparse', func, {}, equal_nan=True)

    check_fluent_regular('csr', 'slice', {'begin': (2, 5), 'end': (4, 7)}, shape=(5, 17))
    check_fluent_regular('row_sparse', 'clip', {'a_min': -0.25, 'a_max': 0.75})
    check_fluent_regular('csr', 'clip', {'a_min': -0.25, 'a_max': 0.75})

    for func in ['sum', 'mean', 'norm']:
        check_fluent_regular('csr', func, {'axis': 0})


@with_seed()
def test_sparse_nd_exception():
    """ test invalid sparse operator will throw a exception """
    a = mx.nd.ones((2,2))
    assertRaises(mx.base.MXNetError, mx.nd.sparse.retain, a, invalid_arg="garbage_value")
    assertRaises(ValueError, mx.nd.sparse.csr_matrix, a, shape=(3,2))
    assertRaises(ValueError, mx.nd.sparse.csr_matrix, (2,2), shape=(3,2))
    assertRaises(ValueError, mx.nd.sparse.row_sparse_array, (2,2), shape=(3,2))
    assertRaises(ValueError, mx.nd.sparse.zeros, "invalid_stype", (2,2))

@with_seed()
def test_sparse_nd_check_format():
    """ test check_format for sparse ndarray """
    shape = rand_shape_2d()
    stypes = ["csr", "row_sparse"]
    for stype in stypes:
        arr, _ = rand_sparse_ndarray(shape, stype)
        arr.check_format()
        arr = mx.nd.sparse.zeros(stype, shape)
        arr.check_format()
    # CSR format index pointer array should be less than the number of rows
    shape = (3, 4)
    data_list = [7, 8, 9]
    indices_list = [0, 2, 1]
    indptr_list = [0, 5, 2, 3]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # CSR format indices should be in ascending order per row
    indices_list = [2, 1, 1]
    indptr_list = [0, 2, 2, 3]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # CSR format indptr should end with value equal with size of indices
    indices_list = [1, 2, 1]
    indptr_list = [0, 2, 2, 4]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # CSR format indices should not be negative
    indices_list = [0, 2, 1]
    indptr_list = [0, -2, 2, 3]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # Row Sparse format indices should be less than the number of rows
    shape = (3, 2)
    data_list = [[1, 2], [3, 4]]
    indices_list = [1, 4]
    a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # Row Sparse format indices should be in ascending order
    indices_list = [1, 0]
    a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # Row Sparse format indices should not be negative
    indices_list = [1, -2]
    a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)

@with_seed()
def test_sparse_nd_norm():
    def check_sparse_nd_norm(stype, shape, density, **kwargs):
        data, _ = rand_sparse_ndarray(shape, stype, density)
        norm = data.norm(**kwargs)
        expected_norm = data.tostype('default').norm(**kwargs)
        assert_almost_equal(norm.asnumpy(), expected_norm.asnumpy())

    shape = (5, 5)
    stypes = ['row_sparse', 'csr']
    densities = [0, 0.5, 1]
    for stype in stypes:
        for density in densities:
           check_sparse_nd_norm(stype, shape, density, axis=None, keepdims=False, ord=2)

    # test fallback
    check_sparse_nd_norm(stype, shape, density, axis=0, keepdims=False, ord=2)
    check_sparse_nd_norm(stype, shape, density, axis=None, keepdims=True, ord=2)

@with_seed()
def test_sparse_fc():
    def check_sparse_fc(batch_size, dim_in, dim_out, stype):
        data = rand_ndarray((batch_size, dim_in), stype, density=0.5)
        weight = rand_ndarray((dim_out, dim_in), 'row_sparse', density=1)
        bias = rand_ndarray((dim_out, 1), 'row_sparse', density=1)
        out = mx.nd.sparse.FullyConnected(data, weight, num_hidden=dim_out, bias=bias)
        data_dns = data.tostype('default')
        weight_dns = weight.tostype('default')
        out_dns = mx.nd.FullyConnected(data_dns, weight_dns, num_hidden=dim_out, bias=bias)
        assert_almost_equal(out.asnumpy(), out_dns.asnumpy())

    # test FC with row_sparse weight w/ density=1, dense data
    check_sparse_fc(5, 10, 8, 'default')
    # test FC with row_sparse weight w/ density=1, csr data (fallback)
    check_sparse_fc(5, 10, 8, 'csr')

@with_seed()
def test_sparse_take():
    def check_sparse_take(density, mode):
        data_shape = rand_shape_2d()
        idx_shape = (np.random.randint(low=1, high=10),)
        data = rand_ndarray(data_shape, 'csr', density=density).astype('int32')
        idx = mx.nd.array(np.random.randint(low=-5, high=15, size=idx_shape))
        data_np = data.asnumpy()
        idx_np = idx.asnumpy().astype('int32')
        expected_result = np.take(data_np, idx_np, mode=mode, axis=0)
        result = mx.nd.take(data, idx, mode=mode)
        assert_almost_equal(result.asnumpy(), expected_result)
        assert result.indptr[0].asscalar() == 0
    densities = [0, 0.5, 1]
    modes = ['clip', 'wrap']
    for d in densities:
        for m in modes:
            check_sparse_take(d, m)

@with_seed()
def test_sparse_getnnz():
    if default_context().device_type is 'gpu':
        return
    def check_sparse_getnnz(density, axis):
        shape = rand_shape_2d()
        data = rand_ndarray(shape, 'csr', density=density)
        data_sp = data.asscipy()
        result = mx.nd.contrib.getnnz(data, axis=axis)
        expected_result = data_sp.getnnz(axis=axis)
        assert_almost_equal(result.asnumpy(), expected_result)

    densities = [0, 0.5, 1]
    axis = [1, None]
    for d in densities:
        for a in axis:
            check_sparse_getnnz(d, a)

if __name__ == '__main__':
    import nose
    nose.runmodule()
