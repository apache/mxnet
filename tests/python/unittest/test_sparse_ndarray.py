import pickle as pkl

from mxnet.ndarray import NDArray
from mxnet.test_utils import *
from numpy.testing import assert_allclose
import numpy.random as rnd

from mxnet.sparse_ndarray import RowSparseNDArray, CSRNDArray, _ndarray_cls


def assert_fcompex(f, *args, **kwargs):
    prev_val = mx.test_utils.set_env_var("MXNET_EXEC_STORAGE_FALLBACK", "0", "1")
    f(*args, **kwargs)
    mx.test_utils.set_env_var("MXNET_EXEC_STORAGE_FALLBACK", prev_val)


def sparse_nd_ones(shape, stype):
    return mx.nd.cast_storage(mx.nd.ones(shape), storage_type=stype)


def check_sparse_nd_elemwise_binary(shapes, storage_types, f, g):
    # generate inputs
    nds = []
    for i, storage_type in enumerate(storage_types):
        if storage_type == 'row_sparse':
            nd, _ = rand_sparse_ndarray(shapes[i], storage_type)
        elif storage_type == 'default':
            nd = mx.nd.array(random_arrays(shapes[i]), dtype = np.float32)
        else:
            assert(False)
        nds.append(nd)
    # check result
    test = f(nds[0], nds[1])
    assert_almost_equal(test.asnumpy(), g(nds[0].asnumpy(), nds[1].asnumpy()))


def test_sparse_nd_elemwise_add():
    num_repeats = 10
    g = lambda x,y: x + y
    op = mx.nd.elemwise_add
    for i in range(num_repeats):
        shape = [rand_shape_2d()] * 2
        assert_fcompex(check_sparse_nd_elemwise_binary,
                       shape, ['default'] * 2, op, g)
        assert_fcompex(check_sparse_nd_elemwise_binary,
                       shape, ['default', 'row_sparse'], op, g)
        assert_fcompex(check_sparse_nd_elemwise_binary,
                       shape, ['row_sparse', 'row_sparse'], op, g)


# Test a operator which doesn't implement FComputeEx
def test_sparse_nd_elementwise_fallback():
    num_repeats = 10
    g = lambda x,y: x + y
    op = mx.nd.add_n
    for i in range(num_repeats):
        shape = [rand_shape_2d()] * 2
        check_sparse_nd_elemwise_binary(shape, ['default'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)


def test_sparse_nd_zeros():
    def check_sparse_nd_zeros(stype, shape):
        zero = mx.nd.zeros(shape)
        sparse_zero = mx.sparse_nd.zeros('row_sparse', shape)
        assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros('row_sparse', shape)
    check_sparse_nd_zeros('csr', shape)
    check_sparse_nd_zeros('default', shape)


def test_sparse_nd_copy():
    def check_sparse_nd_copy(from_stype, to_stype):
        shape = rand_shape_2d()
        from_nd = rand_ndarray(shape, from_stype)
        # copy to ctx
        to_ctx = from_nd.copyto(default_context())
        # copy to stype
        to_nd = rand_ndarray(shape, to_stype)
        to_nd = from_nd.copyto(to_nd)
        assert np.sum(np.abs(from_nd.asnumpy() != to_ctx.asnumpy())) == 0.0
        assert np.sum(np.abs(from_nd.asnumpy() != to_nd.asnumpy())) == 0.0

    check_sparse_nd_copy('row_sparse', 'row_sparse')
    check_sparse_nd_copy('row_sparse', 'default')
    check_sparse_nd_copy('default', 'row_sparse')
    check_sparse_nd_copy('default', 'csr')


def check_sparse_nd_prop_rsp():
    storage_type = 'row_sparse'
    shape = rand_shape_2d()
    nd, (v, idx) = rand_sparse_ndarray(shape, storage_type)
    assert(nd._num_aux == 1)
    assert(nd.indices.dtype == np.int32)
    assert(nd.storage_type == 'row_sparse')
    assert_almost_equal(nd.indices.asnumpy(), idx)


def test_sparse_nd_basic():
    def check_rsp_creation(values, indices, shape):
        rsp = mx.sparse_nd.row_sparse(values, indices, shape)
        dns = mx.nd.zeros(shape)
        dns[1] = mx.nd.array(values[0])
        dns[3] = mx.nd.array(values[1])
        assert_almost_equal(rsp.asnumpy(), dns.asnumpy())
        indices = mx.nd.array(indices).asnumpy()
        assert_almost_equal(rsp.indices.asnumpy(), indices)

    def check_csr_creation(shape):
        csr, (indptr, indices, values) = rand_sparse_ndarray(shape, 'csr')
        assert_almost_equal(csr.indptr.asnumpy(), indptr)
        assert_almost_equal(csr.indices.asnumpy(), indices)
        assert_almost_equal(csr.values.asnumpy(), values)

    shape = (4,2)
    values = np.random.rand(2,2)
    indices = np.array([1,3])
    check_rsp_creation(values, indices, shape)

    values = mx.nd.array(np.random.rand(2,2))
    indices = mx.nd.array([1,3], dtype='int32')
    check_rsp_creation(values, indices, shape)

    values = [[0.1, 0.2], [0.3, 0.4]]
    indices = [1,3]
    check_rsp_creation(values, indices, shape)

    check_csr_creation(shape)
    check_sparse_nd_prop_rsp()


def test_sparse_nd_setitem():
    def check_sparse_nd_setitem(storage_type, shape, dst):
        x = mx.sparse_nd.zeros(storage_type, shape)
        x[:] = dst
        dst_nd = mx.nd.array(dst) if isinstance(dst, (np.ndarray, np.generic)) else dst
        assert same(x.asnumpy(), dst_nd.asnumpy())

    shape = rand_shape_2d()
    for stype in ['row_sparse', 'csr']:
        # ndarray assignment
        check_sparse_nd_setitem(stype, shape, rand_ndarray(shape, 'default'))
        check_sparse_nd_setitem(stype, shape, rand_ndarray(shape, stype))
        # numpy assignment
        check_sparse_nd_setitem(stype, shape, np.ones(shape))


def test_sparse_nd_slice():
    def check_sparse_nd_csr_slice(shape):
        storage_type = 'csr'
        A, _ = rand_sparse_ndarray(shape, storage_type)
        A2 = A.asnumpy()
        start = rnd.randint(0, shape[0] - 1)
        end = rnd.randint(start + 1, shape[0])
        assert same(A[start:end].asnumpy(), A2[start:end])
        assert same(A[start:].asnumpy(), A2[start:])
        assert same(A[:end].asnumpy(), A2[:end])

    shape = (rnd.randint(2, 10), rnd.randint(1, 10))
    check_sparse_nd_csr_slice(shape)


def test_sparse_nd_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.sparse_nd.zeros(stype, shape)
        y = sparse_nd_ones(shape, stype)
        z = x == y
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = 0 == x
        assert (z.asnumpy() == np.ones(shape)).all()


def test_sparse_nd_not_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.sparse_nd.zeros(stype, shape)
        y = sparse_nd_ones(shape, stype)
        z = x != y
        assert (z.asnumpy() == np.ones(shape)).all()
        z = 0 != x
        assert (z.asnumpy() == np.zeros(shape)).all()


def test_sparse_nd_greater():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.sparse_nd.zeros(stype, shape)
        y = sparse_nd_ones(shape, stype)
        z = x > y
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = y > 0
        assert (z.asnumpy() == np.ones(shape)).all()
        z = 0 > y
        assert (z.asnumpy() == np.zeros(shape)).all()


def test_sparse_nd_greater_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.sparse_nd.zeros(stype, shape)
        y = sparse_nd_ones(shape, stype)
        z = x >= y
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = y >= 0
        assert (z.asnumpy() == np.ones(shape)).all()
        z = 0 >= y
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = y >= 1
        assert (z.asnumpy() == np.ones(shape)).all()


def test_sparse_nd_lesser():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.sparse_nd.zeros(stype, shape)
        y = sparse_nd_ones(shape, stype)
        z = y < x
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = 0 < y
        assert (z.asnumpy() == np.ones(shape)).all()
        z = y < 0
        assert (z.asnumpy() == np.zeros(shape)).all()


def test_sparse_nd_lesser_equal():
    for stype in ['row_sparse', 'csr']:
        shape = rand_shape_2d()
        x = mx.sparse_nd.zeros(stype, shape)
        y = sparse_nd_ones(shape, stype)
        z = y <= x
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = 0 <= y
        assert (z.asnumpy() == np.ones(shape)).all()
        z = y <= 0
        assert (z.asnumpy() == np.zeros(shape)).all()
        z = 1 <= y
        assert (z.asnumpy() == np.ones(shape)).all()


def test_sparse_nd_binary():
    N = 100
    def check_binary(fn):
        for _ in range(N):
            ndim = 2
            oshape = np.random.randint(1, 6, size=(ndim,))
            bdim = 2
            lshape = list(oshape)
            rshape = list(oshape[ndim-bdim:])
            for i in range(bdim):
                sep = np.random.uniform(0, 1)
                if sep < 0.33:
                    lshape[ndim-i-1] = 1
                elif sep < 0.66:
                    rshape[bdim-i-1] = 1
            lhs = np.random.uniform(0, 1, size=lshape)
            rhs = np.random.uniform(0, 1, size=rshape)
            lhs_nd_csr = mx.nd.array(lhs).to_csr()
            rhs_nd_csr = mx.nd.array(rhs).to_csr()
            lhs_nd_rsp = mx.nd.array(lhs).to_rsp()
            rhs_nd_rsp = mx.nd.array(rhs).to_rsp()
            for lhs_nd, rhs_nd in [(lhs_nd_csr, rhs_nd_csr), (lhs_nd_rsp, rhs_nd_rsp)]:
                assert_allclose(fn(lhs, rhs),
                                fn(lhs_nd, rhs_nd).asnumpy(),
                                rtol=1e-4, atol=1e-4)

    check_binary(lambda x, y: x + y)
    check_binary(lambda x, y: x - y)
    check_binary(lambda x, y: x * y)
    check_binary(lambda x, y: x / y)
    check_binary(lambda x, y: x ** y)
    check_binary(lambda x, y: x > y)
    check_binary(lambda x, y: x < y)
    check_binary(lambda x, y: x >= y)
    check_binary(lambda x, y: x <= y)
    check_binary(lambda x, y: x == y)


def test_sparse_nd_binary_rop():
    N = 100
    def check(fn):
        for _ in range(N):
            ndim = 2
            shape = np.random.randint(1, 6, size=(ndim,))
            npy_nd = np.random.normal(0, 1, size=shape)
            csr_nd = mx.nd.array(npy_nd).to_csr()
            rsp_nd = mx.nd.array(npy_nd).to_rsp()
            for sparse_nd in [csr_nd, rsp_nd]:
                assert_allclose(
                    fn(npy_nd),
                    fn(sparse_nd).asnumpy(),
                    rtol=1e-4,
                    atol=1e-4
                )
    check(lambda x: 1 + x)
    check(lambda x: 1 - x)
    check(lambda x: 1 * x)
    check(lambda x: 1 / x)
    check(lambda x: 2 ** x)
    check(lambda x: 1 > x)
    check(lambda x: 0.5 > x)
    check(lambda x: 0.5 < x)
    check(lambda x: 0.5 >= x)
    check(lambda x: 0.5 <= x)
    check(lambda x: 0.5 == x)


def test_sparse_nd_negate():
    npy = np.random.uniform(-10, 10, rand_shape_2d())
    arr_csr = mx.nd.array(npy).to_csr()
    arr_rsp = mx.nd.array(npy).to_rsp()
    for arr in [arr_csr, arr_rsp]:
        assert_almost_equal(npy, arr.asnumpy())
        assert_almost_equal(-npy, (-arr).asnumpy())

        # a final check to make sure the negation (-) is not implemented
        # as inplace operation, so the contents of arr does not change after
        # we compute (-arr)
        assert_almost_equal(npy, arr.asnumpy())


def test_sparse_nd_output_fallback():
    shape = (10, 10)
    out = mx.sparse_nd.zeros('row_sparse', shape)
    mx.nd.random_normal(shape=shape, out=out)
    assert(np.sum(out.asnumpy()) != 0)


def test_sparse_nd_astype():
    stypes = ['row_sparse', 'csr']
    for stype in stypes:
        x = mx.sparse_nd.zeros(stype, rand_shape_2d(), dtype='float32')
        y = x.astype('int32')
        assert(y.dtype == np.int32), y.dtype


def test_sparse_ndarray_pickle():
    np.random.seed(0)
    repeat = 10
    dim0 = 40
    dim1 = 40
    stypes = ['row_sparse', 'csr']
    densities = [0, 0.01, 0.1, 0.2, 0.5]
    stype_dict = {'row_sparse': RowSparseNDArray, 'csr': CSRNDArray}
    for _ in range(repeat):
        shape = rand_shape_2d(dim0, dim1)
        for stype in stypes:
            for density in densities:
                a, _ = rand_sparse_ndarray(shape, stype, density)
                assert isinstance(a, stype_dict[stype])
                data = pkl.dumps(a)
                b = pkl.loads(data)
                assert isinstance(b, stype_dict[stype])
                assert same(a.asnumpy(), b.asnumpy())


def test_sparse_ndarray_save_load():
    # TODO(junwu): This function is a duplicate of mx.nd.load
    # which must be modified to use _ndarray_cls to generate
    # dense/sparse ndarrays. However, a circular import issue
    # arises when _ndarray_cls is used in mx.nd.load since
    # ndarray.py and sparse_ndarray.py would import each other.
    # We propose to put _ndarray_cls and all the functions calling
    # it in ndarray.py and sparse_ndarray.py into a util file
    # to resolve the circular import issue. This function will be
    # kept till then.
    def load(fname):
        """Loads an array from file.
        See more details in ``save``.
        Parameters
        ----------
        fname : str
            The filename.
        Returns
        -------
        list of NDArray or dict of str to NDArray
            Loaded data.
        """
        from mxnet.base import string_types, mx_uint, NDArrayHandle, check_call, c_str, _LIB
        if not isinstance(fname, string_types):
            raise TypeError('fname required to be a string')
        out_size = mx_uint()
        out_name_size = mx_uint()
        import ctypes
        handles = ctypes.POINTER(NDArrayHandle)()
        names = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXNDArrayLoad(c_str(fname),
                                      ctypes.byref(out_size),
                                      ctypes.byref(handles),
                                      ctypes.byref(out_name_size),
                                      ctypes.byref(names)))
        if out_name_size.value == 0:
            return [_ndarray_cls(NDArrayHandle(handles[i])) for i in range(out_size.value)]
        else:
            assert out_name_size.value == out_size.value
            from mxnet.base import py_str
            return dict(
                (py_str(names[i]), _ndarray_cls(NDArrayHandle(handles[i]))) for i in range(out_size.value))

    np.random.seed(0)
    repeat = 1
    stypes = ['default', 'row_sparse', 'csr']
    stype_dict = {'default': NDArray, 'row_sparse': RowSparseNDArray, 'csr': CSRNDArray}
    num_data = 20
    densities = [0, 0.01, 0.1, 0.2, 0.5]
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

        data_list2 = load(fname)
        assert len(data_list1) == len(data_list2)
        for x, y in zip(data_list1, data_list2):
            assert same(x.asnumpy(), y.asnumpy())

        data_map1 = {'ndarray xx %s' % i: x for i, x in enumerate(data_list1)}
        mx.nd.save(fname, data_map1)
        data_map2 = load(fname)
        assert len(data_map1) == len(data_map2)
        for k, x in data_map1.items():
            y = data_map2[k]
            assert same(x.asnumpy(), y.asnumpy())
    os.remove(fname)


if __name__ == '__main__':
    import nose
    nose.runmodule()
