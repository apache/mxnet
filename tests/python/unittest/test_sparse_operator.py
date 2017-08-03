from mxnet.test_utils import *


def check_elemwise_add_ex(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None):
    lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
    rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
    lhs_nd = rand_ndarray(shape, lhs_stype)
    rhs_nd = rand_ndarray(shape, rhs_stype)
    lhs_np = lhs_nd.asnumpy()
    rhs_np = rhs_nd.asnumpy()

    out_np = lhs_np + rhs_np
    test = mx.symbol.elemwise_add(lhs, rhs)
    location = {'lhs': lhs_nd, 'rhs': rhs_nd}
    check_symbolic_forward(test, location, [out_np])
    check_numeric_gradient(test, location)
    grad_stypes = {}
    if lhs_grad_stype is not None and lhs_grad_stype != 'default':
        grad_stypes['lhs'] = lhs_grad_stype
    if rhs_grad_stype is not None and rhs_grad_stype != 'default':
        grad_stypes['rhs'] = rhs_grad_stype
    check_symbolic_backward(test, location, [out_np], [out_np, out_np],
                            grad_stypes=grad_stypes)


def test_elemwise_add_ex():
    shapes = [rand_shape_2d(), rand_shape_3d()]
    for shape in shapes:
        check_elemwise_add_ex('default', 'default', shape)
        check_elemwise_add_ex('default', 'row_sparse', shape)
        check_elemwise_add_ex('row_sparse', 'default', shape)
        check_elemwise_add_ex('row_sparse', 'row_sparse', shape,
                              lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse')


# TODO(haibin) randomize this test
def test_elemwise_add_ex_multiple_stages():
    # prep data
    shape = (4, 2)
    ds_np = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    sp_np1 = np.array([[5, 10], [0, 0], [0, 0], [0, 0]])
    sp_np2 = np.array([[0, 0], [5, 10], [0, 0], [0, 0]])

    val1 = mx.nd.array([[5, 10]]);
    val2 = mx.nd.array([[5, 10]]);
    idx1 = mx.nd.array([0], dtype=np.int64);
    idx2 = mx.nd.array([1], dtype=np.int64);
    sp_nd1 = mx.nd.row_sparse(val1, idx1, shape)
    sp_nd2 = mx.nd.row_sparse(val2, idx2, shape)
    ds_nd = mx.nd.array(ds_np)

    # sparse + sparse = sparse
    sp_data1 = mx.symbol.Variable('sp_data1', stype='row_sparse')
    sp_data2 = mx.symbol.Variable('sp_data2', stype='row_sparse')
    ds_data = mx.symbol.Variable('ds_data')
    plus = mx.symbol.elemwise_add(sp_data1, sp_data2, name='plus')
    # sparse + dense = dense
    test = mx.symbol.elemwise_add(plus, ds_data)
    check_symbolic_forward(test, {'sp_data1': sp_nd1, 'sp_data2': sp_nd2,
                                  'ds_data': ds_nd}, [sp_np1 + sp_np2 + ds_np])

    arr_grads = [mx.nd.zeros(shape) for i in range(3)]
    exec_test = test.bind(default_context(), args={'sp_data1': sp_nd1, 'sp_data2': sp_nd2,
                                                   'ds_data': ds_nd}, args_grad=arr_grads)
    exec_test.forward(is_train=True)
    assert_almost_equal(exec_test.outputs[0].asnumpy(), sp_np1 + sp_np2 + ds_np)
    exec_test.backward(out_grads=exec_test.outputs)
    assert_almost_equal(arr_grads[0].asnumpy(), arr_grads[1].asnumpy())


# TODO(haibin) also add test for backward pass.
def test_cast_storage_ex():
    def test_rsp_to_dns(shape, density):
        rsp_in, (data, row_idx) = rand_sparse_ndarray(shape, 'row_sparse', density)
        dns_out = mx.nd.cast_storage(rsp_in, stype='default')
        assert same(rsp_in.asnumpy(), dns_out.asnumpy())

    def test_dns_to_rsp(shape, density):
        rsp_in, (data, row_idx) = rand_sparse_ndarray(shape, 'row_sparse', density)
        rsp_out = mx.nd.cast_storage(mx.nd.array(rsp_in.todense(), dtype=default_dtype()), stype='row_sparse')
        assert same(rsp_in.asnumpy(), rsp_out.asnumpy())

    def test_csr_to_dns(shape, density):
        csr_in, (indptr, indices, values) = rand_sparse_ndarray(shape, 'csr', density)
        dns_out = mx.nd.cast_storage(csr_in, stype='default')
        assert same(csr_in.asnumpy(), dns_out.asnumpy())

    def test_dns_to_csr(shape, density):
        csr_in, (indptr, colidx, data) = rand_sparse_ndarray(shape, 'csr', density)
        csr_out = mx.nd.cast_storage(mx.nd.array(csr_in.todense(), dtype=default_dtype()), stype='csr')
        assert same(csr_in.asnumpy(), csr_out.asnumpy())

    density = [1.00, 0.50, 0.10, 0.05, 0.01]
    for d in density:
        shape_2d = rand_shape_2d()
        shape_3d = rand_shape_3d()
        test_csr_to_dns(shape_2d, d)
        test_dns_to_csr(shape_2d, d)
        test_rsp_to_dns(shape_2d, d)
        test_dns_to_rsp(shape_2d, d)
        test_rsp_to_dns(shape_3d, d)
        test_dns_to_rsp(shape_3d, d)
        for i in range(4, 6):
            shape = rand_shape_nd(i, 5)
            test_dns_to_rsp(shape, d)
            test_rsp_to_dns(shape, d)
        # Test specific gpu kernels
        if default_context().device_type is 'gpu':
            test_dns_to_csr((rnd.randint(1, 10), rnd.randint(  1,   32)), d) # test gpu thread kernel
            test_dns_to_csr((rnd.randint(1, 10), rnd.randint( 32,  512)), d) # test gpu warp   kernel
            test_dns_to_csr((rnd.randint(1, 10), rnd.randint(512, 1024)), d) # test gpu block  kernel
            test_dns_to_rsp((rnd.randint(1, 10), rnd.randint(  1,   32)), d) # test gpu thread kernel
            test_dns_to_rsp((rnd.randint(1, 10), rnd.randint( 32,  512)), d) # test gpu warp   kernel
            test_dns_to_rsp((rnd.randint(1, 10), rnd.randint(512, 1024)), d) # test gpu block  kernel


def test_sparse_dot():
    def test_dot_csr(lhs_shape, rhs_shape, rhs_stype, trans_lhs, lhs_density, rhs_density):
        lhs_nd = rand_ndarray(lhs_shape, 'csr', density=lhs_density)
        lhs_dns = lhs_nd.todense()
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_density)
        rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.todense()

        out = mx.nd.dot(lhs_nd, rhs_nd, transpose_a=trans_lhs)
        out_dns = mx.nd.dot(lhs_dns, rhs_dns, transpose_a=trans_lhs)
        out_np = out_dns.asnumpy()
        assert_almost_equal(out.asnumpy(), out_np, rtol=1e-4, atol=1e-5)

        # test symbolic forward
        lhs = mx.symbol.Variable('lhs', stype='csr')
        rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
        out = mx.symbol.dot(lhs, rhs, transpose_a=trans_lhs)
        location = {'lhs': lhs_nd, 'rhs': rhs_nd}
        check_symbolic_forward(out, location, [out_np], rtol=1e-3, atol=1e-4)

        # test symbolic backward
        backward_trans = not trans_lhs
        rhs_backward_grad = mx.nd.dot(lhs_dns, out_dns, transpose_a=backward_trans).asnumpy()
        expected = {'rhs': rhs_backward_grad}
        check_symbolic_backward(out, location, [out_np], expected,
                                grad_req={'lhs': 'null', 'rhs': 'write'},
                                rtol=1e-3, atol=1e-4)

    density = [1.00, 0.50, 0.10, 0.05, 0.01]
    for lhs_d in density:
        lhs_shape = rand_shape_2d(50, 200)
        rhs_d = 1
        test_dot_csr(lhs_shape, (lhs_shape[1], 1), 'default', False, lhs_d, rhs_d) # test gpu SpMV
        test_dot_csr(lhs_shape, (lhs_shape[0], 1), 'default', True , lhs_d, rhs_d) # (vector kernel)
        test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(5, 10)), 'default', False, lhs_d, rhs_d) # test gpu SpMM
        test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(5, 10)), 'default', True , lhs_d, rhs_d) # (scalar kernel)
        for rhs_d in density:
            test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'row_sparse', False, lhs_d, rhs_d)
            test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'row_sparse', True, lhs_d, rhs_d)


def test_sparse_slice():
    def check_csr_slice(shape, slice_input):
        storage_type = 'csr'
        B, _ = rand_sparse_ndarray(shape, storage_type)
        np = B.asnumpy()
        begin = rnd.randint(0, B.shape[0] - 1)
        end = rnd.randint(begin + 1, B.shape[0])
        nd_slice = mx.nd.crop(B, begin=begin, end=end)
        assert same(nd_slice.asnumpy(), np[begin:end]), (nd_slice.asnumpy(), np[begin:end])

    shape = (rnd.randint(7, 15), rnd.randint(1, 10))
    check_csr_slice(shape, True)
    check_csr_slice(shape, False)


def test_sparse_retain():
    def check_sparse_retain(shape):
        num_rows = shape[0]
        rsp, _ = rand_sparse_ndarray(shape=shape, stype='row_sparse', density=0.5)
        length = np.random.randint(1, num_rows + 1)
        idx = random_sample(list(range(0, num_rows)), length)
        idx.sort()
        dns = rsp.asnumpy()
        tensor_retained_expected = np.zeros(shape)
        for i in idx:
            tensor_retained_expected[i][:] = dns[i]
        indices = mx.nd.array(idx)
        rsp_retained = mx.nd.sparse_retain(rsp, indices=indices)
        assert same(tensor_retained_expected, rsp_retained.asnumpy())

        # check numeric gradient
        data = mx.symbol.Variable('data')
        idx = mx.symbol.Variable('indices')
        sym = mx.sym.sparse_retain(data=data, indices=idx)
        check_numeric_gradient(sym, [rsp, indices], grad_nodes=['data'], grad_stype_dict={'data': 'row_sparse'})
    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    check_sparse_retain(shape)
    check_sparse_retain(shape_3d)


def test_sparse_nd_zeros():
    def check_sparse_nd_zeros(stype, shape):
        zero = mx.nd.zeros(shape)
        sparse_zero = mx.nd.zeros(shape=shape, stype=stype)
        assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros('row_sparse', shape)
    check_sparse_nd_zeros('csr', shape)
    check_sparse_nd_zeros('default', shape)


def test_sparse_square_sum():
    dim0 = 30
    dim1 = 30
    axes = [0, 1]
    keepdims = [False, True]
    densities = [0, 0.01, 0.1, 0.2, 0.5]
    for density in densities:
        shape = rand_shape_2d(dim0, dim1)
        rsp = rand_ndarray(shape, 'row_sparse', density)
        dns = rsp.todense()
        for axis in axes:
            for keepdim in keepdims:
                ret = mx.nd._internal._square_sum(rsp, axis=axis, keepdims=keepdim)
                if axis == 1 and keepdim:
                    assert ret.stype == 'row_sparse'
                else:
                    assert ret.stype == 'default'
                ret_expected = mx.nd.sum(dns*dns, axis=axis, keepdims=keepdim)
                # check forward result
                assert same(ret.asnumpy(), ret_expected.asnumpy())

                # check numeric gradient
                data = mx.sym.Variable('data', stype='row_sparse')
                test = mx._symbol_internal._square_sum(data, axis=axis, keepdims=keepdim)
                check_numeric_gradient(test, [rsp], grad_stype_dict={'data': 'row_sparse'},
                                       atol=1e-2, rtol=0.1)


def test_sparse_elementwise_sum():
    def check_sparse_elementwise_sum_with_shape(stype, shape, n):
        # forward
        inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
        out = mx.symbol.add_n(*inputs, name='esum')
        arr = []
        arr_grad = [mx.nd.empty(shape) for _ in range(n)]
        densities = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
        for i in range(n):
            arr.append(rand_ndarray(shape, stype, np.random.randint(0, len(densities))))

        exec1 = out.bind(default_context(),
                         args=arr,
                         args_grad=arr_grad)
        exec1.forward(is_train=True)
        out1 = exec1.outputs[0].asnumpy()
        out = sum(a.asnumpy() for a in arr)
        assert_almost_equal(out, out1)

        out_grad = mx.nd.empty(shape)
        out_grad[:] = np.random.uniform(-10, 10, shape)
        # backward
        exec1.backward([out_grad])
        for a in arr_grad:
            assert_almost_equal(a.asnumpy(), out_grad.asnumpy())

    maxdim = 5
    for dim in range(2, maxdim):
        shape = tuple(np.random.randint(5, 10, size=dim))
        check_sparse_elementwise_sum_with_shape('row_sparse', shape, np.random.randint(1, 9))


if __name__ == '__main__':
    import nose
    nose.runmodule()
