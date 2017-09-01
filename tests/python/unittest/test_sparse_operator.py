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

from mxnet.test_utils import *


def check_elemwise_add_ex(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None):
    lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
    rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
    lhs_nd = rand_ndarray(shape, lhs_stype)
    rhs_nd = rand_ndarray(shape, rhs_stype)
    lhs_np = lhs_nd.asnumpy()
    rhs_np = rhs_nd.asnumpy()

    out_np = lhs_np + rhs_np
    test = mx.symbol.sparse.elemwise_add(lhs, rhs)
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
    if default_context().device_type == 'cpu':
        shapes = [rand_shape_2d(), rand_shape_3d()]
        for shape in shapes:
            check_elemwise_add_ex('default', 'default', shape)
            check_elemwise_add_ex('default', 'row_sparse', shape)
            check_elemwise_add_ex('row_sparse', 'default', shape)
            check_elemwise_add_ex('row_sparse', 'row_sparse', shape,
                                  lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse')


# TODO(haibin) randomize this test
def test_elemwise_add_ex_multiple_stages():
    if default_context().device_type == 'cpu':
        # prep data
        shape = (4, 2)
        ds_np = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        sp_np1 = np.array([[5, 10], [0, 0], [0, 0], [0, 0]])
        sp_np2 = np.array([[0, 0], [5, 10], [0, 0], [0, 0]])

        val1 = mx.nd.array([[5, 10]]);
        val2 = mx.nd.array([[5, 10]]);
        idx1 = mx.nd.array([0], dtype=np.int64);
        idx2 = mx.nd.array([1], dtype=np.int64);
        sp_nd1 = mx.nd.sparse.row_sparse_array(val1, idx1, shape)
        sp_nd2 = mx.nd.sparse.row_sparse_array(val2, idx2, shape)
        ds_nd = mx.nd.array(ds_np)

        # sparse + sparse = sparse
        sp_data1 = mx.symbol.Variable('sp_data1', stype='row_sparse')
        sp_data2 = mx.symbol.Variable('sp_data2', stype='row_sparse')
        ds_data = mx.symbol.Variable('ds_data')
        plus = mx.symbol.sparse.elemwise_add(sp_data1, sp_data2, name='plus')
        # sparse + dense = dense
        test = mx.symbol.sparse.elemwise_add(plus, ds_data)
        check_symbolic_forward(test, {'sp_data1': sp_nd1, 'sp_data2': sp_nd2,
                                      'ds_data': ds_nd}, [sp_np1 + sp_np2 + ds_np])

        arr_grads = [mx.nd.zeros(shape) for i in range(3)]
        exec_test = test.bind(default_context(), args={'sp_data1': sp_nd1, 'sp_data2': sp_nd2,
                                                       'ds_data': ds_nd}, args_grad=arr_grads)
        exec_test.forward(is_train=True)
        assert_almost_equal(exec_test.outputs[0].asnumpy(), sp_np1 + sp_np2 + ds_np)
        exec_test.backward(out_grads=exec_test.outputs)
        assert_almost_equal(arr_grads[0].asnumpy(), arr_grads[1].asnumpy())

def test_cast_storage_ex():
    def check_cast_storage(shape, density, from_stype, to_stype, check_numeric_grad=True):
        x = mx.symbol.Variable('x', stype=from_stype)
        x_nd = rand_ndarray(shape, from_stype, density=density)
        x_np = x_nd.asnumpy()
        out_np = x_np
        test = mx.symbol.cast_storage(x, stype=to_stype)
        location = {'x': x_nd}
        check_symbolic_forward(test, location, [out_np])
        # consider disable the numeric grad check for gpu block kernel since the input is large
        if check_numeric_grad:
            check_numeric_gradient(test, location)
        grad_stypes = {'x': to_stype}
        check_symbolic_backward(test, location, [out_np], [out_np], grad_stypes=grad_stypes)

    density = [1.00, 0.50, 0.05, 0.01]
    for d in density:
        shape_2d = rand_shape_2d()
        shape_3d = rand_shape_3d()
        check_cast_storage(shape_2d, d, 'csr', 'default')
        check_cast_storage(shape_2d, d, 'default', 'csr')
        check_cast_storage(shape_2d, d, 'row_sparse', 'default')
        check_cast_storage(shape_2d, d, 'default', 'row_sparse')
        check_cast_storage(shape_3d, d, 'row_sparse', 'default')
        check_cast_storage(shape_3d, d, 'default', 'row_sparse')
        for i in range(4, 6):
            shape = rand_shape_nd(i, 5)
            check_cast_storage(shape, d, 'default', 'row_sparse')
            check_cast_storage(shape, d, 'row_sparse', 'default')
        # Test specific gpu kernels
        if default_context().device_type is 'gpu':
            dim0 = rnd.randint(1, 10)
            # test gpu thread kernel
            check_cast_storage((dim0, rnd.randint(  1,   32)), d, 'default', 'csr')
            # test gpu warp   kernel
            check_cast_storage((dim0, rnd.randint( 32,  512)), d, 'default', 'csr')
            # test gpu block  kernel
            check_cast_storage((dim0, rnd.randint(512, 1024)), d, 'default', 'csr',
                               check_numeric_grad=False)
            # test gpu thread kernel
            check_cast_storage((dim0, rnd.randint(  1,   32)), d, 'default', 'row_sparse')
            # test gpu warp   kernel
            check_cast_storage((dim0, rnd.randint( 32,  512)), d, 'default', 'row_sparse')
            # test gpu block  kernel
            check_cast_storage((dim0, rnd.randint(512, 1024)), d, 'default', 'row_sparse',
                               check_numeric_grad=False)

def test_sparse_dot():
    def test_dot_csr(lhs_shape, rhs_shape, rhs_stype, trans_lhs, lhs_density, rhs_density):
        lhs_nd = rand_ndarray(lhs_shape, 'csr', density=lhs_density)
        lhs_dns = lhs_nd.tostype('default')
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_density)
        rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.tostype('default')

        out = mx.nd.dot(lhs_nd, rhs_nd, transpose_a=trans_lhs)
        out_dns = mx.nd.dot(lhs_dns, rhs_dns, transpose_a=trans_lhs)
        out_np = out_dns.asnumpy()
        assert_almost_equal(out.asnumpy(), out_np, rtol=1e-4, atol=1e-5)

        # test symbolic forward
        lhs = mx.symbol.Variable('lhs', stype='csr')
        rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
        out = mx.symbol.sparse.dot(lhs, rhs, transpose_a=trans_lhs)
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
    def check_sparse_retain(shape, density, index_type=np.int64):
        num_rows = shape[0]
        rsp, _ = rand_sparse_ndarray(shape=shape, stype='row_sparse', density=density)
        length = np.random.randint(1, num_rows + 1)
        idx = random_sample(list(range(0, num_rows)), length)
        idx.sort()
        dns = rsp.asnumpy()
        tensor_retained_expected = np.zeros(shape)
        for i in idx:
            tensor_retained_expected[i][:] = dns[i]
        indices = mx.nd.array(idx, dtype=index_type)
        rsp_retained = mx.nd.sparse.retain(rsp, indices=indices)
        assert same(tensor_retained_expected, rsp_retained.asnumpy())

        # check numeric gradient
        data = mx.symbol.Variable('data')
        idx = mx.symbol.Variable('indices')
        sym = mx.sym.sparse.retain(data=data, indices=idx)
        check_numeric_gradient(sym, [rsp, indices], grad_nodes=['data'],
                               grad_stype_dict={'data': 'row_sparse'})

    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    densities = [0.01, 0.1, 0.2, 0.5, 0.8, 1.0]
    index_types = [np.float32, np.int32, np.int64]
    for density in densities:
        for itype in index_types:
            check_sparse_retain(shape, density, itype)
            check_sparse_retain(shape_3d, density, itype)


def test_sparse_nd_zeros():
    def check_sparse_nd_zeros(stype, shape):
        zero = mx.nd.zeros(shape)
        sparse_zero = mx.nd.zeros(shape=shape, stype=stype)
        assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros('row_sparse', shape)
    check_sparse_nd_zeros('csr', shape)
    check_sparse_nd_zeros('default', shape)

def test_sparse_nd_zeros_like():
    def check_sparse_nd_zeros_like(stype, shape):
        zero = mx.nd.zeros(shape, stype=stype)
        zero_like = mx.nd.sparse.zeros_like(zero)
        assert_almost_equal(zero.asnumpy(), zero_like.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros_like('row_sparse', shape)
    check_sparse_nd_zeros_like('csr', shape)


def test_sparse_square_sum():
    if default_context().device_type == 'cpu':
        dim0 = 30
        dim1 = 30
        axes = [0, 1]
        keepdims = [False, True]
        densities = [0, 0.01, 0.1, 0.2, 0.5]
        for density in densities:
            shape = rand_shape_2d(dim0, dim1)
            rsp = rand_ndarray(shape, 'row_sparse', density)
            dns = rsp.tostype('default')
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

                    rsp_data = mx.sym.Variable('data', stype='row_sparse')
                    test = mx.symbol._internal._square_sum(rsp_data, axis=axis, keepdims=keepdim)

                    # check symbolic backward since ograd can be a rsp
                    # and cannot be checked through check_numeric_gradient
                    # because it will add a loss layer as the output layer
                    # which makes ograd of the square_sum dense
                    if axis == 1 and keepdims:
                        dns_data = mx.sym.Variable('data')
                        baseline = mx.sym.sum(mx.sym.square(dns_data), axis=axis, keepdims=keepdim)
                        igrad_expected = mx.nd.empty(dns.shape)
                        baseline_exec = baseline.bind(default_context(), args=[dns],
                                                      args_grad=[igrad_expected])
                        baseline_exec.forward(is_train=True)
                        baseline_exec.backward([ret_expected])
                        check_symbolic_backward(test, [rsp], [ret], [igrad_expected.asnumpy()],
                                                grad_stypes={'data': 'row_sparse'})

                    # check numeric gradient
                    check_numeric_gradient(test, [rsp], grad_stype_dict={'data': 'row_sparse'},
                                           atol=1e-2, rtol=0.1)

def test_sparse_storage_fallback():
    """ test operators which don't implement FComputeEx or FStatefulComputeEx """
    if default_context().device_type == 'cpu':
        def check_broadcast_add(shape, lhs_stype, rhs_stype):
            lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
            rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
            lhs_nd = rand_ndarray(shape, lhs_stype)
            rhs_nd = rand_ndarray(shape, rhs_stype)
            lhs_dns = mx.nd.cast_storage(lhs_nd, stype='default')
            rhs_dns = mx.nd.cast_storage(rhs_nd, stype='default')

            out_dns = (lhs_dns + rhs_dns).asnumpy()
            test = mx.symbol.broadcast_add(lhs, rhs)
            location = {'lhs': lhs_nd, 'rhs': rhs_nd}
            check_symbolic_forward(test, location, [out_dns])
            check_numeric_gradient(test, location)
            check_symbolic_backward(test, location, [out_dns], [out_dns, out_dns])

        def np_softmax(x, axis=-1):
            # fix for old numpy on Travis not supporting keepdims
            # x = x - np.max(x, axis=-1, keepdims=True)
            x = x - np.max(x, axis=axis, keepdims=True)
            x = np.exp(x)
            # x /= np.sum(x, axis=-1, keepdims=True)
            x /= np.sum(x, axis=axis, keepdims=True)
            return x

        def check_softmax_with_shape(lhs_stype, rhs_stype, shape, preserve_shape=False):
            # bind with label
            ctx = default_context()
            X = mx.symbol.Variable('X', stype=lhs_stype)
            L = mx.symbol.Variable('L', stype=rhs_stype)
            Y = mx.symbol.SoftmaxOutput(data=X, label=L, preserve_shape=preserve_shape)
            x = rand_ndarray(shape, lhs_stype)
            l = rand_ndarray(shape, rhs_stype)
            l[:] = np_softmax(l.asnumpy())
            grad = mx.nd.empty(shape, ctx=ctx)
            exec1 = Y.bind(ctx, args = [x, l], args_grad = {'X': grad})
            exec1.forward(is_train=True)
            out = exec1.outputs[0].asnumpy()
            assert_almost_equal(out, np_softmax(x.asnumpy()), rtol=1e-4)
            exec1.backward()
            assert_almost_equal(grad.asnumpy(), np_softmax(x.asnumpy()) - l.asnumpy(),
                                rtol=1e-3, atol=1e-4)

        def check_concat(shape, lhs_stype, rhs_stype):
            x = mx.symbol.Variable('x', stype=lhs_stype)
            w = mx.symbol.Variable('w', stype=rhs_stype)
            test = mx.sym.Concat(x, w)
            x_nd = rand_ndarray(shape, lhs_stype)
            w_nd = rand_ndarray(shape, rhs_stype)
            location = {'x': x_nd, 'w': w_nd}
            check_numeric_gradient(test, location)

        shape = rand_shape_2d()
        stypes = ['default', 'csr', 'row_sparse']
        for lhs in stypes:
            for rhs in stypes:
                check_broadcast_add(shape, lhs, rhs)
                check_concat(shape, lhs, rhs)
                check_softmax_with_shape(lhs, rhs, shape, preserve_shape=False)
                check_softmax_with_shape(rhs, rhs, shape, preserve_shape=True)


def test_sparse_elementwise_sum():
    if default_context().device_type == 'cpu':
        def check_sparse_elementwise_sum_with_shape(stype, shape, n):
            # forward
            inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
            out = mx.symbol.sparse.add_n(*inputs, name='esum')
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
