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
from mxnet.base import MXNetError
import pytest
from common import assertRaises
import random
import warnings

def is_scalar(var):
    return False if hasattr(var, "__len__") else True

def get_result_type(call, dflt_stype):
    """Try to infer result storage type for a sparse matrix and a given unary operation"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        result = do_normalize(call(zero))
        if not almost_equal(result, zero, equal_nan=True):
            expected_result_type = 'default'
        else:
            if dflt_stype is not None:
                expected_result_type = dflt_stype;
            else:
                expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_result_type_with_scalar(call, dflt_stype):
    """Try to infer result storage type when operating a sparse matrices and a scalar"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        result = call(zero, 5)

        if not almost_equal(result, zero, equal_nan=True):
            expected_result_type = 'default'
        else:
            if dflt_stype is not None:
                expected_result_type = dflt_stype;
            else:
                expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_result_type_2(call, dflt_stype):
    """Try to infer result storage type when operating on two sparse matrices"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        need_default = False
        for outer in [zero, np.ones(zero.shape)]:
            for inner in [zero, np.ones(zero.shape)]:
                result = do_normalize(call(outer, inner))
                if not almost_equal(result, zero, equal_nan=True):
                    need_default = True
                    break
            if need_default is True:
                break

        if not need_default and dflt_stype is not None:
            expected_result_type = dflt_stype
        else:
            expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_result_type_3(call, dflt_stype):
    """Try to infer result storage type when operating on three sparse matrices"""
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        need_default = False
        for moon in [zero]:
            for outer in [zero]:
                for inner in [zero]:
                    res_1, res_2 = call(moon, outer, inner)
                    result = do_normalize(res_1)
                    if not almost_equal(result, zero, equal_nan=True):
                        need_default = True
                        break
                    result = do_normalize(res_2)
                    if not almost_equal(result, zero, equal_nan=True):
                        need_default = True
                        break
                if need_default is True:
                    break
            if need_default is True:
                break

        if not need_default and dflt_stype is not None:
            expected_result_type = dflt_stype
        else:
            expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_fw_bw_result_types(forward_numpy_call,  fwd_res_dflt,
                           backward_numpy_call, bwd_res_dflt):

    return (get_result_type(forward_numpy_call,  fwd_res_dflt),
            get_result_type(backward_numpy_call, bwd_res_dflt))


def get_fw_bw_result_types_2(forward_numpy_call,  fwd_res_dflt,
                             backward_numpy_call, bwd_res_dflt):
    return (get_result_type(forward_numpy_call,  fwd_res_dflt),
            get_result_type_2(backward_numpy_call, bwd_res_dflt))

def get_fw_bw_result_types_with_scalar(forward_numpy_call,  fwd_res_dflt,
                                       backward_numpy_call, bwd_res_dflt):
    return (get_result_type_with_scalar(forward_numpy_call,  fwd_res_dflt),
            get_result_type_with_scalar(backward_numpy_call, bwd_res_dflt))

def gen_rsp_random_indices(shape, density=.5, force_indices=None):
    assert density >= 0 and density <= 1
    indices = set()
    if force_indices is not None:
        for val in force_indices:
            indices.add(int(val))
    if not np.isclose(density, .0, rtol=1.e-3, atol=1.e-3, equal_nan=True) and len(shape) > 0:
        row_count = shape[0]
        for i in range(row_count):
            r = random.uniform(0, 1)
            if r <= density and len(indices) < shape[0]:
                indices.add(i)
    assert len(indices) <= shape[0]
    return list(indices)


def all_zero(var):
    return 0

@pytest.mark.skip(reason="https://github.com/apache/incubator-mxnet/issues/18740")
def test_elemwise_binary_ops():
    # skip testing on GPU because only CPU ops are implemented
    if default_context().device_type is 'gpu':
        return

    def test_elemwise_binary_op(name, lhs_stype, rhs_stype, shape,
                                forward_mxnet_call, forward_numpy_call, backward_numpy_call,
                                lhs_grad_stype,
                                rhs_grad_stype,
                                expected_result_storage_type=None,
                                modifier_func=None,
                                lhs_density=.5,
                                rhs_density=.5,
                                force_lr_overlap=False,
                                force_grad_overlap=False,
                                ograd_density=0.0,
                                skip_gradient_check=False,
                                shuffle_csr_indices=True,
                                verbose=False):
        if lhs_grad_stype is None:
            lhs_grad_stype = lhs_stype
        if rhs_grad_stype is None:
            rhs_grad_stype = rhs_stype

        lhs_grad_stype = get_result_type_3(backward_numpy_call, lhs_grad_stype)
        rhs_grad_stype = get_result_type_3(backward_numpy_call, rhs_grad_stype)

        if verbose is True:
            print("testing: {}  lhs={}, rhs={}, lhs_grad_stype={}, rhs_grad_stype={}"
                  .format(name, lhs_stype, rhs_stype, lhs_grad_stype, rhs_grad_stype))

        # Output type should be same as lvalue type, unless otherwise specified
        if expected_result_storage_type is None:
            if lhs_stype == 'default' or rhs_stype == 'default':
                expected_result_storage_type = 'default'
            else:
                expected_result_storage_type = lhs_stype

        lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
        rhs = mx.symbol.Variable('rhs', stype=rhs_stype)

        grad_stypes = dict()
        grad_stypes['lhs'] = lhs_grad_stype
        grad_stypes['rhs'] = rhs_grad_stype

        if lhs_stype == 'default':
            lhs_nd = rand_ndarray(shape, 'default')
            if abs(lhs_density) < 1e-4:
                func = all_zero
            else:
                func = modifier_func
            lhs_nd = mx.nd.array(assign_each(lhs_nd.asnumpy(), func))
        else:
            lhs_nd = create_sparse_array_zd(
                shape, lhs_stype, density=lhs_density,
                modifier_func=modifier_func,
                shuffle_csr_indices=shuffle_csr_indices,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=lhs_density,
                    force_indices=[(shape[0]/2)] if force_lr_overlap is True else None
                ))

        if rhs_stype == 'default':
            rhs_nd = rand_ndarray(shape, 'default')
            if abs(rhs_density) < 1e-4:
                func = all_zero
            else:
                func = modifier_func
            rhs_nd = mx.nd.array(assign_each(rhs_nd.asnumpy(), func))
        else:
            rhs_nd = create_sparse_array_zd(
                shape, rhs_stype, density=rhs_density,
                modifier_func=modifier_func,
                shuffle_csr_indices=shuffle_csr_indices,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=rhs_density,
                    force_indices=[(shape[0]/2)] if force_lr_overlap is True else None
                ))

        lhs_np = lhs_nd.asnumpy()
        rhs_np = rhs_nd.asnumpy()

        if verbose is True:
            print("lhs input: {}".format(lhs_np))
            print("rhs input: {}".format(rhs_np))

        out_np = forward_numpy_call(lhs_np, rhs_np)

        if verbose is True:
            print("out_np: {}".format(out_np))

        test = forward_mxnet_call(lhs, rhs)

        location = {'lhs': lhs_nd, 'rhs': rhs_nd}

        outputs = check_symbolic_forward(test, location, [out_np], equal_nan=True)
        assert len(outputs) == 1
        assert outputs[0].stype == expected_result_storage_type

        if verbose is True:
            print ("mx forward output: ", outputs[0].asnumpy())
            print ("lhs_nd: ", lhs_nd.stype)
            print ("rhs_nd: ", rhs_nd.stype)
            print ("forward output: ", outputs[0].stype)

        if outputs[0].stype != 'default':
            out_grad = create_sparse_array_zd(
                shape, outputs[0].stype, density=ograd_density,
                data_init=1,
                modifier_func=lambda x: 2,
                shuffle_csr_indices=shuffle_csr_indices,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=ograd_density,
                    force_indices=[(shape[0]/2)] if force_grad_overlap is True else None
                ))
        else:
            if abs(ograd_density) < 1e-4:
                out_grad = mx.nd.array(np.zeros(shape))
            else:
                out_grad = mx.nd.array(np.ones(shape))


        out_grad_np = out_grad.asnumpy()

        if verbose is True:
            print("out_grad_np", out_grad_np)

        ingrad_lhs_np, ingrad_rhs_np = backward_numpy_call(out_grad_np, lhs_np, rhs_np)

        if verbose is True:
            print("out_grad", out_grad.asnumpy())
            print("ingrad_lhs_np", ingrad_lhs_np)
            print("ingrad_rhs_np", ingrad_rhs_np)

        igrads_result = check_symbolic_backward(test, location, [out_grad],
                                                [ingrad_lhs_np, ingrad_rhs_np],
                                                grad_stypes=grad_stypes,
                                                equal_nan=True)

        if verbose is True:
            print("ingrad_lhs", igrads_result['lhs'].asnumpy())
            print("ingrad_rhs", igrads_result['rhs'].asnumpy())

        assert len(igrads_result) == 2

        if lhs_grad_stype is not None:
            assert igrads_result['lhs'].stype == lhs_grad_stype
        if rhs_grad_stype is not None:
            assert igrads_result['rhs'].stype == rhs_grad_stype
        if not skip_gradient_check:
            check_numeric_gradient(test, location,
                                   grad_stype_dict=grad_stypes)

    def check_all(l, r, check_function):
        assert l.shape == r.shape
        return check_function(l, r)

    def gt(l, r):
        return check_all(l, r, lambda a, b: a > b)

    def ge(l, r):
        return check_all(l, r, lambda a, b: a >= b)

    def lt(l, r):
        return check_all(l, r, lambda a, b: a < b)

    def le(l, r):
        return check_all(l, r, lambda a, b: a <= b)

    def elemwise_mul_stype(lstype, rstype):
        if lstype == rstype:
            return lstype
        elif lstype == 'default' and rstype == 'row_sparse':
            return 'row_sparse'
        elif lstype == 'row_sparse' and rstype == 'default':
            return 'row_sparse'
        else:
            return 'default'

    def elemwise_mul_lhs_grad_stype(lstype, rstype):
        return elemwise_mul_stype(elemwise_mul_stype(lstype, rstype), rstype)

    def elemwise_mul_rhs_grad_stype(lstype, rstype):
        return elemwise_mul_stype(elemwise_mul_stype(lstype, rstype), lstype)

    def check_elemwise_binary_ops(lhs_stype, rhs_stype, shape,
                                  lhs_grad_stype=None, rhs_grad_stype=None,
                                  lhs_density=.5, rhs_density=.5,
                                  force_lr_overlap=False,
                                  force_grad_overlap=False,
                                  ograd_density=0.0):
        test_elemwise_binary_op("elemwise_add", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.sparse.elemwise_add(l, r),
                                lambda l, r: l + r,
                                lambda outg, l, r: (outg, outg),
                                lhs_grad_stype, rhs_grad_stype,
                                ograd_density=ograd_density,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                verbose=False)

        if ((lhs_stype is 'default' and rhs_stype is 'row_sparse') or
            (lhs_stype is 'row_sparse' and rhs_stype is 'row_sparse') and (rhs_density == 0.0)):
            test_elemwise_binary_op("elemwise_add", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_add(l, r, out=l),
                                    lambda l, r: l + r,
                                    lambda outg, l, r: (outg, outg),
                                    lhs_grad_stype, rhs_grad_stype,
                                    ograd_density=ograd_density,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    verbose=False)
            test_elemwise_binary_op("elemwise_sub", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_sub(l, r, out=l),
                                    lambda l, r: l - r,
                                    lambda outg, l, r: (outg, -outg),
                                    lhs_grad_stype, rhs_grad_stype,
                                    ograd_density=ograd_density,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    verbose=False)
        if ((lhs_stype is 'row_sparse' and rhs_stype is 'row_sparse') and (lhs_density == 0.0)):
            test_elemwise_binary_op("elemwise_add", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_add(l, r, out=r),
                                    lambda l, r: l + r,
                                    lambda outg, l, r: (outg, outg),
                                    lhs_grad_stype, rhs_grad_stype,
                                    ograd_density=ograd_density,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    verbose=False)
            test_elemwise_binary_op("elemwise_sub", lhs_stype, rhs_stype, shape,
                                    lambda l, r: mx.sym.sparse.elemwise_sub(l, r, out=l),
                                    lambda l, r: l - r,
                                    lambda outg, l, r: (outg, -outg),
                                    lhs_grad_stype, rhs_grad_stype,
                                    ograd_density=ograd_density,
                                    force_lr_overlap=force_lr_overlap,
                                    force_grad_overlap=force_grad_overlap,
                                    lhs_density=lhs_density, rhs_density=rhs_density,
                                    verbose=False)

        test_elemwise_binary_op("elemwise_sub", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.sparse.elemwise_sub(l, r),
                                lambda l, r: l - r,
                                lambda outg, l, r: (outg, -outg),
                                lhs_grad_stype, rhs_grad_stype,
                                ograd_density=ograd_density,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density,
                                rhs_density=rhs_density,
                                verbose=False)

        test_elemwise_binary_op("elemwise_mul", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.sparse.elemwise_mul(l, r),
                                lambda l, r: l * r,
                                lambda outg, l, r: (outg * r, outg * l),
                                elemwise_mul_lhs_grad_stype(lhs_stype, rhs_stype),
                                elemwise_mul_rhs_grad_stype(lhs_stype, rhs_stype),
                                expected_result_storage_type=elemwise_mul_stype(lhs_stype, rhs_stype),
                                ograd_density=ograd_density,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                verbose=False)

        test_elemwise_binary_op("elemwise_div", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.sparse.elemwise_div(l, r),
                                lambda l, r: l / r,
                                lambda outg, l, r: (outg * (1/r), outg * (-l/(r*r))),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                ograd_density=ograd_density,
                                expected_result_storage_type='default',
                                skip_gradient_check=True,
                                verbose=False)

        test_elemwise_binary_op("maximum", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym._internal._maximum(l, r),
                                lambda l, r: np.maximum(l, r),
                                lambda outg, l, r: (outg * ge(l, r), outg * lt(l, r)),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                skip_gradient_check=True,
                                ograd_density=ograd_density,
                                verbose=False)

        test_elemwise_binary_op("minimum", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym._internal._minimum(l, r),
                                lambda l, r: np.minimum(l, r),
                                lambda outg, l, r: (outg * le(l, r), outg * gt(l, r)),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                ograd_density=ograd_density,
                                skip_gradient_check=True,
                                verbose=False)

        test_elemwise_binary_op("hypot", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym._internal._hypot(l, r),
                                lambda l, r: np.hypot(l, r),
                                lambda outg, l, r: (
                                    outg * assign_each2(
                                        l, r, lambda a, b: a/np.sqrt(a * a + b * b)),
                                    outg * assign_each2(
                                        l, r, lambda a, b: b/np.sqrt(a * a + b * b))
                                ),
                                lhs_grad_stype, rhs_grad_stype,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                ograd_density=ograd_density,
                                skip_gradient_check=True,
                                verbose=False)

    # Run basic tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for ii in range(1):
            # Run defaults
            check_elemwise_binary_ops('default', 'default', rand_shape_2d(5, 5))

            # Try different densities
            shape = rand_shape_2d(5, 5)
            for lhs_density in [0.0, random.uniform(0, 1), 1.0]:
                for rhs_density in [0.0, random.uniform(0, 1), 1.0]:
                    for ograd_density in [0.0, random.uniform(0, 1), 1.0]:

                        print("lhs_density={}, rhs_density={}, ograd_density={}, shape: {}".format(
                            lhs_density, rhs_density, ograd_density, shape))

                        # Try row_sparse overlaps
                        for force_lr_overlap in [False, True]:
                            for force_grad_overlap in [False, True]:
                                print("  force_lr_overlap={}, force_grad_overlap={}, shape={}".
                                      format(force_lr_overlap, force_grad_overlap, shape))

                                # Back to left-right overlap possiblities
                                check_elemwise_binary_ops('row_sparse', 'row_sparse', shape,
                                                          lhs_grad_stype='row_sparse',
                                                          rhs_grad_stype='row_sparse',
                                                          lhs_density=lhs_density,
                                                          rhs_density=rhs_density,
                                                          force_lr_overlap=force_lr_overlap,
                                                          force_grad_overlap=force_grad_overlap,
                                                          ograd_density=ograd_density)


def test_elemwise_csr_same_zeros():
    # Zeroes
    a = mx.nd.sparse.zeros('csr', (1,1))
    b = mx.nd.elemwise_add(a,a)
    res = a.asnumpy() + a.asnumpy()
    assert_almost_equal(b.asnumpy(), res)


def as_dense(arr):
    if arr.stype != 'default':
        return mx.nd.cast_storage(arr, stype='default')
    else:
        return arr;

# Make sure that 0's look like 0's when we do a comparison
def do_normalize(arr):
    ret = arr.copy()
    idx = np.isclose(arr, -0, rtol=1.e-3, atol=1.e-3, equal_nan=True)
    ret[idx] = 0
    return ret

def check_sparse_mathematical_core(name, stype,
                                   forward_mxnet_call, forward_numpy_call, backward_numpy_call=None,
                                   rhs_arg=None, data_init=9., grad_init=2., output_grad_stype=None,
                                   input_grad_stype=None, force_overlap=False, density=.5,
                                   ograd_density=.5, verbose=False, shuffle_csr_indices=True):
    if verbose is True:
        print("TESTING: " + name)

    data = mx.symbol.Variable('data', stype=stype)

    temp_input_grad_stype = input_grad_stype

    if temp_input_grad_stype is None:
        temp_input_grad_stype = stype

    if rhs_arg is not None:
        if is_scalar(rhs_arg):
            expected_result_type, expected_grad_result_type = \
                get_fw_bw_result_types_with_scalar(forward_numpy_call, stype,
                                                   backward_numpy_call, temp_input_grad_stype)
        else:
            expected_result_type, expected_grad_result_type = \
                get_fw_bw_result_types_2(forward_numpy_call, stype,
                                         backward_numpy_call, temp_input_grad_stype)
    else:
        expected_result_type, expected_grad_result_type = \
            get_fw_bw_result_types(forward_numpy_call, stype,
                                   backward_numpy_call, temp_input_grad_stype)

    if input_grad_stype is not None and input_grad_stype != expected_grad_result_type:
        print("{}: explicit override of deduced input grad type '{}' with '{}'".format(
            name, expected_grad_result_type, input_grad_stype))
        expected_grad_result_type = input_grad_stype

    shape = rand_shape_2d()

    if verbose is True:
        print("Shape: ", shape, "density: ", density, "force_overlap", force_overlap)

    if stype == 'default':
        data_tmp = np.zeros(shape)
        if abs(density) >= 1e-4:
            data_tmp[:] = data_init
        arr_data = mx.nd.array(data_tmp)
    else:
        arr_data = create_sparse_array_zd(
            shape, stype, density=density,
            data_init=data_init,
            shuffle_csr_indices=shuffle_csr_indices,
            rsp_indices=gen_rsp_random_indices(
                shape,
                density=density,
                force_indices=[(shape[0]/2)] if force_overlap is True else None
            )
        )
        data_tmp = arr_data.asnumpy()
        if verbose is True:
            print("arr_data indices", arr_data.indices.asnumpy())

    if verbose is True:
        print("input", data_tmp)

    if backward_numpy_call is None:
        arr_grad = None
    elif expected_grad_result_type == 'default':
        if abs(density) < 1e-4:
            arr_grad = mx.nd.zeros(shape)
        else:
            arr_grad = mx.nd.ones(shape)
    else:
        arr_grad = create_sparse_array_zd(
            shape,
            expected_grad_result_type,
            density=density,
            data_init=1,
            shuffle_csr_indices=shuffle_csr_indices,
            rsp_indices=gen_rsp_random_indices(
                shape,
                density=density,
                force_indices=[(shape[0]/2)] if force_overlap is True else None
            )
        )

    if rhs_arg is not None:
        test = forward_mxnet_call(data, rhs_arg)
    else:
        test = forward_mxnet_call(data)

    args = list()
    args.append(arr_data)

    if arr_grad is not None:
        exe_test = test._bind(default_context(), args=args, args_grad=[arr_grad])
    else:
        exe_test = test._bind(default_context(), args=args)

    exe_test.forward(is_train=True)
    assert exe_test.outputs[0].stype == expected_result_type
    out = exe_test.outputs[0].asnumpy()

    if rhs_arg is not None:
        npout = forward_numpy_call(data_tmp, rhs_arg)
    else:
        npout = forward_numpy_call(data_tmp)

    if verbose is True:
        print("out", out)
        print("npout", npout)

    assert_almost_equal(out, npout, equal_nan=True)

    if backward_numpy_call is not None:
        if output_grad_stype == 'default' or output_grad_stype is None:
            out_grad = mx.nd.empty(shape)
            out_grad[:] = grad_init
        else:
            out_grad = create_sparse_array_zd(
                shape, output_grad_stype,
                density=density,
                data_init=grad_init,
                shuffle_csr_indices=shuffle_csr_indices,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=ograd_density,
                    force_indices=[(shape[0]/2)] if force_overlap is True else None))

        npout_grad = out_grad.asnumpy()

        if verbose is True:
            print("npout_grad", npout_grad)

        if rhs_arg is not None:
            temp = backward_numpy_call(data_tmp, rhs_arg)
        else:
            temp = backward_numpy_call(data_tmp)
        input_grad = npout_grad * temp

        if verbose is True:
            print(arr_grad.asnumpy())
        exe_test.backward(out_grad)
        if verbose is True:
            print(arr_grad.asnumpy())

        assert arr_grad.stype == expected_grad_result_type

        if verbose is True:
            print(name)
            print("arr_grad", arr_grad.asnumpy())
            print("input_grad", input_grad)

        assert_almost_equal(arr_grad, input_grad, equal_nan=True)


@pytest.mark.serial
@pytest.mark.skip(reason='https://github.com/apache/incubator-mxnet/issues/18829')
def test_sparse_mathematical_core():
    def util_sign(a):
        if np.isclose(a, -0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
            return 0
        elif np.isclose(a, 0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
            return 0
        elif a < 0.0:
            return -1
        else:  # a > 0.0:
            return 1

    # Check scalar binary operators
    def check_binary_op_with_scalar(stype,
                                    output_grad_stype=None,
                                    input_grad_stype=None,
                                    density=.5, ograd_density=.5,
                                    force_overlap=False,):
        # mul_scalar
        check_sparse_mathematical_core("mul_scalar", stype,
                                       lambda x, y: x * y,
                                       lambda x, y: x * y,
                                       lambda input, rhs: rhs,
                                       rhs_arg=5.0,
                                       data_init=2, grad_init=3,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       density=density, ograd_density=ograd_density,
                                       force_overlap=force_overlap,
                                       verbose=False)

        # plus_scalar
        check_sparse_mathematical_core("plus_scalar", stype,
                                       lambda x, y: x + y,
                                       lambda x, y: x + y,
                                       lambda input, rhs: 1,
                                       rhs_arg=5.0,
                                       data_init=2, grad_init=3,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       density=density, ograd_density=ograd_density,
                                       force_overlap=force_overlap,
                                       verbose=False)

        # minus_scalar
        check_sparse_mathematical_core("minus_scalar", stype,
                                       lambda x, y: x - y,
                                       lambda x, y: x - y,
                                       lambda input, rhs: 1,
                                       rhs_arg=5.0,
                                       data_init=2, grad_init=3,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       density=density, ograd_density=ograd_density,
                                       force_overlap=force_overlap,
                                       verbose=False)

    # Check many basic unary operators
    def check_mathematical_core(stype, output_grad_stype=None,
                                input_grad_stype=None, force_overlap=False,
                                density=.5, ograd_density=.5):

        # negative
        check_sparse_mathematical_core("negative", stype,
                                       lambda x: mx.sym.sparse.negative(x),
                                       lambda x: np.negative(x),
                                       force_overlap=force_overlap,
                                       density=density,
                                       input_grad_stype=input_grad_stype,
                                       ograd_density=ograd_density)

        # square
        check_sparse_mathematical_core("square", stype,
                                       lambda x: mx.sym.sparse.square(x),
                                       lambda x: np.square(x),
                                       lambda x: 2 * x,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density,
                                       verbose=False)

        # sqrt
        check_sparse_mathematical_core("sqrt", stype,
                                       lambda x: mx.sym.sparse.sqrt(x),
                                       lambda x: np.sqrt(x),
                                       lambda x: 1.0/(2.0 * np.sqrt(x)),
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density,
                                       verbose=False)

        # cbrt
        check_sparse_mathematical_core("cbrt", stype,
                                       lambda x: mx.sym.sparse.cbrt(x),
                                       lambda x: np.cbrt(x),
                                       lambda x: 1.0/(3.0 * np.cbrt(x) * np.cbrt(x)),
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density,
                                       verbose=False)

        # rint
        check_sparse_mathematical_core("rint", stype,
                                       lambda x: mx.sym.sparse.rint(x),
                                       lambda x: np.rint(x),
                                       force_overlap=force_overlap, density=density,
                                       input_grad_stype=input_grad_stype,
                                       ograd_density=ograd_density)

        # fix
        check_sparse_mathematical_core("fix", stype,
                                       lambda x: mx.sym.sparse.fix(x),
                                       lambda x: np.fix(x),
                                       force_overlap=force_overlap, density=density,
                                       input_grad_stype=input_grad_stype,
                                       ograd_density=ograd_density)

        # floor
        check_sparse_mathematical_core("floor", stype, lambda x: mx.sym.sparse.floor(x),
                                       lambda x: np.floor(x),
                                       force_overlap=force_overlap,
                                       input_grad_stype=input_grad_stype,
                                       density=density, ograd_density=ograd_density)

        # ceil
        check_sparse_mathematical_core("ceil", stype,
                                       lambda x: mx.sym.sparse.ceil(x),
                                       lambda x: np.ceil(x),
                                       force_overlap=force_overlap,
                                       input_grad_stype=input_grad_stype,
                                       density=density, ograd_density=ograd_density)

        # round
        check_sparse_mathematical_core("round", stype,
                                       lambda x: mx.sym.sparse.round(x),
                                       lambda x: np.round(x),
                                       force_overlap=force_overlap,
                                       input_grad_stype=input_grad_stype,
                                       density=density, ograd_density=ograd_density)

        # trunc
        check_sparse_mathematical_core("trunc", stype,
                                       lambda x: mx.sym.sparse.trunc(x),
                                       lambda x: np.trunc(x),
                                       force_overlap=force_overlap,
                                       input_grad_stype=input_grad_stype,
                                       density=density, ograd_density=ograd_density)

        # sign
        check_sparse_mathematical_core("sign", stype,
                                       lambda x: mx.sym.sparse.sign(x),
                                       lambda x: np.sign(x),
                                       lambda x: np.zeros(x.shape),
                                       output_grad_stype=output_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        # log1p
        check_sparse_mathematical_core("log1p", stype,
                                       lambda x: mx.sym.sparse.log1p(x),
                                       lambda x: np.log1p(x),
                                       lambda x: 1. / (1.0 + x),
                                       data_init=0.5, grad_init=0.5,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap, density=density,
                                       ograd_density=ograd_density)

        # expm1
        check_sparse_mathematical_core("expm1", stype,
                                        lambda x: mx.sym.sparse.expm1(x),
                                        lambda x: np.expm1(x),
                                        lambda x: np.exp(x),
                                        data_init=0.5, grad_init=0.5,
                                        output_grad_stype=output_grad_stype,
                                        input_grad_stype=input_grad_stype,
                                        force_overlap=force_overlap, density=density,
                                        ograd_density=ograd_density)

        # sin
        check_sparse_mathematical_core("sin", stype,
                                       lambda x: mx.sym.sparse.sin(x),
                                       lambda x: np.sin(x),
                                       lambda x: np.cos(x),
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        # tan
        check_sparse_mathematical_core("tan", stype,
                                       lambda x: mx.sym.sparse.tan(x),
                                       lambda x: np.tan(x),
                                       lambda x: np.tan(x) ** 2 + 1,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       density=density,
                                       ograd_density=ograd_density)

        # arcsin
        check_sparse_mathematical_core("arcsin", stype,
                                       lambda x: mx.sym.sparse.arcsin(x),
                                       lambda x: np.arcsin(x),
                                       lambda x: 1. / (1. - x ** 2) ** (1. / 2.),
                                       data_init=0.5, grad_init=0.5,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        # arctan
        check_sparse_mathematical_core("arctan", stype,
                                       lambda x: mx.sym.sparse.arctan(x),
                                       lambda x: np.arctan(x),
                                       lambda x: 1. / (x ** 2. + 1.),
                                       data_init=0.5, grad_init=0.5,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        # degrees
        check_sparse_mathematical_core("degrees", stype,
                                       lambda x: mx.sym.sparse.degrees(x),
                                       lambda x: np.degrees(x),
                                       lambda x: assign_each(x, lambda a: 180./np.pi),
                                       data_init=0.5, grad_init=0.5,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        # radians
        check_sparse_mathematical_core("radians", stype,
                                       lambda x: mx.sym.sparse.radians(x),
                                       lambda x: np.radians(x),
                                       lambda x: assign_each(x, lambda a: np.pi / 180.),
                                       data_init=0.6, grad_init=1,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        # sinh
        check_sparse_mathematical_core("sinh", stype,
                                       lambda x: mx.sym.sparse.sinh(x),
                                       lambda x: np.sinh(x),
                                       lambda x: np.cosh(x),
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        # tanh
        check_sparse_mathematical_core("tanh", stype,
                                       lambda x: mx.sym.sparse.tanh(x),
                                       lambda x: np.tanh(x),
                                       lambda x: 1. - np.tanh(x) ** 2,
                                       data_init=0.5, grad_init=1,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap, density=density,
                                       ograd_density=ograd_density)

        # arcsinh
        check_sparse_mathematical_core("arcsinh", stype,
                                       lambda x: mx.sym.sparse.arcsinh(x),
                                       lambda x: np.arcsinh(x),
                                       lambda x: 1./(x**2 + 1.)**(1./2.),
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap, density=density,
                                       ograd_density=ograd_density)

        # arctanh
        check_sparse_mathematical_core("arctanh", stype,
                                       lambda x: mx.sym.sparse.arctanh(x),
                                       lambda x: np.arctanh(x),
                                       lambda x: -1./(x**2 - 1.),
                                       data_init=0.5,
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap, density=density,
                                       ograd_density=ograd_density)

        # abs
        check_sparse_mathematical_core("abs", stype,
                                       lambda x: mx.sym.sparse.abs(x),
                                       lambda x: np.abs(x),
                                       lambda x: assign_each(x, function=util_sign),
                                       output_grad_stype=output_grad_stype,
                                       input_grad_stype=input_grad_stype,
                                       force_overlap=force_overlap,
                                       density=density, ograd_density=ograd_density)

        if stype != "csr":
            # rsqrt
            check_sparse_mathematical_core("rsqrt", stype,
                                           lambda x: mx.sym.sparse.rsqrt(x),
                                           lambda x: 1 / np.sqrt(x),
                                           lambda x: -(1.0 / (2.0 * x * np.sqrt(x))),
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap,
                                           density=density, ograd_density=ograd_density)

            # cos
            check_sparse_mathematical_core("cos", stype,
                                           lambda x: mx.sym.sparse.cos(x),
                                           lambda x: np.cos(x),
                                           lambda x: -np.sin(x),
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap,
                                           density=density, ograd_density=ograd_density)

            # arccos
            check_sparse_mathematical_core("arccos", stype,
                                           lambda x: mx.sym.sparse.arccos(x),
                                           lambda x: np.arccos(x),
                                           lambda x: -1. / (1. - x ** 2.) ** (1. / 2.),
                                           data_init=0.5, grad_init=0.5,
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap, density=density,
                                           ograd_density=ograd_density)

            # cosh
            check_sparse_mathematical_core("cosh", stype,
                                           lambda x: mx.sym.sparse.cosh(x),
                                           lambda x: np.cosh(x),
                                           lambda x: np.sinh(x),
                                           data_init=5, grad_init=5,
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap,
                                           density=density, ograd_density=ograd_density)

            # arccosh
            check_sparse_mathematical_core("arccosh", stype,
                                           lambda x: mx.sym.sparse.arccosh(x),
                                           lambda x: np.arccosh(x),
                                           lambda x: 1./(x**2 - 1.)**(1./2.),
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap, density=density,
                                           ograd_density=ograd_density)

            # log10
            check_sparse_mathematical_core("log10", stype,
                                           lambda x: mx.sym.sparse.log10(x),
                                           lambda x: np.log10(x),
                                           lambda x: 1. / (x * np.log(10.)),
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap, density=density,
                                           ograd_density=ograd_density)

            # log2
            check_sparse_mathematical_core("log2", stype,
                                           lambda x: mx.sym.sparse.log2(x),
                                           lambda x: np.log2(x),
                                           lambda x: 1. / (x * np.log(2.)),
                                           output_grad_stype=output_grad_stype,
                                           input_grad_stype=input_grad_stype,
                                           force_overlap=force_overlap, density=density,
                                           ograd_density=ograd_density)


            try:
                from scipy import special as scipy_special
                # On scipy v1.0, psi([0, -1, -2, -3, ...]) = [ inf, inf, inf, inf, ...]
                # On scipy v1.1, psi([0, -1, -2, -3, ...]) = [-inf, nan, nan, nan, ...]
                # Map the behavior of v1.1 psi() to that of v1.0 for ints <= 0 for consistency
                scipy_psi = np.vectorize(lambda x: np.inf if float(x).is_integer() and x <= 0 else
                                         scipy_special.psi(x))
                # gamma
                check_sparse_mathematical_core("gamma", stype,
                                               lambda x: mx.sym.sparse.gamma(x),
                                               lambda x: scipy_special.gamma(x),
                                               lambda x: scipy_special.gamma(x) * scipy_psi(x),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)
                # gammaln
                check_sparse_mathematical_core("gammaln", stype,
                                               lambda x: mx.sym.sparse.gammaln(x),
                                               lambda x: scipy_special.gammaln(x),
                                               lambda x: scipy_psi(x),
                                               output_grad_stype=output_grad_stype,
                                               input_grad_stype=input_grad_stype,
                                               force_overlap=force_overlap,
                                               density=density, ograd_density=ograd_density)

            except ImportError:
                print("Could not import scipy. Skipping unit tests for special functions")

    for i in range(1):
        print("pass", i)
        for density in [0.0, random.uniform(0, 1), 1.0]:
            for ograd_density in [0.0, random.uniform(0, 1), 1.0]:
                for force_overlap in [False, True]:
                    print("{}, {}, {}".format(density, ograd_density, force_overlap))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # Check unary ops (unary fwd, binary bwd)
                        check_mathematical_core('default', force_overlap=force_overlap,
                                                density=density, ograd_density=ograd_density)
                        check_mathematical_core('row_sparse', force_overlap=force_overlap,
                                                density=density, ograd_density=ograd_density)
                        check_mathematical_core('row_sparse', output_grad_stype='default',
                                                force_overlap=force_overlap,
                                                density=density, ograd_density=ograd_density)
                        check_mathematical_core('row_sparse', output_grad_stype='row_sparse',
                                                force_overlap=force_overlap,
                                                density=density, ograd_density=ograd_density)
                        check_mathematical_core('csr', output_grad_stype='default',
                                                force_overlap=force_overlap,
                                                density=density, ograd_density=ograd_density)
                        check_mathematical_core('csr', output_grad_stype='csr',
                                                force_overlap=force_overlap,
                                                density=density, ograd_density=ograd_density)

                        # Check binary with scalar ops
                        check_binary_op_with_scalar('default',
                                                    density=density,
                                                    ograd_density=ograd_density,
                                                    force_overlap=force_overlap)
                        check_binary_op_with_scalar('row_sparse',
                                                    density=density,
                                                    ograd_density=ograd_density,
                                                    force_overlap=force_overlap)
                        check_binary_op_with_scalar('row_sparse', output_grad_stype='default',
                                                    density=density,
                                                    ograd_density=ograd_density,
                                                    force_overlap=force_overlap)
                        check_binary_op_with_scalar('row_sparse',
                                                    output_grad_stype='row_sparse',
                                                    density=density, ograd_density=ograd_density,
                                                    force_overlap=force_overlap)
                        check_binary_op_with_scalar('csr',
                                                    output_grad_stype='csr',
                                                    input_grad_stype='default',
                                                    density=density,
                                                    ograd_density=ograd_density,
                                                    force_overlap=force_overlap)
                        check_binary_op_with_scalar('csr',
                                                    output_grad_stype='csr',
                                                    input_grad_stype='csr',
                                                    density=density,
                                                    ograd_density=ograd_density,
                                                    force_overlap=force_overlap)
                        check_binary_op_with_scalar('csr',
                                                    output_grad_stype='default',
                                                    density=density,
                                                    ograd_density=ograd_density,
                                                    force_overlap=force_overlap)



@pytest.mark.serial
def test_elemwise_add_ex():
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

    shapes = [rand_shape_2d(), rand_shape_3d()]
    for shape in shapes:
        check_elemwise_add_ex('default', 'default', shape)
        check_elemwise_add_ex('row_sparse', 'row_sparse', shape,
                              lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse')


@pytest.mark.serial
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

    density = [1.00, 0.50, 0.01]
    for d in density:
        shape_2d = rand_shape_2d()
        shape_3d = rand_shape_3d()
        check_cast_storage(shape_2d, d, 'csr', 'default')
        check_cast_storage(shape_2d, d, 'default', 'csr')
        check_cast_storage(shape_2d, d, 'csr', 'csr')
        check_cast_storage(shape_2d, d, 'row_sparse', 'default')
        check_cast_storage(shape_2d, d, 'default', 'row_sparse')
        check_cast_storage(shape_2d, d, 'row_sparse', 'row_sparse')
        check_cast_storage(shape_3d, d, 'row_sparse', 'default')
        check_cast_storage(shape_3d, d, 'default', 'row_sparse')
        check_cast_storage(shape_3d, d, 'row_sparse', 'row_sparse')
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
            # check race condition in block kernel
            check_cast_storage((200, 128 * 2 + 1), d, 'default', 'csr',
                               check_numeric_grad=False)
            # test gpu thread kernel
            check_cast_storage((dim0, rnd.randint(  1,   32)), d, 'default', 'row_sparse')
            # test gpu warp   kernel
            check_cast_storage((dim0, rnd.randint( 32,  512)), d, 'default', 'row_sparse')
            # test gpu block  kernel
            check_cast_storage((dim0, rnd.randint(512, 1024)), d, 'default', 'row_sparse',
                               check_numeric_grad=False)


@pytest.mark.serial
def test_sparse_dot():
    def test_infer_forward_stype(lhs_shape, rhs_shape, lhs_density, rhs_density, trans_a, trans_b):
        all_stypes = ["default", "csr", "row_sparse"]
        lhs_nd = rand_ndarray(lhs_shape, 'default', density=lhs_density)
        rhs_nd = rand_ndarray(rhs_shape, 'default', density=rhs_density)
        out_nd = mx.nd.dot(lhs_nd, rhs_nd, transpose_a=trans_a, transpose_b=trans_b)
        out_np = out_nd.asnumpy()
        for lhs_stype in all_stypes:
            for rhs_stype in all_stypes:
                for forward_stype in all_stypes:
                    lhs = lhs_nd.tostype(lhs_stype)
                    rhs = rhs_nd.tostype(rhs_stype)
                    out = mx.nd.dot(lhs, rhs, forward_stype=forward_stype,
                                    transpose_a=trans_a, transpose_b=trans_b)
                    assert_almost_equal(out.tostype('default').asnumpy(), out_np, rtol=1e-3, atol=1e-4)
                    lhs_var = mx.symbol.Variable('lhs', stype=lhs_stype)
                    rhs_var = mx.symbol.Variable('rhs', stype=rhs_stype)
                    out = mx.symbol.sparse.dot(lhs_var, rhs_var,
                                               forward_stype=forward_stype,
                                               transpose_a=trans_a, transpose_b=trans_b)
                    location = {'lhs': lhs, 'rhs': rhs}
                    check_symbolic_forward(out, location, [out_np], rtol=1e-3, atol=1e-4)
    def test_dot_csr(lhs_shape, rhs_shape, rhs_stype, trans_lhs, lhs_density, rhs_density):
        lhs_nd = rand_ndarray(lhs_shape, 'csr', density=lhs_density, shuffle_csr_indices=False)
        lhs_dns = lhs_nd.tostype('default')
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_density)
        rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.tostype('default')

        out = mx.nd.dot(lhs_nd, rhs_nd, transpose_a=trans_lhs)
        out_dns = mx.nd.dot(lhs_dns, rhs_dns, transpose_a=trans_lhs)
        out_np = out_dns.asnumpy()
        assert_almost_equal(out.asnumpy(), out_np, rtol=1e-3, atol=1e-5)

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

    def test_dot_dns_csr(lhs_shape, rhs_shape, lhs_density, rhs_density, trans_lhs=False, trans_rhs=False):
        lhs_nd = rand_ndarray(lhs_shape, stype='default', density=lhs_density)
        rhs_nd = rand_ndarray(rhs_shape, stype='csr', density=rhs_density)
        rhs_dns = rhs_nd.tostype('default')

        if default_context() == mx.cpu():
            forward_stype = 'csr'
        else:
            forward_stype = 'default'
        out = mx.nd.sparse.dot(lhs_nd, rhs_nd, transpose_a=trans_lhs, transpose_b=trans_rhs, forward_stype=forward_stype)
        out_dns = mx.nd.dot(lhs_nd, rhs_dns, transpose_a=trans_lhs, transpose_b=trans_rhs, forward_stype=forward_stype)
        out_np = out_dns.asnumpy()
        assert_almost_equal(out.asnumpy(), out_np, rtol=1e-3, atol=1e-5)

        # test symbolic forward
        lhs = mx.symbol.Variable('lhs', stype='default')
        rhs = mx.symbol.Variable('rhs', stype='csr')
        out = mx.symbol.sparse.dot(lhs, rhs, transpose_a=trans_lhs, transpose_b=trans_rhs, forward_stype=forward_stype)
        location = {'lhs': lhs_nd, 'rhs': rhs_nd}
        check_symbolic_forward(out, location, [out_np], rtol=1e-3, atol=1e-4)

        if default_context() == mx.cpu():
            # test symbolic backward
            backward_trans = not trans_lhs
            rhs_backward_grad = mx.nd.dot(lhs_nd, out_dns, transpose_a=backward_trans).asnumpy()
            if trans_rhs is True:
                rhs_backward_grad = rhs_backward_grad.T
            expected = {'rhs': rhs_backward_grad}
            check_symbolic_backward(out, location, [out_np], expected,
                                    grad_req={'lhs': 'null', 'rhs': 'write'},
                                    rtol=1e-3, atol=1e-4)
        else:
            transpose_b = not trans_rhs
            lhs_backward_grad = mx.nd.dot(out_dns, rhs_dns, transpose_b=transpose_b)
            expected = {'lhs': lhs_backward_grad.asnumpy()}
            check_symbolic_backward(out, location, [out_np], expected,
                                    grad_req={'lhs': 'write', 'rhs': 'null'},
                                    rtol=1e-3, atol=1e-4)

    def test_sparse_dot_zero_output(lhs_shape, trans_lhs, rhs_num_cols):
        """Test for nnr_out = 0. Before the fix, the test would fail."""
        lhs = mx.nd.zeros(lhs_shape)
        irow = np.random.randint(0, lhs_shape[0])
        icol = np.random.randint(0, lhs_shape[1])
        lhs[irow, icol] = 1.0
        if trans_lhs:
            rhs = rand_ndarray(shape=(lhs_shape[0], rhs_num_cols), stype='default')
            rhs[irow, :] = 0
        else:
            rhs = rand_ndarray(shape=(lhs_shape[1], rhs_num_cols), stype='default')
            rhs[icol, :] = 0
        dns_out = mx.nd.dot(lhs, rhs, transpose_a=trans_lhs)
        assert mx.nd.sum(mx.nd.abs(dns_out)).asscalar() == 0
        sps_out = mx.nd.sparse.dot(lhs.tostype('csr'), rhs.tostype('row_sparse'), transpose_a=trans_lhs)
        assert same(dns_out.asnumpy(), sps_out.asnumpy())

    density = [1.00, 0.5, 0.01]
    for lhs_d in density:
        lhs_shape = rand_shape_2d(50, 200)
        rhs_d = 1
        test_dot_csr(lhs_shape, (lhs_shape[1], 1), 'default', False, lhs_d, rhs_d)  # test gpu SpMV
        test_dot_csr(lhs_shape, (lhs_shape[0], 1), 'default', True,  lhs_d, rhs_d)  # (vector kernel)
        test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(5, 10)), 'default', False, lhs_d, rhs_d)  # test gpu SpMM
        test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(5, 10)), 'default', True, lhs_d, rhs_d)  # (scalar kernel)
        test_dot_dns_csr(lhs_shape, (lhs_shape[1], rnd.randint(50, 200)), lhs_d, lhs_d)
        test_dot_dns_csr(lhs_shape, (rnd.randint(50, 200), lhs_shape[1]), lhs_d, lhs_d, trans_rhs=True)
        for rhs_d in density:
            test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'row_sparse', False, lhs_d, rhs_d)
            test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'row_sparse', True, lhs_d, rhs_d)
            test_infer_forward_stype(lhs_shape, (lhs_shape[1], rnd.randint(10, 20)),
                                     lhs_d, rhs_d, False, False)
            test_infer_forward_stype(lhs_shape, (rnd.randint(10, 20), lhs_shape[1]),
                                     lhs_d, rhs_d, False, True)
            test_infer_forward_stype(lhs_shape, (lhs_shape[0], rnd.randint(10, 20)),
                                     lhs_d, rhs_d, True, False)
            test_infer_forward_stype(lhs_shape, (rnd.randint(10, 20), lhs_shape[0]),
                                     lhs_d, rhs_d, True, True)

    test_sparse_dot_zero_output(rand_shape_2d(50, 200), False, 40)
    test_sparse_dot_zero_output(rand_shape_2d(50, 200), True, 40)

@pytest.mark.serial
def test_sparse_dot_determinism():
    def check_dot_determinism(lhs_stype, rhs_stype, lhs_density, rhs_density, transpose_a, transpose_b, forward_stype):
        lhs_row = rnd.randint(50, 100)
        lhs_col = rnd.randint(50, 100)
        if transpose_a:
            if transpose_b:
                rhs_shape = (rnd.randint(50, 100), lhs_row)
            else:
                rhs_shape = (lhs_row, rnd.randint(50, 100))
        else:
            if transpose_b:
                rhs_shape = (rnd.randint(50, 100), lhs_col)
            else:
                rhs_shape = (lhs_col, rnd.randint(50, 100))
        lhs_shape = (lhs_row, lhs_col)
        lhs = rand_ndarray(lhs_shape, lhs_stype, density=lhs_density)
        rhs = rand_ndarray(rhs_shape, rhs_stype, density=rhs_density)
        res1 = mx.nd.sparse.dot(lhs, rhs, transpose_a=transpose_a, transpose_b=transpose_b, forward_stype=forward_stype)
        res2 = mx.nd.sparse.dot(lhs, rhs, transpose_a=transpose_a, transpose_b=transpose_b, forward_stype=forward_stype)
        assert_almost_equal(res1.asnumpy(), res2.asnumpy(), rtol=0.0, atol=0.0)

    check_dot_determinism('csr', 'default', 0.1, 1.0, True, False, 'row_sparse')
    forward_stype = 'csr' if default_context() == mx.cpu() else 'default'
    check_dot_determinism('default', 'csr', 1.0, 0.1, False, False, forward_stype)
    check_dot_determinism('default', 'csr', 1.0, 0.1, False, True, forward_stype)
    check_dot_determinism('csr', 'default', 0.1, 1.0, True, False, 'default')


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


@pytest.mark.serial
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
    densities = [0.01, 0.5, 1.0]
    index_types = [np.float32, np.int32, np.int64]
    for density in densities:
        for itype in index_types:
            check_sparse_retain(shape, density, itype)
            check_sparse_retain(shape_3d, density, itype)


def test_sparse_unary_with_numerics():
    def check_sparse_simple(name, stype, mxnet_func, forward_numpy_call,
                            backward_numpy_call, output_grad_stype=None,
                            backward_is_use_output=False):
        if output_grad_stype is None:
            output_grad_stype = stype

        expected_result_type, expected_grad_result_type = \
            get_fw_bw_result_types_2(forward_numpy_call, stype, backward_numpy_call, output_grad_stype)
        if backward_is_use_output is True:
            expected_grad_result_type = expected_result_type

        shape = (3, 4)
        data = mx.symbol.Variable("data")

        grad_stypes = {'data' : expected_grad_result_type}

        y = mxnet_func(data)
        if stype == 'default':
            xa = np.random.uniform(low=-1.0, high=1.0, size=shape)
            xa_np = xa
        else:
            xa = create_sparse_array(shape, stype, data_init=None, rsp_indices=[1],
                                     modifier_func=lambda a: a - 0.5,
                                     shuffle_csr_indices=True)
            xa_np = xa.asnumpy()

        if output_grad_stype != 'default':
            out_grad = create_sparse_array(shape, output_grad_stype, data_init=None,
                                           rsp_indices=[1, 2],
                                           modifier_func=lambda a: a - 0.5,
                                           shuffle_csr_indices=True)
            out_grad_np = out_grad.asnumpy()
        else:
            out_grad_np = np.ones(xa.shape)
            out_grad = mx.nd.array(out_grad_np)

        output_np = forward_numpy_call(xa_np)
        input_grad_np = backward_numpy_call(output_np, out_grad_np)

        outputs = check_symbolic_forward(y, [xa], [output_np])
        output = outputs[0]

        assert output.stype == expected_result_type

        input_grad_dict = check_symbolic_backward(y, location=[xa], out_grads=[out_grad],
                                                  expected=[input_grad_np],
                                                  grad_stypes=grad_stypes)
        inp_grad = input_grad_dict["data"]

        assert inp_grad.stype == expected_grad_result_type

    def check_sparse_function(name, mxnet_func, forward_numpy_call, backward_numpy_call,
                              backward_is_use_output=False):
        check_sparse_simple(name, 'default', mxnet_func, forward_numpy_call, backward_numpy_call)
        for output_grad_stype in [None, "row_sparse", "default"]:
            check_sparse_simple(name, 'row_sparse', mxnet_func, forward_numpy_call, backward_numpy_call,
                                output_grad_stype=output_grad_stype,
                                backward_is_use_output=backward_is_use_output)

        for output_grad_stype in [None, "csr", "default"]:
            check_sparse_simple(name, 'csr', mxnet_func, forward_numpy_call, backward_numpy_call,
                                output_grad_stype=output_grad_stype,
                                backward_is_use_output=backward_is_use_output)

    check_sparse_function('relu',
                          lambda x: mx.sym.relu(x),
                          lambda x: np.maximum(x, 0.0),
                          lambda output, outg: outg * assign_each(output, lambda x: x > 0.0), backward_is_use_output=True)

    check_sparse_function('sigmoid',
                          lambda x: mx.sym.sigmoid(x),
                          lambda x: np.divide(1.0, (1.0 + np.exp(-x))),
                          lambda output, outg: outg * assign_each(output, lambda x: x * (1.0 - x)),
                          backward_is_use_output=True)


@pytest.mark.serial
def test_sparse_nd_zeros():
    def check_sparse_nd_zeros(stype, shape):
        zero = mx.nd.zeros(shape)
        sparse_zero = mx.nd.zeros(shape=shape, stype=stype)
        assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros('row_sparse', shape)
    check_sparse_nd_zeros('csr', shape)
    check_sparse_nd_zeros('default', shape)


@pytest.mark.serial
def test_sparse_nd_zeros_like():
    def check_sparse_nd_zeros_like(stype, shape):
        zero = mx.nd.zeros(shape, stype=stype)
        zero_like = mx.nd.sparse.zeros_like(zero)
        assert_almost_equal(zero.asnumpy(), zero_like.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros_like('row_sparse', shape)
    check_sparse_nd_zeros_like('csr', shape)


@pytest.mark.serial
def test_sparse_axis_operations():
    def test_variations(func_name):
        dim0 = 30
        dim1 = 100
        axes = [0, 1]
        densities = [0, 0.5, 1]
        for density in densities:
            shape = rand_shape_2d(dim0, dim1)
            csr_array = rand_ndarray(shape=shape, stype='csr', density=density)
            dns = csr_array.tostype('default')
            for axis in axes:
                ret = func_name(csr_array, axis=axis)
                assert ret.stype == 'default'
                ret_expected = func_name(dns, axis=axis)
                assert_almost_equal(ret.asnumpy(), ret_expected.asnumpy())

    def test_fallback(func_name, axis=0, keepdims=True, exclude=True):
        dim0 = 30
        dim1 = 100
        shape = rand_shape_2d(dim0, dim1)
        csr_array = rand_ndarray(shape=shape, stype='csr', density=0.01)
        ret= func_name(csr_array, axis=axis, keepdims=keepdims,
                       exclude=exclude)

    test_variations(mx.nd.sum)
    test_fallback(mx.nd.sum, axis=0, keepdims=True, exclude=True)
    test_variations(mx.nd.mean)
    test_fallback(mx.nd.mean, axis=0, keepdims=True, exclude=True)


@pytest.mark.serial
def test_sparse_square_sum():
    dim0 = 30
    dim1 = 30
    axes = [0, 1]
    keepdims = [False, True]
    densities = [0, 0.01, 0.2, 0.5, 1.0]
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
                assert_almost_equal(ret.asnumpy(), ret_expected.asnumpy())

                rsp_data = mx.sym.Variable('data', stype='row_sparse')
                test = mx.symbol._internal._square_sum(rsp_data, axis=axis, keepdims=keepdim)

                # check symbolic backward since ograd can be an rsp
                # and cannot be checked through check_numeric_gradient
                # because it will add a loss layer as the output layer
                # which makes ograd of the square_sum dense
                if axis == 1 and keepdim:
                    dns_data = mx.sym.Variable('data')
                    baseline = mx.sym.sum(mx.sym.square(dns_data), axis=axis, keepdims=keepdim)
                    igrad_expected = mx.nd.empty(dns.shape)
                    baseline_exec = baseline._bind(default_context(), args=[dns],
                                                  args_grad=[igrad_expected])
                    baseline_exec.forward(is_train=True)
                    baseline_exec.backward([ret_expected])
                    # check backward when ograd is row sparse
                    check_symbolic_backward(test, [rsp], [ret_expected.tostype('row_sparse')],
                                            [igrad_expected.asnumpy()], grad_stypes={'data': 'row_sparse'})

                    # check backward when ograd is dense
                    # the stype of output of the square_sum is deteremined in symbol binding stage.
                    # The ograd stype of the last layer is the same as the output stype of the last layer.
                    # Need to add one more layer after square_sum to trigger the kernel for ograd
                    # with default stype in square_sum op.
                    baseline1 = baseline + 1
                    baseline_exec1 = baseline1._bind(default_context(), args=[dns],
                                                    args_grad=[igrad_expected])
                    baseline_exec1.forward(is_train=True)
                    baseline_exec1.backward([ret_expected])
                    test1 = test + 1
                    check_symbolic_backward(test1, [rsp], [ret_expected], [igrad_expected.asnumpy()],
                                            grad_stypes={'data': 'row_sparse'})

                # check numeric gradient
                check_numeric_gradient(test, [rsp], grad_stype_dict={'data': 'row_sparse'},
                                       atol=1e-2, rtol=0.1)


@pytest.mark.serial
@pytest.mark.flaky
def test_sparse_storage_fallback():
    """ test operators which don't implement FComputeEx or FStatefulComputeEx """
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
        x = x - np.max(x, axis=axis, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=axis, keepdims=True)
        return x

    def check_concat(shape, lhs_stype, rhs_stype):
        x = mx.symbol.Variable('x', stype=lhs_stype)
        w = mx.symbol.Variable('w', stype=rhs_stype)
        test = mx.sym.Concat(x, w)
        x_nd = rand_ndarray(shape, lhs_stype)
        w_nd = rand_ndarray(shape, rhs_stype)
        location = {'x': x_nd, 'w': w_nd}
        check_numeric_gradient(test, location)

    def check_operator_with_temp_resource(shape, stype):
        x = mx.symbol.Variable('x', stype=stype)
        test = mx.sym.sum(x)
        x_nd = rand_ndarray(shape, stype)
        location = {'x': x_nd}
        check_numeric_gradient(test, location)

    shape = rand_shape_2d()
    stypes = ['default', 'csr', 'row_sparse']
    for lhs in stypes:
        check_operator_with_temp_resource(shape, lhs)
        for rhs in stypes:
            check_broadcast_add(shape, lhs, rhs)
            check_concat(shape, lhs, rhs)


@pytest.mark.serial
def test_sparse_elementwise_sum():
    def check_sparse_elementwise_sum_with_shape(stypes, shape, n):
        # forward
        inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
        out = mx.symbol.sparse.add_n(*inputs, name='esum')
        arr = []
        arr_grad = [mx.nd.empty(shape, stype=stype) for stype in stypes]
        densities = [0, 0.01, 0.5, 1.0]
        for stype in stypes:
            arr.append(rand_ndarray(shape, stype, densities[np.random.randint(0, len(densities))]))

        exec1 = out._bind(default_context(),
                         args=arr,
                         args_grad=arr_grad)
        exec1.forward(is_train=True)
        out1 = exec1.outputs[0].asnumpy()
        out = sum(a.asnumpy() for a in arr)
        assert_almost_equal(out, out1, atol=1e-5)

        out_grad = mx.nd.empty(shape)
        out_grad[:] = np.random.uniform(-10, 10, shape)
        # backward
        exec1.backward([out_grad])
        for a in arr_grad:
            assert_almost_equal(a.asnumpy(), out_grad.asnumpy(), atol=1e-5)

    all_stypes = ['default', 'csr', 'row_sparse']
    for dim in range(2, 4):
        shape = tuple(np.random.randint(5, 10, size=dim))
        rsp_test_cnt = np.random.randint(1, 9)
        check_sparse_elementwise_sum_with_shape(['row_sparse' for i in range(rsp_test_cnt)], shape, rsp_test_cnt)
        if dim is 2:
            check_sparse_elementwise_sum_with_shape(['default', 'csr', 'default'], shape, 3)
            test_len = np.random.randint(5, 10)
            # at least one default type
            stypes = ['default']
            for i in range(test_len):
                pick_side = np.random.randint(2)
                pick_type = np.random.randint(3)
                stypes = ([all_stypes[pick_type]] if pick_side is 0 else []) + stypes + ([all_stypes[pick_type]] if pick_side is 1 else [])
            check_sparse_elementwise_sum_with_shape(stypes, shape, test_len+1)


@pytest.mark.serial
def test_sparse_embedding():
    ''' test sparse embedding operator '''
    def check_sparse_embedding(in_dim, out_dim, batch, densities, sparse_grad):
        target_stype = 'row_sparse' if sparse_grad else 'default'
        # init executor
        data = mx.sym.Variable("data")
        weight = mx.sym.Variable("embed_weight")
        embed = mx.sym.sparse.Embedding(data=data, weight=weight, input_dim=in_dim,
                                        sparse_grad=sparse_grad, output_dim=out_dim, name='embed')
        grad_req = {'data': 'null', 'embed_weight': 'write'}
        args = {'embed_weight': mx.nd.zeros((in_dim, out_dim)), 'data': mx.nd.ones((batch,))}
        weight_grad = mx.nd.zeros((in_dim, out_dim))
        if sparse_grad:
            weight_grad = weight_grad.tostype('row_sparse')
        args_grad = {'embed_weight': weight_grad}
        exe_test = embed._bind(default_context(), args=args, args_grad=args_grad, grad_req=grad_req)
        arg_map = dict(zip(embed.list_arguments(), exe_test.arg_arrays))
        grad_map = dict(zip(embed.list_arguments(), exe_test.grad_arrays))
        # init data
        np_data = np.random.randint(low=0, high=in_dim, size=batch)
        np_onehot = np.zeros((batch, in_dim)).astype(np.float32)
        np_onehot[np.arange(batch), np_data] = 1.0
        arg_map["data"][:] = np_data
        # weight
        weight = arg_map["embed_weight"]
        for density in densities:
            # update weight based on density
            weight[:] = rand_ndarray(weight.shape, 'default', density=density)
            # check forward
            exe_test.forward(is_train=True)
            # init grad
            np_grad = np.random.uniform(-1, 1, exe_test.outputs[0].shape)
            grad = mx.nd.zeros(np_grad.shape)
            grad[:] = np_grad
            assert_almost_equal(exe_test.outputs[0].asnumpy(), np.dot(np_onehot, weight.asnumpy()), atol=1e-4)
            # check backward
            exe_test.backward([grad])
            assert_almost_equal(grad_map["embed_weight"].asnumpy(), np.dot(np_onehot.T, grad.asnumpy()), atol=1e-4)
            # check grad stype
            assert(grad_map["embed_weight"].stype == target_stype)

    densities = [0, 0.5, 1]
    in_dim = 50
    out_dim = 3
    batch = 8
    sparse_grads = [True, False]
    for sparse_grad in sparse_grads:
        check_sparse_embedding(in_dim, out_dim, batch, densities, sparse_grad)

def test_sparse_broadcast_add_sub():
    def check_broadcast_add(mx_lhs, mx_rhs, np_lhs, np_rhs, dtype):
        assert_almost_equal(mx.nd.sparse.add(mx_lhs, mx_rhs).asnumpy(), np.add(np_lhs, np_rhs), atol=1e-4)
    def check_broadcast_sub(mx_lhs, mx_rhs, np_lhs, np_rhs, dtype):
        assert_almost_equal(mx.nd.sparse.subtract(mx_lhs, mx_rhs).asnumpy(), np.subtract(np_lhs, np_rhs), atol=1e-4)
    stype = 'csr'
    shape = rand_shape_2d()
    num_rows = shape[0]
    num_cols = shape[1]
    for density in [0.1 * i for i in range(10)]:
        mx_lhs = rand_ndarray(shape, stype, density)
        np_lhs = mx_lhs.asnumpy()
        mx_rhs_row_2D = rand_ndarray((1, num_cols), 'default')
        mx_rhs_row_1D = mx_rhs_row_2D.reshape((num_cols))
        mx_rhs_col = rand_ndarray((num_rows, 1), 'default')
        mx_rhs_scalar_2D = rand_ndarray((1, 1), 'default')
        mx_rhs_scalar_1D = mx_rhs_scalar_2D.reshape((1, ))
        for mx_rhs in [mx_rhs_row_2D, mx_rhs_row_1D, mx_rhs_col, mx_rhs_scalar_2D, mx_rhs_scalar_1D]:
            np_rhs = mx_rhs.asnumpy()
            check_broadcast_add(mx_lhs, mx_rhs, np_lhs, np_rhs, np.float32)
            check_broadcast_sub(mx_lhs, mx_rhs, np_lhs, np_rhs, np.float32)
            check_broadcast_add(mx_rhs, mx_lhs, np_rhs, np_lhs, np.float32)
            check_broadcast_sub(mx_rhs, mx_lhs, np_rhs, np_lhs, np.float32)

def test_sparse_broadcast_mul_div():
    def check_broadcast_mul(mx_lhs, mx_rhs, np_lhs, np_rhs, dtype):
        assert_almost_equal(mx.nd.sparse.multiply(mx_lhs, mx_rhs).asnumpy(), np.multiply(np_lhs, np_rhs), atol=1e-4)
    def check_broadcast_div(mx_lhs, mx_rhs, np_lhs, np_rhs, dtype):
        assert_almost_equal(mx.nd.sparse.divide(mx_lhs, mx_rhs).asnumpy(), np.divide(np_lhs, np_rhs), atol=1e-4)
    stype = 'csr'
    shape = rand_shape_2d()
    num_rows = shape[0]
    num_cols = shape[1]
    for density in [0.1 * i for i in range(10)]:
        mx_lhs = rand_ndarray(shape, stype, density)
        np_lhs = mx_lhs.asnumpy()
        mx_rhs_row_2D = rand_ndarray((1, num_cols), 'default')
        mx_rhs_row_1D = mx_rhs_row_2D.reshape((num_cols))
        mx_rhs_col = rand_ndarray((num_rows, 1), 'default')
        mx_rhs_scalar_2D = rand_ndarray((1, 1), 'default')
        mx_rhs_scalar_1D = mx_rhs_scalar_2D.reshape((1, ))
        for mx_rhs in [mx_rhs_row_2D, mx_rhs_row_1D, mx_rhs_col, mx_rhs_scalar_2D, mx_rhs_scalar_1D]:
            np_rhs = mx_rhs.asnumpy()
            check_broadcast_mul(mx_lhs, mx_rhs, np_lhs, np_rhs, np.float32)
            check_broadcast_div(mx_lhs, mx_rhs, np_lhs, np_rhs, np.float32)

def test_batchnorm_fallback():
    # same test as test_operator.test_batchnorm_training, but tests fallback logic of batchnorm
    stype = 'row_sparse'
    for shape in [(2, 3), (2, 3, 2, 2)]:
        data_tmp = np.random.normal(-0.1, 0.1, size=shape)
        s = shape[1],
        gamma = np.ones(s)
        beta = np.ones(s)
        gamma[1] = 3
        beta[0] = 3

        rolling_mean = np.random.uniform(size=s)
        rolling_std = np.random.uniform(size=s)

        data = mx.symbol.Variable('data', stype=stype)
        in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(gamma).tostype(stype),
                        mx.nd.array(beta).tostype(stype)]
        mean_std = [mx.nd.array(rolling_mean).tostype(stype), mx.nd.array(rolling_std).tostype(stype)]

        test = mx.symbol.BatchNorm(data, fix_gamma=True)
        assertRaises(MXNetError, check_numeric_gradient, test, in_location, mean_std, numeric_eps=1e-3, rtol=0.16, atol=1e-2)

        test = mx.symbol.BatchNorm(data, fix_gamma=True, use_global_stats=True)
        assertRaises(MXNetError, check_numeric_gradient, test, in_location, mean_std, numeric_eps=1e-3, rtol=0.16, atol=1e-2)

        test = mx.symbol.BatchNorm(data, fix_gamma=False)
        check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-3, rtol=0.16, atol=1e-2)

        test = mx.symbol.BatchNorm(data, fix_gamma=False, use_global_stats=True)
        check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-3, rtol=0.16, atol=1e-2)

        # Test varying channel axis
        dim = len(shape)
        for chaxis in range(-dim, dim):
            chaxis_true = chaxis
            if chaxis < 0:
                chaxis_true = dim + chaxis

            shapex = shape

            channel_count = shapex[chaxis_true]
            data_tmp = np.random.normal(-0.1, 0.1, size=shapex)

            gamma = np.ones(channel_count)
            beta = np.ones(channel_count)
            if channel_count > 1:
                gamma[1] = 3
            beta[0] = 3

            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(gamma).tostype(stype),
                            mx.nd.array(beta).tostype(stype)]

            xrolling_mean = np.random.uniform(size=channel_count)
            xrolling_std = np.random.uniform(size=channel_count)
            xmean_std = [mx.nd.array(xrolling_mean).tostype(stype),
                            mx.nd.array(xrolling_std).tostype(stype)]

            test = mx.symbol.BatchNorm(data, fix_gamma=True, axis=chaxis)
            assertRaises(MXNetError, check_numeric_gradient, test, in_location, xmean_std, numeric_eps=1e-3, rtol=0.2, atol=0.01)

            test = mx.symbol.BatchNorm(data, fix_gamma=True, use_global_stats=True, axis=chaxis)
            assertRaises(MXNetError, check_numeric_gradient, test, in_location, xmean_std, numeric_eps=1e-3, rtol=0.2, atol=0.01)

            test = mx.symbol.BatchNorm(data, fix_gamma=False, axis=chaxis)
            check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-3, rtol=0.2, atol=0.01)

            test = mx.symbol.BatchNorm(data, fix_gamma=False, use_global_stats=True, axis=chaxis)
            check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-3, rtol=0.2, atol=0.01)


@pytest.mark.serial
def test_mkldnn_sparse():
    # This test is trying to create a race condition describedd in
    # https://github.com/apache/incubator-mxnet/issues/10189
    arr = mx.nd.random.uniform(shape=(10, 10, 32, 32))
    weight1 = mx.nd.random.uniform(shape=(10, 10, 3, 3))
    arr = mx.nd.Convolution(data=arr, weight=weight1, no_bias=True, kernel=(3, 3), num_filter=10)

    rs_arr = mx.nd.sparse.row_sparse_array((mx.nd.zeros_like(arr), np.arange(arr.shape[0])))
    weight2 = mx.nd.random.uniform(shape=(10, np.prod(arr.shape[1:4])))
    fc_res = mx.nd.FullyConnected(data=arr, weight=weight2, no_bias=True, num_hidden=10)
    sum_res = mx.nd.elemwise_sub(arr, rs_arr)
    res1 = np.dot(mx.nd.flatten(sum_res).asnumpy(), weight2.asnumpy().T)
    print(res1 - fc_res.asnumpy())
    almost_equal(res1, fc_res.asnumpy())

@pytest.mark.serial
def test_sparse_nd_where():
    def get_forward_expected_output(condition, x, y):
        original_shape = x.shape
        out = np.zeros(original_shape)
        if condition.shape == x.shape:
            for index, c in np.ndenumerate(condition):
                if c != 0:
                    out[index] = x[index]
                else:
                    out[index] = y[index]
        else:
            raise RuntimeError("Invalid condition shape for where op")

        out = out.reshape(original_shape)
        return out

    def get_forward_inputs_same_shape(shape):
        condition_np = np.random.randint(0, 2, np.prod(shape)).reshape(shape)
        x_np = np.random.randint(1, 6, np.prod(shape)).reshape(shape)
        y_np = np.random.randint(7, 11, np.prod(shape)).reshape(shape)
        return condition_np, x_np, y_np

    def get_backward_input(shape):
        return np.random.randint(20, 30, np.prod(shape)).reshape(shape)

    def get_backward_expected_outputs(grad_in, condition):
        shape = grad_in.shape
        grad_cond = np.zeros(condition.shape)
        grad_x = np.empty(shape)
        grad_y = np.empty(shape)

        for index, c in np.ndenumerate(condition):
            if 0 != c:
                grad_x[index] = grad_in[index]
                grad_y[index] = 0
            else:
                grad_x[index] = 0
                grad_y[index] = grad_in[index]

        return grad_cond, grad_x, grad_y

    def test_where_helper(shape):
        condition_np, x_np, y_np = get_forward_inputs_same_shape(shape)

        out_expected = get_forward_expected_output(condition_np, x_np, y_np)

        grad_in_np = get_backward_input(shape)
        grad_expected_cond, grad_expected_x, grad_expected_y \
            = get_backward_expected_outputs(grad_in_np, condition_np)

        condition = mx.sym.Variable('condition', stype='csr')
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        grad_in_mx = mx.nd.array(grad_in_np, dtype=np.int32)
        where_sym = mx.sym.where(condition, x, y)

        cond_nd = mx.nd.array(condition_np)
        args = {'condition': cond_nd.tostype('csr'), 'x': mx.nd.array(x_np),
                'y' : mx.nd.array(y_np)}
        args_grad = {'condition': mx.nd.zeros_like(cond_nd),
                     'x': mx.nd.array(x_np).tostype('csr'), 'y' : mx.nd.array(y_np)}
        # test req='write'
        where_exe_write = where_sym._bind(ctx=default_context(), args=args,
                                         args_grad=args_grad, grad_req='write')

        # test forward req='write'
        outputs = where_exe_write.forward(is_train=True)
        assert same(outputs[0].asnumpy(), out_expected)
        # test backward req='write'
        where_exe_write.backward(grad_in_mx.astype('float32'))
        assert same(where_exe_write.grad_dict['x'].asnumpy(), grad_expected_x)
        assert same(where_exe_write.grad_dict['y'].asnumpy(), grad_expected_y)
        assert same(where_exe_write.grad_dict['condition'].asnumpy(), grad_expected_cond)

        # test req='add'
        x_grad_init = np.random.randint(30, 40, np.prod(shape)).reshape(shape)
        y_grad_init = np.random.randint(40, 50, np.prod(shape)).reshape(shape)
        where_exe_add = where_sym._bind(ctx=default_context(), args=args,
                                       args_grad=args_grad, grad_req='add')
        where_exe_add.grad_dict['x'][:] = x_grad_init
        where_exe_add.grad_dict['y'][:] = y_grad_init
        # test forward req='add'
        outputs = where_exe_add.forward(is_train=True)
        assert same(outputs[0].asnumpy(), out_expected)

    def test_where_numeric_gradient(shape):
        condition = mx.sym.Variable('condition', stype='csr')
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        where_sym = mx.sym.where(condition, x, y)
        condition_np, x_np, y_np = get_forward_inputs_same_shape(shape)
        check_numeric_gradient(where_sym, [condition_np, x_np, y_np], grad_nodes=['x', 'y'])

    test_where_helper((5, 9))
    test_where_numeric_gradient((5, 9))

@pytest.mark.serial
def test_sparse_quadratic_function():
    def f(x, a, b, c):
        return a * x**2 + b * x + c

    def check_sparse_quadratic_function(a, b, c, expected_stype):
      # check forward and compare the result with dense op
      ndim = 2
      shape = rand_shape_nd(ndim, 5)
      data = rand_ndarray(shape=shape, stype='csr')
      data_np = data.asnumpy()
      expected = f(data_np, a, b, c)
      output = mx.nd.contrib.quadratic(data, a=a, b=b, c=c)
      assert(output.stype == expected_stype)
      assert_almost_equal(output.asnumpy(), expected)

    a = np.random.random_sample()
    b = np.random.random_sample()
    check_sparse_quadratic_function(a, b, 0.0, 'csr')
    check_sparse_quadratic_function(a, b, 1.0, 'default')

def test_reshape_backward_fallback():
    """
     out
     |  \
    w_x  x
     /
    w
    in which x is a sparse tensor.
    Due to sparse gradient optimization in sym.dot, grad(w_x) is sparse.
    Though sym.reshape itself does not have sparse version,
    if we somehow make grad(w) sparse as well, e.g.,
        - by setting args_grad in symbol.bind
        - or, we can have out_y = sym.dot(sparse_y, w), then grad(w) will be inferred as sparse
    reshape backward (from w_x to w) needs to understand how to handle sparse inputs.
    """
    ctx = default_context()
    w_shape = (12, 4)
    w_x_shape = (1, 48)
    x_nd = rand_ndarray((4, 1), 'csr')

    w_nd = rand_ndarray(w_shape)

    w_x_nd = w_nd.reshape(w_x_shape)
    out_x_nd = mx.nd.dot(x_nd, w_x_nd)

    w_x_backward_grad = mx.nd.dot(x_nd, out_x_nd, transpose_a=True).asnumpy()
    expected_grad_nd = w_x_backward_grad.reshape(w_shape)

    x = mx.sym.Variable('x', stype='csr')
    w = mx.sym.Variable('w')

    w_x = mx.sym.reshape(w, w_x_shape, name="w_x")
    out = mx.sym.sparse.dot(x, w_x, name='out_x')

    grad_w_nd = rand_ndarray(w_shape, 'row_sparse')
    executor = out._bind(ctx=ctx, args={"x": x_nd, "w": w_nd},
                        args_grad={"w": grad_w_nd})
    executor.forward(is_train=True)
    executor.backward(out_x_nd)

    assert_almost_equal(grad_w_nd.asnumpy(), expected_grad_nd)
