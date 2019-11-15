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


import tvm
import topi
from .. import defop
from ..utils import reduce_axes, assign_by_req


def _compute_sum(itype, otype, ndim, reduce1st_dim, req):
    axes = ([reduce1st_dim, 1 - reduce1st_dim] * ndim)[:ndim]
    a = tvm.placeholder([tvm.var() for _ in range(ndim)], name='a', dtype=itype)
    reduce_output = reduce_axes(a, axes, tvm.sum, otype)
    output_placeholder, final_output = assign_by_req(reduce_output, req)
    s = tvm.create_schedule(final_output.op)
    return s, a, output_placeholder, final_output, [reduce_output, final_output]


@defop(name='sum_cpu', target='cpu', itype=['bool'],
       otype=['float32', 'float64', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       attrs=["reduce1st_dim", "req"])
def _sum_cpu(itype, otype, ndim, reduce1st_dim, req):
    s, a, output_placeholder, final_output, tensor_list = _compute_sum(
        itype, otype, ndim, reduce1st_dim, req)
    for t in tensor_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [a, output_placeholder, final_output]


@defop(name='sum_gpu', target='gpu', itype=['bool'],
       otype=['float32', 'float64', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       attrs=["reduce1st_dim", "req"])
def _sum_gpu(itype, otype, ndim, reduce1st_dim, req):
    s, a, output_placeholder, final_output, tensor_list = _compute_sum(
        itype, otype, ndim, reduce1st_dim, req)
    num_threads = 64
    for t in tensor_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_threads)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [a, output_placeholder, final_output]


def accumulate(X, axis, op):
     def swapaxis(idx, axis1, axis2):
         ret = list(idx)
         if axis1 != axis2:
             ret[axis1], ret[axis2] = ret[axis2], ret[axis1]
         return ret
     ishape = list(X.shape)
     sshape = swapaxis(ishape, 0, axis)
     s_state = tvm.placeholder(sshape, dtype=X.dtype)
     s_init = tvm.compute([1] + sshape[1:], lambda *idx: X[tuple(swapaxis(idx, 0, axis))])
     s_update = tvm.compute(sshape, lambda *idx: op(s_state[(idx[0] - 1, ) + idx[1:]], X[tuple(swapaxis(idx, 0, axis))]))
     s_scan = tvm.scan(s_init, s_update, s_state)
     ret = tvm.compute(ishape, lambda *idx: s_scan[tuple(swapaxis(idx, 0, axis))])
     return ret, [ret], [s_init, s_update]


def compute_cumprod(dtype, ndim, axis):
    ishape = [tvm.var() for _ in range(ndim)]
    X = tvm.placeholder(ishape, name='X', dtype=dtype)
    if ndim == 0:
        tshape = [tvm.var()]
        ret = tvm.compute(tshape, lambda *idx:X[()])
        c_list, s_list = [ret], []
    elif axis is None:
        tshape = [tvm.var()]
        Y = topi.reshape(X, tshape)
        ret, c_list, s_list = accumulate(Y, 0, lambda x, y: x * y)
        c_list.append(Y)
    else:
        ret, c_list, s_list = accumulate(X, axis, lambda x, y: x * y)
    s = tvm.create_schedule(ret.op)
    return s, X, ret, c_list, s_list


@defop(name="cumprod", target="cpu", auto_broadcast=False,
    dtype=["float32", "float64", "uint8", "int8", "int32", "int64"], ndim=list(range(0, 6)), axis=list(range(0, 5)) + [None],
    attrs=["axis"], attrs_valid=lambda **kwargs: kwargs['axis'] is None or kwargs['axis'] < kwargs['ndim'])
def cumprod(dtype, ndim, axis):
    s, X, ret, c_list, s_list = compute_cumprod(dtype, ndim, axis)
    for t in s_list:
        axes = [axis for axis in t.op.axis[1:]]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [X, ret]


@defop(name="cuda_cumprod", target="gpu", auto_broadcast=False,
    dtype=["float32", "float64", "uint8", "int8", "int32", "int64"],
    ndim=list(range(0, 6)), axis=list(range(0, 5)) + [None],
    attrs=["axis"], attrs_valid=lambda **kwargs: kwargs['axis'] is None or kwargs['axis'] < kwargs['ndim'])
def cuda_cumprod(dtype, ndim, axis):
    s, X, ret, c_list, s_list = compute_cumprod(dtype, ndim, axis)
    num_thread = 64
    for (i, t) in enumerate(s_list):
        axes = [axis for axis in t.op.axis[1:]]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    for (i, t) in enumerate(c_list):
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [X, ret]


def kernel_backward_cumprod(out_grad, X, axis):
    def swapaxis(idx, axis1, axis2):
        ret = list(idx)
        if axis1 != axis2:
            ret[axis1], ret[axis2] = ret[axis2], ret[axis1]
        return ret if isinstance(idx, list) else tuple(ret)

    ishape = list(X.shape)
    sshape = swapaxis(ishape, 0, axis) + [ishape[axis]]
    s_state = tvm.placeholder(sshape, dtype=X.dtype)
    s_init = tvm.compute([1] + sshape[1:],
                        lambda *idx: tvm.expr.Select(idx[-1] > 0,
                                                    tvm.const(0, X.dtype),
                                                    tvm.const(1, X.dtype)))
    s_update = tvm.compute(sshape,
                        lambda *idx: tvm.expr.Select(idx[0] < idx[-1],
                                                        tvm.const(0, X.dtype),
                                                        tvm.expr.Select(idx[0] == idx[-1],
                                                                        s_state[(idx[0] - 1, ) + idx[1:-1] + (idx[-1] - 1, )]
                                                                        * X[swapaxis((idx[0] - 1, ) + idx[1:-1], 0, axis)],
                                                                        s_state[(idx[0] - 1, ) + idx[1:]]
                                                                        * X[swapaxis(idx[:-1], 0, axis)])))
    s_scan = tvm.scan(s_init, s_update, s_state)
    A = tvm.compute(sshape, lambda *idx: s_scan[idx] * out_grad[swapaxis(idx[:-1], 0, axis)])
    k = tvm.reduce_axis((0, sshape[0]), name="k")
    if axis != 0:
        ret = tvm.compute(ishape,
                        lambda* idx: tvm.sum(A[(k,) + idx[1: axis] + (idx[0],) + idx[axis + 1:] + (idx[axis],)],
                                            axis=k), name="ret")
    else:
        ret = tvm.compute(ishape,
                        lambda* idx: tvm.sum(A[(k,) + idx[1:] + (idx[0],)],
                                            axis=k), name="ret")
    return ret, [A, ret], [s_init, s_update]


def compute_backward_cumprod(dtype, ndim, axis, req):
    ishape = [tvm.var() for _ in range(ndim)]
    X = tvm.placeholder(ishape, name='X', dtype=dtype)
    if ndim == 0:
        tshape = [tvm.var()]
        out_grad = tvm.placeholder(tshape, name='out_grad', dtype=dtype)
        ret = tvm.compute(ishape, lambda *idx: tvm.const(1, dtype))
        c_list = [ret]
        s_list = []
    elif axis == None:
        tshape = [tvm.var()]
        out_grad = tvm.placeholder(tshape, name='out_grad', dtype=dtype)
        Y = topi.reshape(X, tshape)
        ret, c_list, s_list = kernel_backward_cumprod(out_grad, Y, 0)
        ret = topi.reshape(ret, ishape)
        c_list.extend([Y, ret])
    else:
        out_grad = tvm.placeholder(ishape, name='out_grad', dtype=dtype)
        ret, c_list, s_list = kernel_backward_cumprod(out_grad, X, axis)
    in_grad_old, in_grad = assign_by_req(ret, req)
    c_list.append(in_grad)
    s = tvm.create_schedule(in_grad.op)
    return s, out_grad, X, in_grad_old, in_grad, c_list, s_list


@defop(name="backward_cumprod", target="cpu", auto_broadcast=False,
    dtype=["float32", "float64", "uint8", "int8", "int32", "int64"],
    ndim=list(range(0, 6)), axis=list(range(0, 5)) + [None],
    req=["kWriteTo", "kAddTo"], attrs=["axis", "req"],
    attrs_valid=lambda **kwargs: kwargs['axis'] is None or kwargs['axis'] < kwargs['ndim'])
def backward_cumprod(dtype, ndim, axis, req):
    s, out_grad, X, in_grad_old, in_grad, c_list, s_list = compute_backward_cumprod(dtype, ndim, axis, req)
    for t in s_list:
        axes = [axis for axis in t.op.axis[1:]]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [out_grad, X, in_grad, in_grad_old]


@defop(name="cuda_backward_cumprod", target="gpu", auto_broadcast=False,
    dtype=["float32", "float64", "uint8", "int8", "int32", "int64"],
    ndim=list(range(0, 6)), axis=list(range(0, 5)) + [None],
    req=["kWriteTo", "kAddTo"], attrs=["axis", "req"],
    attrs_valid=lambda **kwargs: kwargs['axis'] is None or kwargs['axis'] < kwargs['ndim'])
def cuda_backward_cumprod(dtype, ndim, axis, req):
    s, out_grad, X, in_grad_old, in_grad, c_list, s_list = compute_backward_cumprod(dtype, ndim, axis, req)
    num_thread = 64
    for t in s_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis[1:]]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    for t in c_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [out_grad, X, in_grad, in_grad_old]
