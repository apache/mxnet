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

# coding: utf-8
import tvm
from .. import defop, AllTypes, RealTypes
from .. import assign_by_req, reduce_axes

def compute_add(dtype, ndim):
    A = tvm.placeholder([tvm.size_var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.size_var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.size_var() for _ in range(ndim)],
                    lambda *index: A[index] + B[index], name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C


@defop(name="vadd", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=[5])
def vadd(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)

    return s, [A, B, C]


@defop(name="cuda_vadd", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=[5])
def vadd_gpu(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    s = tvm.create_schedule(C.op)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]


def compute_backward_vadd(dtype, ndim, reduce1st, req):
    # The backward of broadcast op is basically a reduction on broadcast axes.
    # We label the reduce axes as 1 and other axes as 0, and they form a bit string.
    # Each bit string correponds to a kernel, so the number of kernels is as many as `2^n`
    # To reduce it, the bit string is compressed by combining consecutive 0s or 1s.
    # In this way, the number of bit string (the number of kernels) is reduced to `2 * n`
    # They compressed bit string is stored in `axes`. And `reduce1st` represents the first bit
    # of the compressed bit string. Credit to @junrushao1994 and @yzhliu.
    axes = ([reduce1st, 1 - reduce1st] * ndim)[:ndim]
    X = tvm.placeholder([tvm.size_var() for _ in range(ndim)], name='X', dtype=dtype)
    reducer = tvm.comm_reducer(lambda x, y: x + y,
        lambda t: tvm.const(0, dtype=t), name="sum")
    ret = reduce_axes(X, axes, reducer)
    in_grad_a, in_grad = assign_by_req(ret, req)
    s = tvm.create_schedule(in_grad.op)
    return s, X, in_grad_a, in_grad, [ret, in_grad]


@defop(name="backward_vadd", target="cpu", dtype=AllTypes, 
       ndim=[5], reduce1st=[0, 1],
       req=["kWriteTo", "kAddTo"], attrs=["reduce1st", "req"])
def backward_vadd(dtype, ndim, reduce1st, req):
    s, X, in_grad_a, in_grad, c_list = compute_backward_vadd(dtype, ndim, reduce1st, req)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [X, in_grad_a, in_grad]


@defop(name="cuda_backward_vadd", target="gpu", dtype=["float32", "float64"],
       ndim=[5], reduce1st=[0, 1],
       req=["kWriteTo", "kAddTo"], attrs=["reduce1st", "req"])
def backward_vadd_gpu(dtype, ndim, reduce1st, req):
    s, X, in_grad_a, in_grad, c_list = compute_backward_vadd(dtype, ndim, reduce1st, req)
    num_thread = 64
    for t in c_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [X, in_grad_a, in_grad]


def compute_degandrad(dtype, ndim, n):
    A = tvm.placeholder([tvm.size_var() for _ in range(ndim)], name='A', dtype=dtype)
    import math
    if n == 0:
        B = tvm.compute([tvm.size_var() for _ in range(ndim)],
                        lambda *index: A[index] * tvm.const(math.pi, dtype) / tvm.const(180, dtype), name='B')
    else:
        B = tvm.compute([tvm.size_var() for _ in range(ndim)],
                        lambda *index: A[index] / tvm.const(math.pi, dtype) * tvm.const(180, dtype), name='B')
    s = tvm.create_schedule(B.op)
    return s, A, B


@defop(name="deg2rad", target="cpu", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def deg2rad(dtype, ndim):
    s, A, B = compute_degandrad(dtype, ndim, 0)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    s[B].parallel(fused)
    return s, [A, B]


@defop(name="rad2deg", target="cpu", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def rad2deg(dtype, ndim):
    s, A, B = compute_degandrad(dtype, ndim, 1)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    s[B].parallel(fused)
    return s, [A, B]


@defop(name="cuda_deg2rad", target="cuda", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def deg2rad_gpu(dtype, ndim):
    s, A, B = compute_degandrad(dtype, ndim, 0)
    s = tvm.create_schedule(B.op)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    bx, tx = s[B].split(fused, factor=64)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B]


@defop(name="cuda_rad2deg", target="cuda", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)))
def rad2deg_gpu(dtype, ndim):
    s, A, B = compute_degandrad(dtype, ndim, 1)
    s = tvm.create_schedule(B.op)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    bx, tx = s[B].split(fused, factor=64)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B]


def compute_backward_degandrad(dtype, ndim, req, n):
    ishape = [tvm.size_var() for _ in range(ndim)]
    in_grad_tmp = tvm.placeholder(ishape, name='in_grad_tmp', dtype=dtype)
    in_grad = tvm.placeholder(ishape, name='in_grad', dtype=dtype)
    out_grad = tvm.placeholder(ishape, name='out_grad', dtype=dtype)
    import math
    if n == 0:
        ret = tvm.compute(ishape, lambda *index: out_grad[index] * tvm.const(math.pi, dtype) / tvm.const(180, dtype))
    else:
        ret = tvm.compute(ishape, lambda *index: out_grad[index] / tvm.const(math.pi, dtype) * tvm.const(180, dtype))
    if (req == "kAddTo"):
        in_grad = tvm.compute(ishape, lambda *index: in_grad_tmp[index] + ret[index])
    else:
        in_grad = tvm.compute(ishape, lambda *index: ret[index])
    s = tvm.create_schedule(in_grad.op)
    return s, out_grad, in_grad_tmp, in_grad, [ret, in_grad]


@defop(name="backward_deg2rad", target="cpu", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)), req=["kWriteTo", "kAddTo"],
       attrs=["req"])
def backward_deg2rad(dtype, ndim, req):
    s, out_grad, in_grad_tmp, in_grad, c_list = compute_backward_degandrad(dtype, ndim, req, 0)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [out_grad, in_grad, in_grad_tmp]


@defop(name="backward_rad2deg", target="cpu", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)), req=["kWriteTo", "kAddTo"],
       attrs=["req"])
def backward_rad2deg(dtype, ndim, req):
    s, out_grad, in_grad_tmp, in_grad, c_list = compute_backward_degandrad(dtype, ndim, req, 1)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [out_grad, in_grad, in_grad_tmp]


@defop(name="cuda_backward_deg2rad", target="gpu", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)), req=["kWriteTo", "kAddTo"],
       attrs=["req"])
def cuda_backward_deg2rad(dtype, ndim, req):
    s, out_grad, in_grad_tmp, in_grad, c_list = compute_backward_degandrad(dtype, ndim, req, 0)
    num_thread = 64
    for t in c_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [out_grad, in_grad, in_grad_tmp]


@defop(name="cuda_backward_rad2deg", target="gpu", auto_broadcast=False,
       dtype=["float32", "float64"], ndim=list(range(0, 6)), req=["kWriteTo", "kAddTo"],
       attrs=["req"])
def cuda_backward_rad2deg(dtype, ndim, req):
    s, out_grad, in_grad_tmp, in_grad, c_list = compute_backward_degandrad(dtype, ndim, req, 1)
    num_thread = 64
    for t in c_list:
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        bx, tx = s[t].split(fused, factor=num_thread)
        s[t].bind(bx, block_x)
        s[t].bind(tx, thread_x)
    return s, [out_grad, in_grad, in_grad_tmp]
