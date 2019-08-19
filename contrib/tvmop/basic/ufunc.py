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
from .. import defop, AllTypes

def compute_add(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: A[index] + B[index], name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C


@defop(name="vadd", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(1, 6)))
def vadd(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)

    return s, [A, B, C]


@defop(name="cuda_vadd", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(1, 6)))
def vadd_gpu(dtype, ndim):
    s, A, B, C = compute_add(dtype, ndim)
    s = tvm.create_schedule(C.op)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]


def assign_by_req(a, req):
    b = tvm.placeholder(a.shape, name='assign_by_req_b', dtype=a.dtype)
    if (req == "kAddTo"):
        c = tvm.compute(a.shape, lambda *idx: a[idx] + b[idx])
    else:
        c = tvm.compute(a.shape, lambda *idx: a[idx])
    return b, c


def reduce_axes(X, axes, reducer):
    def get_index(idx, ridx):
        j = 0
        k = 0
        ret = []
        for val in axes:
            ret.append(idx[j] if val == 0 else ridx[k])
            j += (val == 0)
            k += (val != 0)
        return tuple(ret)
    
    ishape = X.shape
    odim = (len(ishape) + 1 - axes[0]) // 2
    oshape = [tvm.var() for _ in range(odim)]
    ridx = [tvm.reduce_axis((0, ishape[i])) for (i, val) in enumerate(axes) if val == 1]
    ret = tvm.compute(oshape, lambda *idx: reducer(X[get_index(idx, ridx)], axis=ridx), name='ret')
    return ret


def compute_backward_vadd(dtype, ndim, reduce1st, req):
    axes = ([reduce1st, 1 - reduce1st] * ndim)[:ndim]
    X = tvm.placeholder([tvm.var() for _ in range(ndim)], name='X', dtype=dtype)
    reducer = tvm.comm_reducer(lambda x, y: x + y,
        lambda t: tvm.const(0, dtype=t), name="sum")
    ret = reduce_axes(X, axes, reducer)
    in_grad_a, in_grad = assign_by_req(ret, req)
    s = tvm.create_schedule(in_grad.op)
    return s, X, in_grad_a, in_grad, [ret, in_grad]


@defop(name="backward_vadd", target="cpu", dtype=AllTypes, 
       ndim=list(range(1, 6)), reduce1st=[0, 1],
       req=["kWriteTo", "kAddTo"], attrs=["reduce1st", "req"])
def backward_vadd(dtype, ndim, reduce1st, req):
    s, X, in_grad_a, in_grad, c_list = compute_backward_vadd(dtype, ndim, reduce1st, req)
    for t in c_list:
        axes = [axis for axis in t.op.axis]
        fused = s[t].fuse(*axes)
        s[t].parallel(fused)
    return s, [X, in_grad_a, in_grad]


@defop(name="cuda_backward_vadd", target="gpu", dtype=["float32", "float64"],
       ndim=list(range(1, 6)), reduce1st=[0, 1],
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
