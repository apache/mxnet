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
import numpy as np

def compute_sinc(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='input', dtype=dtype)
    if dtype in ['float16', 'float32', 'float64']:
        var = tvm.const(np.pi, dtype)
        B = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: tvm.if_then_else(A[index] == 0, tvm.const(1, dtype),
                                                        tvm.sin(var * A[index]) / (A[index] * var)),
                                                        name='output')
    else:
        var = tvm.const(np.pi, "float64")
        B = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: tvm.if_then_else(A[index] == 0, tvm.const(1, 'float64'),
                                                        tvm.sin(var * A[index].astype('float64')) /
                                                        (A[index].astype("float64") * var)),
                                                        name='output')

    s = tvm.create_schedule(B.op)
    return s, A, B


@defop(name="sinc_cpu", target="cpu", auto_broadcast=False,
       dtype=AllTypes, ndim=[5])
def _sinc_cpu(dtype, ndim):
    s, A, B = compute_sinc(dtype, ndim)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    s[B].parallel(fused)
    return s, [A, B]


@defop(name="sinc_gpu", target="cuda", auto_broadcast=False,
       dtype=AllTypes, ndim=[5])
def _sinc_gpu(dtype, ndim):
    s, A, B = compute_sinc(dtype, ndim)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    bx, tx = s[B].split(fused, factor=64)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B]


def compute_backward_sinc(dtype, ndim, req):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.placeholder([tvm.var() for _ in range(ndim)], name='C', dtype=dtype)
    var = tvm.const(np.pi, dtype)
    D = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: tvm.if_then_else(B[index] == 0, tvm.const(0, dtype),
                                                    (tvm.cos(var * B[index]) / B[index] -
                                                    C[index] / B[index]) * A[index]), name='in_grad')
    in_grad_a, in_grad = assign_by_req(D, req)
    s = tvm.create_schedule(in_grad.op)
    s[D].compute_inline()
    return s, A, B, C, in_grad_a, in_grad


@defop(name="backward_sinc_cpu", target="cpu", dtype=RealTypes,
       ndim=[5], req=["kWriteTo", "kAddTo"], attrs=["req"])
def _backward_sinc_cpu(dtype, ndim, req):
    s, A, B, C, in_grad_a, in_grad = compute_backward_sinc(dtype, ndim, req)
    axes = [axis for axis in in_grad.op.axis]
    fused = s[in_grad].fuse(*axes)
    s[in_grad].parallel(fused)
    return s, [A, B, C, in_grad_a, in_grad]


@defop(name="backward_sinc_gpu", target="cuda", dtype=RealTypes,
       ndim=[5], req=["kWriteTo", "kAddTo"], attrs=["req"])
def _backward_sinc_gpu(dtype, ndim, req):
    s, A, B, C, in_grad_a, in_grad = compute_backward_sinc(dtype, ndim, req)
    num_thread = 64
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    axes = [axis for axis in in_grad.op.axis]
    fused = s[in_grad].fuse(*axes)
    bx, tx = s[in_grad].split(fused, factor=num_thread)
    s[in_grad].bind(bx, block_x)
    s[in_grad].bind(tx, thread_x)
    return s, [A, B, C, in_grad_a, in_grad]
