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


# fmax forward
def fmax_forward(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: tvm.if_then_else(A[index] > B[index], A[index], B[index]), name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C


@defop(name="fmax_forward", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(1, 6)))
def fmax_forward_cpu(dtype, ndim):
    s, A, B, C = fmax_forward(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)
    return s, [A, B, C]


@defop(name="cuda_fmax_forward", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(1, 6)))
def fmax_forward_gpu(dtype, ndim):
    s, A, B, C = fmax_forward(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]


# fmax backward
def fmax_backward(dtype, ndim):
    dC = tvm.placeholder([tvm.var() for _ in range(ndim)], name='dC', dtype=dtype)
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    dA = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: tvm.if_then_else(A[index] > B[index], dC[index], 0), name='dA')
    dB = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: tvm.if_then_else(A[index] > B[index], 0, dC[index]), name='dB')
    s = tvm.create_schedule([dA.op, dB.op])
    return s, dC, A, B, dA, dB


@defop(name="fmax_backward", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(1, 6)))
def fmax_backward_cpu(dtype, ndim):
    s, dC, A, B, dA, dB = fmax_backward(dtype, ndim)
    axes = [axis for axis in dA.op.axis]
    fused = s[dA].fuse(*axes)
    s[dA].parallel(fused)
    axes = [axis for axis in dA.op.axis]
    fused = s[dB].fuse(*axes)
    s[dB].parallel(fused)
    return s, [dC, A, B, dA, dB]


@defop(name="cuda_fmax_backward", target="cuda", auto_broadcast=True,
       dtype=["float32", "float64"], ndim=list(range(1, 6)))
def fmax_backward_gpu(dtype, ndim):
    s, dC, A, B, dA, dB = fmax_backward(dtype, ndim)
    axes = [axis for axis in dA.op.axis]
    fused = s[dA].fuse(*axes)
    bx, tx = s[dA].split(fused, factor=64)
    s[dA].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[dA].bind(tx, tvm.thread_axis("threadIdx.x"))
    axes = [axis for axis in dB.op.axis]
    fused = s[dB].fuse(*axes)
    bx, tx = s[dB].split(fused, factor=64)
    s[dB].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[dB].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [dC, A, B, dA, dB]
