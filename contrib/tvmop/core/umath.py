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
import topi
from .. import defop, AllTypes
from .. import assign_by_req, reduce_axes


def compute_exp2(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    if dtype in ['float32', 'float64']:
        B = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: topi.power(2, A[index]), name='B')
    else:
        B = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: topi.power(2, A[index].astype('float32')).astype(dtype),
                        name='B')
    s = tvm.create_schedule(B.op)
    return s, A, B

@defop(name="exp2_cpu", target="cpu", auto_broadcast=False,
       dtype=AllTypes, ndim=list(range(0, 6)))
def _exp2_cpu(dtype, ndim):
    s, A, B = compute_exp2(dtype, ndim)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    s[B].reorder(fused)
    s[B].parallel(fused)
    return s, [A, B]


@defop(name="exp2_gpu", target="cuda", auto_broadcast=False,
       dtype=AllTypes, ndim=list(range(0, 6)))
def _exp2_gpu(dtype, ndim):
    s, A, B= compute_exp2(dtype, ndim)
    s = tvm.create_schedule(B.op)
    axes = [axis for axis in B.op.axis]
    fused = s[B].fuse(*axes)
    bx, tx = s[B].split(fused, factor=64)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B]


def compute_backward_exp2(dtype, ndim):
    log2 = 0.6931471805599453
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index:  A[index] * B[index] * tvm.const(log2, dtype=dtype), name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C


@defop(name="backward_exp2_cpu", target="cpu", auto_broadcast=False,
       dtype=AllTypes, ndim=list(range(0, 6)))
def _backward_exp2_cpu(dtype, ndim):
    s, A, B, C = compute_backward_exp2(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].reorder(fused)
    s[C].parallel(fused)
    return s, [A, B, C]


@defop(name="backward_exp2_gpu", target="cuda", auto_broadcast=False,
       dtype=AllTypes, ndim=list(range(0, 6)))
def _backward_exp2_gpu(dtype, ndim):
    s, A, B, C= compute_backward_exp2(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]
