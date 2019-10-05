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
from .. import defop, AllTypes, AllTypesButHalf

def compute_true_divide(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    if dtype in ["float16", "float32", "float64"]:
        C = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: A[index] / B[index], name='C')
    else:
        C = tvm.compute([tvm.var() for _ in range(ndim)],
                        lambda *index: A[index].astype("float32") /
                                       B[index].astype("float32"), name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C

@defop(name="true_divide", target="cpu", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(6)))
def true_divide(dtype, ndim):
    s, A, B, C = compute_true_divide(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)
    return s, [A, B, C]

@defop(name="cuda_true_divide", target="cuda", auto_broadcast=True,
       dtype=AllTypes, ndim=list(range(6)))
def true_divide_gpu(dtype, ndim):
    s, A, B, C = compute_true_divide(dtype, ndim)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]

#  r represents the position of tensor
def compute_true_divide_scalar(dtype, ndim, r):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.var(name='B', dtype="float64")
    if dtype in ["float16", "float32", "float64"]:
        if r == 0:
            C = tvm.compute([tvm.var() for _ in range(ndim)],
                            lambda *index: A[index] / B.astype(dtype), name='C')
        else:
            C = tvm.compute([tvm.var() for _ in range(ndim)],
                            lambda *index: B.astype(dtype) / A[index], name='C')
    else:
        if r == 0:
            C = tvm.compute([tvm.var() for _ in range(ndim)],
                            lambda *index: A[index].astype("float32") / B.astype("float32"), name='C')
        else:
            C = tvm.compute([tvm.var() for _ in range(ndim)],
                            lambda *index: B.astype("float32") / A[index].astype("float32"), name='C')
    s = tvm.create_schedule(C.op)
    return s, A, B, C

@defop(name="true_divide_scalar", target="cpu",
       dtype=AllTypes, ndim=list(range(6)))
def true_divide_scalar(dtype, ndim):
    s, A, B, C = compute_true_divide_scalar(dtype, ndim, 0)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)
    return s, [A, B, C]

@defop(name="cuda_true_divide_scalar", target="cuda",
       dtype=AllTypes, ndim=list(range(6)))
def true_divide_scalar_gpu(dtype, ndim):
    s, A, B, C = compute_true_divide_scalar(dtype, ndim, 0)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]

@defop(name="rtrue_divide_scalar", target="cpu",
       dtype=AllTypes, ndim=list(range(6)))
def rtrue_divide_scalar(dtype, ndim):
    s, A, B, C = compute_true_divide_scalar(dtype, ndim, 1)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    s[C].parallel(fused)
    return s, [A, B, C]

@defop(name="cuda_rtrue_divide_scalar", target="cuda",
       dtype=AllTypes, ndim=list(range(6)))
def rtrue_divide_scalar_gpu(dtype, ndim):
    s, A, B, C = compute_true_divide_scalar(dtype, ndim, 1)
    axes = [axis for axis in C.op.axis]
    fused = s[C].fuse(*axes)
    bx, tx = s[C].split(fused, factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]
