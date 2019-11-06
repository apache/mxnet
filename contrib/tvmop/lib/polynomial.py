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
from tvm import autotvm
from .. import defop, AllTypes, RealTypes, IntegralTypes
from .. import assign_by_req, reduce_axes


def decl_args(dtype):
    N = tvm.var("N")
    M = tvm.var("M")
    P = tvm.placeholder((N,), name="P", dtype=dtype)
    X = tvm.placeholder((M,), name="X", dtype=dtype)
    return N, M, P, X


def polyval_compute(dtype):
    N, M, P, X = decl_args(dtype)
    Px = tvm.compute((N, M),
                    lambda i, j: P[i]*tvm.power(X[j], N.astype(dtype)-i-1),
                    name="Px")
    n = tvm.reduce_axis((0, N), 'n')
    V = tvm.compute(X.shape,
                    lambda j: tvm.sum(Px[n, j], axis=n),
                    name="V")
    s = tvm.create_schedule(V.op)
    s[Px].compute_inline()
    return s, [P, X, V]


@defop(name="polyval", target="cpu", dtype=['float32', 'float64'])
def polyval(dtype):
    return polyval_compute(dtype)


def cuda_split(sch, tensor):
    axes = [axis for axis in tensor.op.axis]
    fused = sch[tensor].fuse(*axes)
    bx, tx = sch[tensor].split(fused, factor=64)
    sch[tensor].bind(bx, tvm.thread_axis("blockIdx.x"))
    sch[tensor].bind(tx, tvm.thread_axis("threadIdx.x"))


@defop(name="polyval_cuda", target="cuda", dtype=['float32', 'float64'])
def polyval_cuda(dtype):
    s, [P, X, V] = polyval_compute(dtype)
    cuda_split(s, V)
    return s, [P, X, V]


def horner(x):
    return tvm.comm_reducer(lambda a, b: a*x + b, lambda t: tvm.const(0, t), name="horner")


def polyval_horner_compute(dtype):
    N, M, P, X = decl_args(dtype)
    n = tvm.reduce_axis((0, N), 'n')
    V = tvm.compute(X.shape, lambda j: horner(X[j])(P[n], axis=n), name="V")
    s = tvm.create_schedule(V.op)
    return s, [P, X, V]


@defop(name="polyval_horner", target="cpu", dtype=IntegralTypes)
def polyval_horner(dtype):
    return polyval_horner_compute(dtype)


@defop(name="polyval_horner_cuda", target="cuda", dtype=IntegralTypes)
def polyval_horner_cuda(dtype):
    s, [P, X, V] = polyval_horner_compute(dtype)
    cuda_split(s, V)
    return s, [P, X, V]


def backward_polyval_compute(dtype, req):
    N, M, P, X = decl_args(dtype)
    ograd = tvm.placeholder(X.shape, name="ograd", dtype=dtype)
    n = tvm.reduce_axis((0, N-1), 'n')
    m = tvm.reduce_axis((0, M), 'm')
    d_p = tvm.compute((N, M),
                        lambda i, j: tvm.power(X[j], (N-i-1).astype(dtype)), name="d_p")
    igrad_p_bc = tvm.compute((N, M), lambda i, j: ograd[j]*d_p[i, j], name="igrad_p_bc")
    igrad_p = tvm.compute(P.shape,
                        lambda i: tvm.sum(igrad_p_bc[i, m], axis=m), name="igrad_p")
    d_x = tvm.compute(X.shape, lambda j: horner(X[j])((N-n-1)*P[n], axis=n), name="d_x")
    igrad_x = tvm.compute(X.shape, lambda j: d_x[j]*ograd[j], name="igrad_x")
    igrad_p_placeholder, igrad_p_out = assign_by_req(igrad_p, req)
    igrad_x_placeholder, igrad_x_out = assign_by_req(igrad_x, req)
    s = tvm.create_schedule([igrad_p_out.op, igrad_x_out.op])
    s[d_p].compute_inline()
    s[igrad_p_bc].compute_inline()
    s[igrad_p].compute_inline()
    s[igrad_x].compute_inline()
    return s, [ograd, P, X,
    igrad_p_placeholder, igrad_p_out,
    igrad_x_placeholder, igrad_x_out], [d_x, igrad_p, igrad_x]


@defop(name="backward_polyval", target="cpu", dtype=['float32', 'float64'],
       req=['kWriteTo', 'kAddTo'], attrs=['req'])
def backward_polyval(dtype, req):
    s, tensors, _ = backward_polyval_compute(dtype, req)
    return s, tensors


@defop(name="backward_polyval_cuda", target="cuda", dtype=['float32', 'float64'],
       req=['kWriteTo', 'kAddTo'], attrs=['req'])
def backward_polyval_cuda(dtype, req):
    s, [ograd, P, X, _p, igrad_p, _x, igrad_x], [d_x, o_p, o_x] = backward_polyval_compute(dtype, req)
    cuda_split(s, igrad_p)
    cuda_split(s, igrad_x)
    cuda_split(s, d_x)
    cuda_split(s, o_p)
    cuda_split(s, o_x)
    return s, [ograd, P, X, _p, igrad_p, _x, igrad_x]
