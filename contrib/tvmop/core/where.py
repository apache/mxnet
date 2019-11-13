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

from .. import defop, AllTypes, RealTypes
from .. import assign_by_req, reduce_axes


def compute_where(cond_type, data_type, ndim):
    cond = tvm.placeholder([tvm.var() for _ in range(ndim)], name='cond', dtype=cond_type)
    x = tvm.placeholder([tvm.var() for _ in range(ndim)], name='x', dtype=data_type)
    y = tvm.placeholder([tvm.var() for _ in range(ndim)], name='y', dtype=data_type)
    out = tvm.compute([tvm.var() for _ in range(ndim)],
        lambda *i: tvm.if_then_else(cond[i] != tvm.const(0, cond_type), x[i], y[i]), name='out')
    s = tvm.create_schedule(out.op)
    return s, [cond, x, y, out]


@defop(name="where_cpu", target="cpu", auto_broadcast=True, ndim=[5],
       cond_type=AllTypes+['bool'], data_type=AllTypes+['bool'])
def where_cpu(cond_type, data_type, ndim):
    s, [cond, x, y, out] = compute_where(cond_type, data_type, ndim)
    axes = [axis for axis in out.op.axis]
    fused = s[out].fuse(*axes)
    bx, tx = s[out].split(fused, factor=64)
    s[out].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[out].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [cond, x, y, out]


@defop(name="where_gpu", target="gpu", auto_broadcast=True, ndim=[5],
       cond_type=AllTypes+['bool'], data_type=AllTypes+['bool'])
def where_gpu(cond_type, data_type, ndim):
    return compute_where(cond_type, data_type, ndim)


def compute_backward_where(cond_type, data_type, ndim, reduce1st_dim, req):
    axes = ([reduce1st_dim, 1 - reduce1st_dim] * ndim)[:ndim]
    reducer = tvm.comm_reducer(lambda x, y: x + y, lambda t: tvm.const(0, dtype=t), name="sum")
    ograd = tvm.placeholder([tvm.var() for _ in range(ndim)], name='ograd', dtype=data_type)
    cond = tvm.placeholder([tvm.var() for _ in range(ndim)], name='cond', dtype=cond_type)
    dx = tvm.compute([tvm.var() for _ in range(ndim)],
        lambda *i: tvm.if_then_else(cond[i] != tvm.const(0, cond_type), ograd[i], tvm.const(0, data_type)), name='dx')
    dy = tvm.compute([tvm.var() for _ in range(ndim)],
        lambda *i: tvm.if_then_else(cond[i] != tvm.const(0, cond_type), tvm.const(0, data_type), ograd[i]), name='dy')
    ret_x = reduce_axes(dx, axes, reducer)
    ret_x_origin, ret_x_new = assign_by_req(ret_x, req)
    ret_y = reduce_axes(dy, axes, reducer)
    ret_y_origin, ret_y_new = assign_by_req(ret_y, req)
    s = tvm.create_schedule([ret_x_new.op, ret_y_new.op])
    s[ret_x].compute_inline()
    s[ret_y].compute_inline()
    return s, [ograd, cond, ret_x_origin, ret_x_new, ret_y_origin, ret_y_new]


@defop(name="backward_where_cpu", target="cpu", ndim=list(range(1, 6)),
       cond_type=AllTypes+['bool'], data_type=RealTypes, reduce1st_dim=[0, 1],
       req=["kWriteTo", "kAddTo"], attrs=["reduce1st_dim", "req"])
def backward_where_cpu(cond_type, data_type, ndim, reduce1st_dim, req):
    return compute_backward_where(cond_type, data_type, ndim, reduce1st_dim, req)


@defop(name="backward_where_gpu", target="gpu", ndim=list(range(1, 6)),
       cond_type=AllTypes+['bool'], data_type=RealTypes, reduce1st_dim=[0, 1],
       req=["kWriteTo", "kAddTo"], attrs=["reduce1st_dim", "req"])
def backward_where_gpu(cond_type, data_type, ndim, reduce1st_dim, req):
    s, [ograd, cond, ret_x_origin, ret_x_new, ret_y_origin, ret_y_new] = \
        compute_backward_where(cond_type, data_type, ndim, reduce1st_dim, req)
    for out in [ret_x_new, ret_y_new]:
        axes = [axis for axis in out.op.axis]
        fused = s[out].fuse(*axes)
        bx, tx = s[out].split(fused, factor=64)
        s[out].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[out].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s, [ograd, cond, ret_x_origin, ret_x_new, ret_y_origin, ret_y_new]
