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
# AllTypes = ["float32", "float64", "float16", "uint8", "int8", "int32", "int64"]
# RealTypes = ["float32", "float64", "float16"]
# AccTypes = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64'}


# def assign_by_req(a, req, otype):
#     b = tvm.placeholder(a.shape, name='assign_by_req_b', dtype=otype)
#     if (req == "kAddTo"):
#         c = tvm.compute(a.shape, lambda *idx: a[idx].astype(otype) + b[idx])
#     else:
#         c = tvm.compute(a.shape, lambda *idx: a[idx].astype(otype))
#     return b, c
# 
# def reduce_axes(X, axes, reducer, atype=None):
#     def get_index(idx, ridx):
#         j = 0
#         k = 0
#         ret = []
#         for val in axes:
#             ret.append(idx[j] if val == 0 else ridx[k])
#             j += (val == 0)
#             k += (val != 0)
#         return tuple(ret)
# 
#     ishape = X.shape
#     odim = (len(ishape) + 1 - axes[0]) // 2
#     oshape = [tvm.var('odim.%d' % i, 'int32') for i in range(odim)]
#     if atype is None:
#         atype = X.dtype
#     ridx = [tvm.reduce_axis((0, ishape[i]), name='r%d' % i) for (i, val) in enumerate(axes) if val == 1]
#     ret = tvm.compute(oshape, lambda *idx: reducer(X[get_index(idx, ridx)].astype(atype), axis=ridx), name='ret')
#     return ret

def compute_reduce(dtype, otype, reducer, initial, ndim, reduce1st, req):
    axes = ([reduce1st, 1 - reduce1st] * ndim)[:ndim]
    X = tvm.placeholder([tvm.var('idim.%d' % i, 'int32') for i in range(ndim)], name='X', dtype=dtype)
    atype = AccTypes[dtype]
    reducer = tvm.comm_reducer(reducer, initial, name='reducer')
    ret = reduce_axes(X, axes, reducer, atype=atype)
    out_a, out = assign_by_req(ret, req, otype)
    s = tvm.create_schedule(out.op)
    return s, X, out_a, out, [ret, out]

@defop(name="tvm_sum", target='cpu', dtype=RealTypes, ndim=[5],
       reduce1st=[0, 1], otype=RealTypes, attrs=["reduce1st", "otype"])
def sum_forward(dtype, ndim, reduce1st, otype):
    s, X, out_a, out, c_list = compute_reduce(dtype, otype, lambda x, y: x + y,
                                              lambda t: tvm.const(0, dtype=t), ndim, reduce1st,
                                              'kWriteTo')
    # for t in c_list:
        # axes = [axis for axis in t.op.axis]
        # fused = s[t].fuse(*axes)
        # s[t].parallel(fused)
    # print(tvm.build_module.form_body(s))
    print(tvm.build_module.form_body(s))
    return s, [X, out_a, out]


@defop(name="cuda_tvm_sum", target='cuda', dtype=['float32', 'float64'], ndim=[5], reduce1st=[0, 1],
       otype=['float32', 'float64'], attrs=["reduce1st", "otype"])
def sum_forward_gpu(dtype, ndim, reduce1st, otype):
    s, X, out_a, out, c_list = compute_reduce(dtype, otype, lambda x, y: x + y,
                                              lambda t: tvm.const(0, dtype=t), ndim, reduce1st,
                                              'kWriteTo')
    num_thread = 64
    # for t in c_list:
    #     block_x = tvm.thread_axis("blockIdx.x")
    #     thread_x = tvm.thread_axis("threadIdx.x")
    #     axes = [axis for axis in t.op.axis]
    #     fused = s[t].fuse(*axes)
    #     bx, tx = s[t].split(fused, factor=num_thread)
    #     s[t].bind(bx, block_x)
    #     s[t].bind(tx, thread_x)
    # print(tvm.build_module.form_body(s))
    return s, [X, out_a, out]
