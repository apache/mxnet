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
from .. import defop
from ..utils import reduce_axes, assign_by_req


def _compute_sum(itype, otype, ndim, reduce1st_dim, req):
    axes = ([reduce1st_dim, 1 - reduce1st_dim] * ndim)[:ndim]
    a = tvm.placeholder([tvm.size_var() for _ in range(ndim)], name='a', dtype=itype)
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
