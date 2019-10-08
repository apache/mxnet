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
from ...opdef import defop
from ...utils import AllTypes, assign_by_req, reduce_axes

__all__ = ['compute_two2one', 'two2one_cpu', 'two2one_gpu']


def compute_two2one(op, atype, btype, ndim):
    a = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=atype, name='a')
    b = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=btype, name='b')
    c = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *idx: op(a, b), name='c')
    s = tvm.create_schedule(c.op)
    return s, a, b, c


def two2one_cpu(op, atype, btype, ndim):
    s, a, b, c = compute_two2one(op, atype, btype, ndim)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    s[c].parallel(fused)
    return s, [a, b, c]


def two2one_gpu(op, atype, btype, ndim):
    s, a, b, c = compute_two2one(op, atype, btype, ndim)
    axes = [axis for axis in c.op.axis]
    fused = s[c].fuse(*axes)
    bx, tx = s[c].split(fused, factor=64)
    s[c].bind(bx, tvm.thread_axis('blockIdx.x'))
    s[c].bind(tx, tvm.thread_axis('threadIdx.x'))
    return s, [a, b, c]
