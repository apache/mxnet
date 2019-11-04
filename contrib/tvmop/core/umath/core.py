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
from ...utils import assign_by_req

__all__ = ['unary_cpu', 'unary_gpu']


def compute_unary(op, dtype, ndim, req):
    x = tvm.placeholder([tvm.var() for _ in range(ndim)], dtype=dtype, name='x')
    y = tvm.compute([tvm.var() for _ in range(ndim)], lambda *idx: op(x[idx]), name='y')
    old, new = assign_by_req(y, req)
    s = tvm.create_schedule(new.op)
    s[y].compute_inline()
    return s, x, old, new


def unary_cpu(op, dtype, ndim, req):
    s, x, old, new = compute_unary(op, dtype, ndim, req)
    return s, [x, old, new]


def unary_gpu(op, dtype, ndim, req):
    s, x, old, new = compute_unary(op, dtype, ndim, req)
    bx, tx = s[new].split(new.op.axis[0], factor=64)
    s[new].bind(bx, tvm.thread_axis('blockIdx.x'))
    s[new].bind(tx, tvm.thread_axis('threadIdx.x'))
    return s, [x, old, new]
