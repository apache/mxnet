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
from .. import defop, AllTypes
from .. import assign_by_req, reduce_axes

def compute_dot(A, B):
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    k = tvm.reduce_axis((0, K), 'k')
    C = tvm.compute((M, N),
                    lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
                    name='C')
    return C


@defop(name="dot", target="cpu", dtype=AllTypes)
def dot(dtype, fallback):
    cfg = autotvm.get_config()
    cfg.define_knob("bn", [64] if fallback else [64, 32])
    cfg.define_knob("factor", [4] if fallback else [4])
    M = tvm.var("M")
    K = tvm.var("K")
    N = tvm.var("N")
    A = tvm.placeholder((M, K), name='A', dtype=dtype)
    B = tvm.placeholder((K, N), name='B', dtype=dtype)
    C = compute_dot(A, B)
    s = tvm.create_schedule(C.op)
    # Blocking by loop tiling
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], cfg["bn"].val, cfg["bn"].val)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=cfg["factor"].val)
    # Hoist reduction domain outside the blocking loop
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    return s, [A, B, C]
