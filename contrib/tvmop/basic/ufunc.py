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

@defop(name="vadd", target="cpu", auto_broadcast=True, dtype=AllTypes, ndim=list(range(6)))
def vadd(dtype, ndim):
    A = tvm.placeholder([tvm.var() for _ in range(ndim)], name='A', dtype=dtype)
    B = tvm.placeholder([tvm.var() for _ in range(ndim)], name='B', dtype=dtype)
    C = tvm.compute([tvm.var() for _ in range(ndim)],
                    lambda *index: A[index] + B[index], name='C')

    s = tvm.create_schedule(C.op)

    return s, [A, B, C]

@defop(name="cuda_vadd", target="cuda", auto_broadcast=True, dtype="float32")
def vadd_gpu(dtype):
    m0, m1 = tvm.var("m0"), tvm.var("m1")
    n0, n1 = tvm.var("n0"), tvm.var("n1")
    o0, o1 = tvm.var("o0"), tvm.var("o1")

    A = tvm.placeholder((m0, m1), name='A', dtype=dtype)
    B = tvm.placeholder((n0, n1), name='B', dtype=dtype)
    C = tvm.compute((o0, o1), lambda i, j: A[i, j] + B[i, j], name='C')

    s = tvm.create_schedule(C.op)
    s[C].bind(C.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[C].bind(C.op.axis[1], tvm.thread_axis("threadIdx.x"))
    return s, [A, B, C]
