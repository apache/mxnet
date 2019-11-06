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

import math

import tvm

from ...utils import acc_type_resolver

__all__ = ['abs_cpu', 'abs_gpu', 'sign', 'deg2rad', 'rad2deg']


def abs_cpu(a):
    return tvm.abs(a)


def abs_gpu(a):
    if a.dtype == "float16":
        return tvm.abs(a.astype("float32")).astype("float16")
    else:
        return tvm.abs(a)


# todo(hgt312): handle unsign?
def sign(a):
    dtype = a.dtype
    return tvm.if_then_else(
               a > tvm.const(0, dtype),
               tvm.const(1, dtype),
               tvm.if_then_else(
                   a < tvm.const(0, dtype),
                   tvm.const(-1, dtype),
                   tvm.const(0, dtype)
               )
           )


def deg2rad(a):
    otype = acc_type_resolver[a.dtype]
    return a.astype(otype) * tvm.const(math.pi, otype) / tvm.const(180, otype)


def rad2deg(a):
    otype = acc_type_resolver[a.dtype]
    return a.astype(otype) / tvm.const(math.pi, otype) * tvm.const(180, otype)
