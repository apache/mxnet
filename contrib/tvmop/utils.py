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

AllTypes = ["float32", "float64", "float16", "uint8", "int8", "int32", "int64"]
RealTypes = ["float32", "float64", "float16"]


def assign_by_req(a, req, otype=None):
    b = tvm.placeholder(a.shape, name='assign_by_req_b', dtype=a.dtype)
    if req == "kAddTo":
        c = tvm.compute(a.shape, lambda *idx: a[idx].astype(otype) + b[idx]
                                              if otype else a[idx] + b[idx])
    else:
        c = tvm.compute(a.shape, lambda *idx: a[idx].astype(otype) if otype else a[idx])
    return b, c


def reduce_axes(X, axes, reducer, atype=None):
    def get_index(idx, ridx):
        j = 0
        k = 0
        ret = []
        for val in axes:
            ret.append(idx[j] if val == 0 else ridx[k])
            j += (val == 0)
            k += (val != 0)
        return tuple(ret)
    
    ishape = X.shape
    odim = (len(ishape) + 1 - axes[0]) // 2
    oshape = [tvm.var() for _ in range(odim)]
    ridx = [tvm.reduce_axis((0, ishape[i])) for (i, val) in enumerate(axes) if val == 1]
    ret = tvm.compute(oshape, lambda *idx: reducer(X[get_index(idx, ridx)].astype(atype)
                                                   if atype else X[get_index(idx, ridx)],
                                                   axis=ridx), name='ret')
    return ret
