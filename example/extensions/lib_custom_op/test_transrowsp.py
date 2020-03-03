#!/usr/bin/env python3

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
# pylint: disable=arguments-differ

# This test checks dynamic loading of custom library into MXNet
# and checks end to end compute of a simple 2D gemm custom op

import mxnet as mx
import os

#load library
if (os.name=='posix'):
    path = os.path.abspath('libtransrowsp_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libtransrowsp_lib.dll')
    mx.library.load(path)

a = mx.nd.array([[1,2,3],[0,0,0],[4,0,5],[0,0,0],[0,0,0]])
# a = mx.nd.array([[1,3,0,2,1],[0,1,0,0,0],[0,2,4,5,3]])
a = a.tostype('row_sparse')
print("--------Input CSR Array---------")
print("data:", a.data.asnumpy())
print("indices:", a.indices.asnumpy())

print("--------Start NDArray Compute---------")
b = mx.nd.my_transrowsp(a)
print("Compute Results:")
print("data:", b.data.asnumpy())
print("indices:", b.indices.asnumpy())

print("Stateful Compute Result:")
c = mx.nd.state_transrowsp(a, test_kw=100)
print("data:", c.data.asnumpy())
print("indices:", c.indices.asnumpy())

print("--------start symbolic compute--------")
d = mx.sym.Variable('d')
e = mx.sym.my_transrowsp(d)
f = mx.sym.state_transrowsp(d, test_kw=200)

exe = e.bind(ctx=mx.cpu(),args={'d':a})
exe2 = f.bind(ctx=mx.cpu(),args={'d':a})
out = exe.forward()
print("Compute Results:")
print("data:", out[0].data.asnumpy())
print("indices:", out[0].indices.asnumpy())

out2 = exe2.forward()
out2 = exe2.forward()
print("Stateful Compute Result:")
print("data:", out2[0].data.asnumpy())
print("indices:", out2[0].indices.asnumpy())
