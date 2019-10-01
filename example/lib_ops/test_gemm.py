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

# This test checks if dynamic loading of library into MXNet is successful
# and checks the end of end computation of custom operator

import mxnet as mx
import os

#load library
if (os.name=='posix'):
    path = os.path.abspath('libgemm_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libgemm_lib.dll')
    mx.library.load(path)

#setup inputs to call test operator
a = mx.nd.array([[1,2],[3,4]])
b = mx.nd.array([[5,6],[7,8]])

#print inputs
print(a)
print(b)

#compute and print output
print("--------start ndarray---------")
print(mx.nd.my_gemm(a,b))
print(mx.nd.state_gemm(a,b))

# symbolic compute
print("---------start symbol--------")
s = mx.sym.Variable('s')
t = mx.sym.Variable('t')
c = mx.sym.my_gemm(s,t)
d = mx.sym.state_gemm(s,t)

in_grad = [mx.nd.empty((2,2)),mx.nd.empty((2,2))]
in_grad2 = [mx.nd.empty((2,2)),mx.nd.empty((2,2))]

exe = c.bind(ctx=mx.cpu(),args={'s':a,'t':b},args_grad=in_grad)
exe2 = d.bind(ctx=mx.cpu(),args={'s':a,'t':b},args_grad=in_grad2)

out = exe.forward()
print(out)

out2 = exe2.forward()
out2 = exe2.forward()
print(out2)

print("---------start backward--------")
out_grad = mx.nd.ones((2,2))
exe.backward([out_grad])
print(in_grad)

out_grad2 = mx.nd.ones((2,2))
exe2.backward([out_grad2])
print(in_grad2)
