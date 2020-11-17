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
    path = os.path.abspath('libgemm_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libgemm_lib.dll')
    mx.library.load(path)

a = mx.nd.array([[1,2,3],[4,5,6]])
b = mx.nd.array([[7],[8],[9]])

print("--------start ndarray compute---------")
print(mx.nd.my_gemm(a,b))
print("--------")
print(mx.nd.state_gemm(a,b,test_kw=100))

print("--------start symbolic compute--------")
s = mx.sym.Variable('s')
t = mx.sym.Variable('t')
c = mx.sym.my_gemm(s,t)
d = mx.sym.state_gemm(s,t,test_kw=200)
e = mx.sym.linalg.gemm2(s,t)

out_grad = mx.nd.ones((2,1))

# stateless
block = mx.gluon.nn.SymbolBlock(c,[s,t])
with mx.autograd.record():
    a_ = mx.nd.array([[1,2,3],[4,5,6]])
    b_ = mx.nd.array([[7],[8],[9]])
    a_.attach_grad()
    b_.attach_grad()
    # foward
    out = block(a_,b_)
    print(out)
    print('+++++')
    # backward
    out.backward(out_grad)
    print(a_.grad)
    print(b_.grad)
    print("-------")

# stateful
block2 = mx.gluon.nn.SymbolBlock(d,[s,t])
block2.hybridize(static_alloc=True, static_shape=True)
out2 = block2(a,b)
out2 = block2(a,b)
print(out2)
with mx.autograd.record():
    a_ = mx.nd.array([[1,2,3],[4,5,6]])
    b_ = mx.nd.array([[7],[8],[9]])
    a_.attach_grad()
    b_.attach_grad()
    # forward
    out2 = block2(a_,b_)
    print('+++++')
    # backward
    out2.backward(out_grad)
    print(a_.grad)
    print(b_.grad)
    print("-------")

# baseline
block3 = mx.gluon.nn.SymbolBlock(e,[s,t])
with mx.autograd.record():
    a_ = mx.nd.array([[1,2,3],[4,5,6]])
    b_ = mx.nd.array([[7],[8],[9]])
    a_.attach_grad()
    b_.attach_grad()
    # forward
    out3 = block3(a_,b_)
    print(out3)
    print('+++++')
    # backward
    out3.backward(out_grad)
    print(a_.grad)
    print(b_.grad)
