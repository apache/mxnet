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
import time

#load library
if (os.name=='posix'):
    path = os.path.abspath('librelu_lib.so')
    mx.library.load(path)

a = mx.nd.array([[-2,-1],[1,2]], ctx=mx.cpu())
b = mx.nd.array([[-2,-1],[1,2]], ctx=mx.gpu())

print("--------start ndarray compute---------")
print(mx.nd.my_relu(a))
print(mx.nd.my_relu(b))

print("--------start symbolic compute--------")
c = mx.sym.Variable('c')
d = mx.sym.my_relu(c)
in_grad = [mx.nd.empty((2,2), ctx=mx.gpu())]
exe = d.bind(ctx=mx.gpu(), args={'c':b}, args_grad=in_grad)
out = exe.forward()
print(out)

print("--------start backward compute--------")
out_grad = mx.nd.ones((2,2), ctx=mx.gpu())
exe.backward([out_grad])
print(in_grad)

print("--------start stress test---------")
a = mx.nd.uniform(shape=(1000,1000,100), ctx=mx.cpu())
b = mx.nd.uniform(shape=(1000,1000,100), ctx=mx.gpu())
t1 = time.time()
r1 = mx.nd.my_relu(a)
t2 = time.time()
r2 = mx.nd.my_relu(b)
t3 = time.time()
print("CPU running time:")
print(t2 - t1)
print("GPU running time:")
print(t3 - t2)
