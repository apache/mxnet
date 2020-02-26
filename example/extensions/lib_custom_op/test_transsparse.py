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
    path = os.path.abspath('libtranssparse_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libtranssparse_lib.dll')
    mx.library.load(path)

a = mx.nd.array([[1,3,0,2,1],[0,1,0,0,0],[0,2,4,5,3]])
a = a.tostype('csr')
print(type(a))
print(a.data.asnumpy())
print(a.indices.asnumpy())
print(a.indptr.asnumpy())

# To do: Fix segment fault.
b = mx.nd.my_transsparse(a)
print("B Type:", type(b))
print(b.data.asnumpy())
print(b.indices.asnumpy())
print(b.indptr.asnumpy())
