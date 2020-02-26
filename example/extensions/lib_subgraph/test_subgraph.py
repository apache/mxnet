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
import os, ctypes
from mxnet.base import _LIB, check_call, mx_uint, c_str, c_str_array, SymbolHandle

# load library
if (os.name=='posix'):
    path = os.path.abspath('libsubgraph_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libsubgraph_lib.dll')
    mx.library.load(path)

a = mx.sym.var('a')
b = mx.sym.var('b')
c = a + b
d = mx.sym.exp(c)
sym = mx.sym.log(d)

#execute in MXNet
print('-------------------------------')
print('Testing regular MXNet execution')
exe = sym.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
out = exe.forward()
print(out)

# with propogating shapes/types
print('-------------------------------')
print('Testing partitioning with shapes/types')
arg_array = [mx.nd.ones((3,2),dtype='float32'), mx.nd.ones((3,2),dtype='float32')]
mysym2 = sym.optimize_for("myProp",arg_array)
print(mysym2.tojson())
exe2 = mysym2.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
out2 = exe2.forward()
print(out2)

# with propogating shapes/types, rejecting subgraph
print('-------------------------------')
print('Testing partitioning with shapes/types - rejecting subgraph')
arg_array = [mx.nd.ones((3,2),dtype='float32'), mx.nd.ones((3,2),dtype='float32')]
mysym2 = sym.optimize_for("myProp", arg_array, reject=True)
exe2 = mysym2.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
out2 = exe2.forward()
print(out2)

# without propogating shapes/types
print('-------------------------------')
print('Testing partitioning without shapes/types')
mysym3 = sym.optimize_for("myProp", myOpt='yello')
exe3 = mysym3.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
out3 = exe3.forward()
print(out3)

from mxnet.gluon import nn
from mxnet import nd

# Gluon Hybridize partitioning with shapes/types
print('-------------------------------')
print('Testing Gluon Hybridize partitioning with shapes/types')
inputs = [a,b]
sym_block = nn.SymbolBlock(sym, inputs)
sym_block.initialize()
sym_block.hybridize(backend='myProp')
out4 = sym_block(mx.nd.ones((3,2)),mx.nd.ones((3,2)))
print(out4)

