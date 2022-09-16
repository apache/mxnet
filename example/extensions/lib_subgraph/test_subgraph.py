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

import os, ctypes
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
from mxnet.base import _LIB, check_call, mx_uint, c_str, c_str_array, SymbolHandle

# load library
if (os.name=='posix'):
    path = os.path.abspath('libsubgraph_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libsubgraph_lib.dll')
    mx.library.load(path)

# example model, ops to be partitioned do not have args (use outputs from other ops as inputs)
a = mx.sym.var('a')
b = mx.sym.var('b')
c = a + b
d = mx.sym.exp(c)
sym = mx.sym.log(d)

# example model, ops to be partitioned have args
d2 = mx.sym.exp(a)
sym2 = mx.sym.log(d2)

def test(backend):
    args = {'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))}
    ###############################################
    # Test with subgraph not consuming params
    ###############################################
    #execute in MXNet
    print('-------------------------------')
    print('Testing regular Gluon execution')
    inputs = [a,b]
    sym_block = nn.SymbolBlock(sym, inputs)
    sym_block.initialize()
    out = sym_block(mx.nd.ones((3,2)),mx.nd.ones((3,2)))
    print(out)

    # Gluon Hybridize partitioning with shapes/types without inference
    print('-------------------------------')
    print(f'Testing {backend} Gluon Hybridize partitioning with shapes/types without inference')
    inputs = [a,b]
    sym_block2 = nn.SymbolBlock(sym, inputs)
    sym_block2.initialize()
    sym_block2.optimize_for(mx.nd.ones((3,2)), mx.nd.ones((3,2)), backend=backend)
    sym_block2.export('partitioned')

    # Test with additional input to subgraph op
    print('-------------------------------')
    print(f'Testing {backend} Gluon Hybridize partitioning with extra input')
    sym_block2.optimize_for(mx.nd.ones((3,2)), mx.nd.ones((3,2)), backend="addInputPass")
    out3 = sym_block2(mx.nd.ones((3,2)),mx.nd.ones((3,2)))
    print(out3)
    
    
    ###############################################
    # Test with subgraph directly consuming params
    ###############################################
    args = {'a':mx.nd.ones((3,2))}
    #execute in MXNet
    print('-------------------------------')
    print('Testing regular MXNet execution')
    inputs = [a]
    sym2_block = nn.SymbolBlock(sym2, inputs)
    sym2_block.initialize()
    out5 = sym2_block(mx.nd.ones((3,2)))
    print(out5)

    # Gluon optimize_for partitioning with shapes/types
    print('-------------------------------')
    print(f'Testing {backend} Gluon optimize_for partitioning with shapes/types')
    inputs = [a]
    sym2_block = nn.SymbolBlock(sym2, inputs)
    sym2_block.initialize()
    sym2_block.optimize_for(mx.nd.ones((3,2)), backend=backend)
    out8 = sym2_block(mx.nd.ones((3,2)))
    print(out8)

test("myProp")
test("mySelect")
