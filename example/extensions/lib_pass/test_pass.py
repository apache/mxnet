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
    path = os.path.abspath('libpass_lib.so')
    mx.library.load(path)
elif (os.name=='nt'):
    path = os.path.abspath('libpass_lib.dll')
    mx.library.load(path)

###############################################
# Test with not consuming params
###############################################
# example model, ops do not have args (use outputs from other ops as inputs)
a = mx.sym.var('a')
b = mx.sym.var('b')
c = a + b
d = mx.sym.exp(c)
sym = mx.sym.log(d)

def test_model(pass_name):
    # execute in MXNet
    print('-------------------------------')
    print('Testing regular MXNet execution')
    exe = sym.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
    out = exe.forward()
    print(out)

    # Symbol optimize_for
    # with propogating shapes/types
    print('-------------------------------')
    print('Testing pass "%s" with shapes/types' % pass_name)
    arg_array = [mx.nd.ones((3,2),dtype='float32'), mx.nd.ones((3,2),dtype='float32')]
    aux = []
    mysym2 = sym.optimize_for(pass_name,arg_array,aux)
    print(mysym2.tojson())
    exe2 = mysym2.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
    out2 = exe2.forward()
    print(out2)

    # without propogating shapes/types
    print('-------------------------------')
    print('Testing pass "%s" without shapes/types' % pass_name)
    mysym3 = sym.optimize_for(pass_name, myOpt='yello')
    exe3 = mysym3.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
    out3 = exe3.forward()
    print(out3)

    # Gluon Hybridize
    print('-------------------------------')
    print('Testing pass "%s" Gluon Hybridize with shapes/types' % pass_name)
    inputs = [a,b]
    sym_block = nn.SymbolBlock(sym, inputs)
    sym_block.initialize()
    sym_block.hybridize(backend=pass_name)
    out4 = sym_block(mx.nd.ones((3,2)),mx.nd.ones((3,2)))
    print(out4)
    
    # Gluon optimize_for
    print('-------------------------------')
    print('Testing pass "%s" Gluon Hybridize with shapes/types without inference' % pass_name)
    inputs = [a,b]
    sym_block2 = nn.SymbolBlock(sym, inputs)
    sym_block2.initialize()
    sym_block2.optimize_for(mx.nd.ones((3,2)), mx.nd.ones((3,2)), backend=pass_name)
    sym_block2.export('modified')

test_model('myPass')
test_model('jsonPass')
