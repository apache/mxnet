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
from mxnet import np, npx
from gluonnlp.layers import get_activation
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


class Easynet(nn.HybridBlock):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Dense(in_units=2, units=2, flatten=False)
        #self.l2 = nn.Dense(in_units=2, units=2, flatten=False)
        self.act1 = get_activation('relu')

        #self.seq.add(nn.Dense(in_units=2, units=2, flatten=False))
        #self.seq.add(get_activation('relu'))
        #self.seq.add(nn.Dense(in_units=2, units=2, flatten=False))
        #self.seq.register_op_hook(mon_callback,  monitor_all=True)
        #self.l1.register_op_hook(mon_callback,  monitor_all=True)


    def forward(self, input):
        input = self.l1(input)
        input = self.act1(input)
        return input


def test_model(pass_name):
    model = Easynet()
    model.initialize()
    model.hybridize()


    print('try on model')
    x = np.array([[1,2]])

    model.optimize_for(x, backend = pass_name)

    out = model(x)
    model.export("my_model")
    print(out.shape)
    return
    args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))}
    # execute in MXNet
    print('-------------------------------')
    print('Testing regular MXNet execution')
    inputs = [a,b]
    sym_block = nn.SymbolBlock(sym, inputs)
    sym_block.initialize()
    out = sym_block(mx.nd.ones((3,2)),mx.nd.ones((3,2)))
    print(out)

    # Gluon optimize_for
    print('-------------------------------')
    print('Testing pass "%s" Gluon Hybridize with shapes/types without inference' % pass_name)
    inputs = [a,b]
    sym_block2 = nn.SymbolBlock(sym, inputs)
    sym_block2.initialize()
    sym_block2.optimize_for(mx.nd.ones((3,2)), mx.nd.ones((3,2)), backend=pass_name)
    sym_block2.export('modified')

test_model('myPass')
