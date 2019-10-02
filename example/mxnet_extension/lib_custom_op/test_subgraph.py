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
ret = mx.sym.log(d)

op_names = ['exp','log']
out = SymbolHandle()

check_call(_LIB.MXBuildSubgraphByOpNames(ret.handle,
                                         c_str('default'),
                                         mx_uint(len(op_names)),
                                         c_str_array(op_names),
                                         ctypes.byref(out)))
partitioned_sym = mx.sym.Symbol(out)
json_sym = partitioned_sym.tojson()

mystr = json_sym.replace("_CachedOp","_custom_subgraph_op")
mysym = mx.sym.load_json(mystr)

exe = mysym.bind(ctx=mx.cpu(), args={'a':mx.nd.ones((3,2)), 'b':mx.nd.ones((3,2))})
out = exe.forward()
print(out)
