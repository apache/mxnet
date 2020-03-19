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

import os
import sys
import mxnet as mx
import numpy as np
from random import randint
import warnings
import collections
import ctypes
import itertools
import mxnet.contrib.amp as amp
from nose.tools import assert_raises
from mxnet.test_utils import set_default_context, download_model, same_symbol_structure, assert_almost_equal_with_err, rand_shape_nd
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
from mxnet.contrib.amp import amp
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed
import unittest

bfloat16 = np.dtype([('bfloat16', np.uint16)])

def check_operator_accuracy(sym_fp32, sym_bf16, data_shape, num_input_data=1, bf16_use_fp32_params=False, rtol=1e-1, atol=5e-1, etol=0):
    """
    check accuracy for bfloat16 operators

    sym_fp32: Symbol
        fp32 operator
    sym_bf16: Symbol
        bf16 operator
    data_shape: tuple of int
        input data shape for fp32/bf16 symbol
    num_input_data: int
        number of input data, default is 1, should set different values for those operators with multiple inputs, like concat, elemwise_add, etc.
    bf16_use_fp32_params: bool
        currently only bn use this param as True, since bf16 bn only accept bf16 data with fp32 mean/var/scale/shift
    rtol: float
        the relative threshold
    atol: float
        the absolute threshold
    etol: float 
        The error rate threshold, allow a small amount of value not consistent between bf16 and fp32
    """
    if not isinstance(data_shape, tuple):
        data_shape = tuple(data_shape)
    data_range = (0.0, 10.0)
    data_list_fp32 = list()
    data_list_bf16 = list()
    for i in range(num_input_data):
        data_list_fp32.append(mx.nd.random.uniform(low=data_range[0], high=data_range[1], shape=data_shape))
        data_list_bf16.append(mx.nd.amp_cast(data_list_fp32[i], dtype=bfloat16))

    arg_shapes, _, aux_shapes = sym_fp32.infer_shape(data=data_shape)
    arg_names = sym_fp32.list_arguments()
    aux_names = sym_fp32.list_auxiliary_states()

    exe_fp32 = sym_fp32.simple_bind(ctx=mx.cpu(), data=data_shape)

    arg_params_fp32 = {}
    aux_params_fp32 = {}
    type_dict = {}
    for i, arg_name in enumerate(arg_names):
        if i < num_input_data:
            exe_fp32.arg_dict[arg_name][:] = data_list_fp32[i]
            continue
        arg_params_fp32[arg_name] = mx.nd.random.uniform(low=data_range[0], high=data_range[1], shape=arg_shapes[i])
        exe_fp32.arg_dict[arg_name][:] = arg_params_fp32[arg_name]
        # specify the dtype of arguments
        if not bf16_use_fp32_params:
            type_dict.update({arg_name: bfloat16})

    for i, aux_name in enumerate(aux_names):
        aux_params_fp32[aux_name] = mx.nd.random.uniform(low=data_range[0], high=data_range[1], shape=aux_shapes[i])
        exe_fp32.aux_dict[aux_name][:] = aux_params_fp32[aux_name]

    output_fp32 = exe_fp32.forward()[0]

    exe_bf16 = sym_bf16.simple_bind(ctx=mx.cpu(), data=data_shape, type_dict=type_dict)

    arg_params_bf16 = {}
    aux_params_bf16 = {}
    for i, arg_name in enumerate(arg_names):
        if i < num_input_data:
            exe_bf16.arg_dict[arg_name][:] = data_list_bf16[i]
            continue

        if bf16_use_fp32_params:
            exe_bf16.arg_dict[arg_name][:] = arg_params_fp32[arg_name]
        else:
            exe_bf16.arg_dict[arg_name][:] = mx.nd.amp_cast(arg_params_fp32[arg_name], dtype=bfloat16)
    
    for aux_name in aux_names:
        if bf16_use_fp32_params:
            exe_bf16.aux_dict[aux_name][:] = aux_params_fp32[aux_name]
        else:
            exe_bf16.aux_dict[aux_name][:] = mx.nd.amp_cast(aux_params_fp32[aux_name], dtype=bfloat16)

    output_bf16 = exe_bf16.forward()[0]
    output_bf16.wait_to_read()
    output_bf16_2_fp32 = mx.nd.amp_cast(output_bf16, dtype="float32")
    assert_almost_equal_with_err(output_bf16_2_fp32, output_fp32, rtol=rtol, atol=atol, etol=etol)

@with_seed()
def test_bf16_bn():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)

    bn_params = {"eps": 2e-05, "fix_gamma": False, "use_global_stats": True, "name": "bn"}
    bn_fp32 = mx.sym.BatchNorm(data_sym_fp32, **bn_params)

    bn_bf16 = mx.sym.BatchNorm(data_sym_bf16, **bn_params)
    check_operator_accuracy(sym_fp32=bn_fp32, sym_bf16=bn_bf16, data_shape=(3, 32, 28, 28), bf16_use_fp32_params=True, etol=1e-2)
    check_operator_accuracy(sym_fp32=bn_fp32, sym_bf16=bn_bf16, data_shape=(32, 16, 64, 64), bf16_use_fp32_params=True, etol=1e-2)

@with_seed()
def test_bf16_conv():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)

    conv_params = {"kernel": (3, 3), "num_filter": 128, "pad": (1, 1), "stride": (1, 1), "no_bias": True, "name": "conv"}
    conv_fp32 = mx.sym.Convolution(data_sym_fp32, **conv_params)
    conv_bf16 = mx.sym.Convolution(data_sym_bf16, **conv_params)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(3, 32, 28, 28), bf16_use_fp32_params=False)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(128, 56, 14, 14), bf16_use_fp32_params=False)

    conv_params = {"kernel": (1, 1), "num_filter": 32, "pad": (0, 0), "stride": (1, 1), "no_bias": False, "name": "conv"}
    conv_fp32 = mx.sym.Convolution(data_sym_fp32, **conv_params)
    conv_bf16 = mx.sym.Convolution(data_sym_bf16, **conv_params)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(3, 32, 28, 28), bf16_use_fp32_params=False)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(128, 56, 14, 14), bf16_use_fp32_params=False)

@with_seed()
def test_bf16_fc():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)

    fc_params = {"num_hidden": 10, "no_bias": True, "flatten": True, "name": "fc"}
    fc_fp32 = mx.sym.FullyConnected(data_sym_fp32, **fc_params)
    fc_bf16 = mx.sym.FullyConnected(data_sym_bf16, **fc_params)
    check_operator_accuracy(fc_fp32, fc_bf16, data_shape=(3, 3, 16, 16), bf16_use_fp32_params=False)

    fc_params = {"num_hidden": 10, "no_bias": False, "flatten": False, "name": "fc"}
    fc_fp32 = mx.sym.FullyConnected(data_sym_fp32, **fc_params)
    fc_bf16 = mx.sym.FullyConnected(data_sym_bf16, **fc_params)
    check_operator_accuracy(fc_fp32, fc_bf16, data_shape=(3, 3, 16, 16), bf16_use_fp32_params=False)

@with_seed()
def test_bf16_pooling():
    pool_params = {"kernel": (3, 3), "stride": (1, 1), "pad": (0, 0), "name": "pool"}
    data_shapes = [(3, 16, 28, 28), (3, 32, 7, 7)]
    pool_types = ["max", "avg"]
    pool_conventions = ["full", "valid"]
    for new_params in itertools.product(data_shapes, pool_types, pool_conventions):
        pool_params.update({"pool_type": new_params[1], "pooling_convention": new_params[2]})
        
        data_sym_fp32 = mx.sym.Variable(name='data')
        data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)
        pool_fp32 = mx.sym.Pooling(data_sym_fp32, **pool_params)
        pool_bf16 = mx.sym.Pooling(data_sym_bf16, **pool_params)
        check_operator_accuracy(pool_fp32, pool_bf16, data_shape=new_params[0], bf16_use_fp32_params=False)

@with_seed()
def test_bf16_activation():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)

    dshapes = [(3, 16), (3, 16, 16), (3, 3, 16, 16)]
    act_types = ['relu', 'sigmoid', 'tanh']
    for data_shape, act_type in itertools.product(dshapes, act_types):
        act_fp32 = mx.sym.Activation(data_sym_fp32, act_type=act_type)
        act_bf16 = mx.sym.Activation(data_sym_bf16, act_type=act_type)

        check_operator_accuracy(act_fp32, act_bf16, data_shape, bf16_use_fp32_params=True)

@with_seed()
def test_bf16_elemwiseadd():
    dshape = rand_shape_nd(4)

    a_sym_fp32 = mx.sym.Variable("data")
    b_sym_fp32 = mx.sym.Variable("data_1")
    sym_fp32 = mx.sym.elemwise_add(a_sym_fp32, b_sym_fp32)

    a_sym_bf16 = mx.sym.Variable("data", dtype=bfloat16)
    b_sym_bf16 = mx.sym.Variable("data_1", dtype=bfloat16)
    sym_bf16 = mx.sym.elemwise_add(a_sym_bf16, b_sym_bf16)

    check_operator_accuracy(sym_fp32, sym_bf16, dshape, num_input_data=2, bf16_use_fp32_params=True)

@unittest.skip("env dependent, need check further.")
@with_seed()
def test_bf16_concat():
    dshape = rand_shape_nd(4)
    a_shape = tuple(dshape)
    b_shape = tuple(dshape)

    a_sym_fp32 = mx.sym.Variable("data", shape=a_shape)
    b_sym_fp32 = mx.sym.Variable("data_1", shape=b_shape)

    a_sym_bf16 = mx.sym.Variable("data", dtype=bfloat16, shape=a_shape)
    b_sym_bf16 = mx.sym.Variable("data_1", dtype=bfloat16, shape=b_shape)
    for axis in range(0, 4):
        print(axis, a_shape)
        concat_sym_fp32 = mx.sym.concat(a_sym_fp32, b_sym_fp32, dim=axis)
        concat_sym_bf16 = mx.sym.concat(a_sym_bf16, b_sym_bf16, dim=axis)

        check_operator_accuracy(concat_sym_fp32, concat_sym_bf16, dshape, num_input_data=2, bf16_use_fp32_params=True)

@with_seed()
def test_bf16_abs():
    dshapes = [(16,), (3, 16), (3, 16, 16), (3, 16, 16, 16)]
    for data_shape in dshapes:
        data_sym_fp32 = mx.sym.Variable(name='data')
        data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)
        sym_fp32 = mx.sym.abs(data_sym_fp32)
        sym_bf16 = mx.sym.abs(data_sym_bf16)
        
        check_operator_accuracy(sym_fp32, sym_bf16, data_shape, bf16_use_fp32_params=True)

@with_seed()
def test_bf16_sqrt():
    dshapes = [(16,), (3, 16), (3, 16, 16), (3, 16, 16, 16)]
    for data_shape in dshapes:
        data_sym_fp32 = mx.sym.Variable(name='data')
        data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)
        sym_bf16 = mx.sym.sqrt(data_sym_bf16)
        sym_fp32 = mx.sym.sqrt(data_sym_fp32)

        check_operator_accuracy(sym_fp32, sym_bf16, data_shape, bf16_use_fp32_params=True)

@with_seed()
def test_bf16_square():
    dshapes = [(16,), (3, 16), (3, 16, 16), (3, 16, 16, 16)]
    for data_shape in dshapes:
        data_sym_fp32 = mx.sym.Variable(name='data')
        data_sym_bf16 = mx.sym.Variable(name='data', dtype=bfloat16)
        sym_bf16 = mx.sym.square(data_sym_bf16)
        sym_fp32 = mx.sym.square(data_sym_fp32)

        check_operator_accuracy(sym_fp32, sym_bf16, data_shape, bf16_use_fp32_params=True)

@with_seed()
def test_bf16_flatten_slice_after_conv():
    data_fp32 = mx.symbol.Variable('data')
    data_bf16 = mx.symbol.Variable('data', dtype=bfloat16)

    conv_fp32= mx.symbol.Convolution(data=data_fp32, name='conv', num_filter=64, kernel=(3,3), stride=(1,1))
    flatten_fp32 = mx.symbol.flatten(data=conv_fp32)
    slice_fp32 = mx.symbol.slice(data=flatten_fp32, begin=0, end=1)

    conv_bf16= mx.symbol.Convolution(data=data_bf16, name='conv', num_filter=64, kernel=(3,3), stride=(1,1))
    flatten_bf16 = mx.symbol.flatten(data=conv_bf16)
    slice_bf16 = mx.symbol.slice(data=flatten_bf16, begin=0, end=1)

    shape = (2, 16, 16, 16)
    check_operator_accuracy(slice_fp32, slice_bf16, shape, bf16_use_fp32_params=False)

def test_bf16_fallback():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16=mx.sym.Variable(name='data', dtype=bfloat16)

    bn_params = {"eps": 2e-05, "fix_gamma": False, "use_global_stats": True, "name": "bn"}
    bn_fp32 = mx.sym.BatchNorm(data_sym_fp32, **bn_params)
    bn_bf16=mx.sym.BatchNorm(data_sym_bf16, **bn_params)
    check_operator_accuracy(sym_fp32=bn_fp32, sym_bf16=bn_bf16, data_shape=(3, 32, 28, 28, 3), bf16_use_fp32_params=True, etol=1e-2)

    conv_params = {"kernel": (3, 3, 3), "num_filter": 128, "pad": (1, 1, 1), "stride": (1, 1, 1), "no_bias": True, "name": "conv"}
    conv_fp32 = mx.sym.Convolution(data_sym_fp32, **conv_params)
    conv_bf16 = mx.sym.Convolution(data_sym_bf16, **conv_params)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(3, 32, 28, 28, 4), bf16_use_fp32_params=False)

if __name__ == '__main__':
    import nose
    nose.runmodule()
