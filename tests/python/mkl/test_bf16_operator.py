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
import mxnet.contrib.amp as amp
from nose.tools import assert_raises
from mxnet.test_utils import set_default_context, download_model, same_symbol_structure, assert_almost_equal
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
from mxnet.contrib.amp import amp
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed

bfloat16 = np.dtype([('bfloat16', np.uint16)])

def check_operator_accuracy(sym_fp32, sym_bf16, data_shape, bf16_use_fp32_params=False):
    Batch = collections.namedtuple('Batch',['data'])
    data_range = (0.0, 10.0)
    data_fp32 = mx.nd.random.uniform(low=data_range[0], high=data_range[1], shape=data_shape)
    data_bf16 = mx.nd.amp_cast(data_fp32, dtype=bfloat16)

    arg_shapes, _, aux_shapes = sym_fp32.infer_shape(data=data_shape)
    arg_names = sym_fp32.list_arguments()
    aux_names = sym_fp32.list_auxiliary_states()

    arg_params = dict()
    aux_params = dict()
    for i, arg in enumerate(arg_names):
        if i == 0: continue  # exclude data
        arg_params[arg] = mx.nd.random.uniform(low=data_range[0], high=data_range[1], shape=arg_shapes[i])
    for i, aux in enumerate(aux_names):
        aux_params[aux] = mx.nd.random.uniform(low=data_range[0], high=data_range[1], shape=aux_shapes[i])

    exe_fp32 = mx.mod.Module(symbol=sym_fp32, label_names=None, context=mx.cpu())
    exe_fp32.bind(data_shapes=[('data', data_shape)])
    exe_fp32.set_params(arg_params=arg_params, aux_params=aux_params)
    exe_fp32.forward(Batch([data_fp32]), is_train=False)
    output_fp32 = exe_fp32.get_outputs()[0]

    exe_bf16 = mx.mod.Module(symbol=sym_bf16, label_names=None, context=mx.cpu())
    exe_bf16.bind(data_shapes=[('data', data_shape)])
    if bf16_use_fp32_params:
        exe_bf16.set_params(arg_params=arg_params, aux_params=aux_params)
    else:
        arg_params_bf16 = dict()
        aux_params_bf16 = dict()
        for k, v in arg_params.items():
            arg_params_bf16[k] = mx.nd.amp_cast(v, dtype=bfloat16)
        for k, v in aux_params.items():
            aux_params_bf16[k] = mx.nd.amp_cast(v, dtype=bfloat16)
        exe_bf16.set_params(arg_params=arg_params_bf16, aux_params=aux_params_bf16)
    exe_bf16.forward(Batch([data_bf16]), is_train=False)
    output_bf16 = exe_bf16.get_outputs()[0]
    output_bf16_2_fp32 = mx.nd.amp_cast(output_bf16, dtype="float32")
    assert_almost_equal(output_bf16_2_fp32, output_fp32, rtol=1e-1, atol=5e-1)

@with_seed()
def test_bf16_bn():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16=mx.sym.Variable(name='data', dtype=bfloat16)

    bn_params = {"eps": 2e-05, "fix_gamma": False, "use_global_stats": True, "name": "bn"}
    bn_fp32 = mx.sym.BatchNorm(data_sym_fp32, **bn_params)
    bn_bf16=mx.sym.BatchNorm(data_sym_bf16, **bn_params)
    check_operator_accuracy(sym_fp32=bn_fp32, sym_bf16=bn_bf16, data_shape=(3, 32, 28, 28, 5), bf16_use_fp32_params=True)
    check_operator_accuracy(sym_fp32=bn_fp32, sym_bf16=bn_bf16, data_shape=(32, 16, 64, 64), bf16_use_fp32_params=True)

@with_seed()
def test_bf16_conv():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16=mx.sym.Variable(name='data', dtype=bfloat16)

    conv_params = {"kernel": (3, 3), "num_filter": 128, "pad": (1, 1), "stride": (1, 1), "no_bias": True, "name": "conv"}
    conv_fp32 = mx.sym.Convolution(data_sym_fp32, **conv_params)
    conv_bf16 = mx.sym.Convolution(data_sym_bf16, **conv_params)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(3, 32, 28, 28), bf16_use_fp32_params=False)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(128, 56, 14, 14), bf16_use_fp32_params=False)

    conv_params = {"kernel": (1, 1), "num_filter": 32, "pad": (0, 0), "stride": (1, 1), "no_bias": False, "name": "conv"}
    conv_fp32 = mx.sym.Convolution(data_sym_fp32, **conv_params)
    conv_bf16=mx.sym.Convolution(data_sym_bf16, **conv_params)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(3, 32, 28, 28), bf16_use_fp32_params=False)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(128, 56, 14, 14), bf16_use_fp32_params=False)

@with_seed()
def test_bf16_fallback():
    data_sym_fp32 = mx.sym.Variable(name='data')
    data_sym_bf16=mx.sym.Variable(name='data', dtype=bfloat16)

    bn_params = {"eps": 2e-05, "fix_gamma": False, "use_global_stats": True, "name": "bn"}
    bn_fp32 = mx.sym.BatchNorm(data_sym_fp32, **bn_params)
    bn_bf16=mx.sym.BatchNorm(data_sym_bf16, **bn_params)
    check_operator_accuracy(sym_fp32=bn_fp32, sym_bf16=bn_bf16, data_shape=(3, 32, 28, 28, 3), bf16_use_fp32_params=True)

    conv_params = {"kernel": (3, 3, 3), "num_filter": 128, "pad": (1, 1, 1), "stride": (1, 1, 1), "no_bias": True, "name": "conv"}
    conv_fp32 = mx.sym.Convolution(data_sym_fp32, **conv_params)
    conv_bf16 = mx.sym.Convolution(data_sym_bf16, **conv_params)
    check_operator_accuracy(sym_fp32=conv_fp32, sym_bf16=conv_bf16, data_shape=(3, 32, 28, 28, 4), bf16_use_fp32_params=False)

if __name__ == '__main__':
    import nose
    nose.runmodule()
