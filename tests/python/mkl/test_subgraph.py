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

import sys
import os
import mxnet as mx
import numpy as np
import unittest
import ctypes
from mxnet.symbol import Symbol
from importlib import import_module
from numpy.testing import assert_allclose
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str
from mxnet.test_utils import DummyIter
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../unittest/'))
from common import with_seed
from mxnet.test_utils import assert_almost_equal, assert_almost_equal_with_err
import itertools
import pytest
import tempfile

OP_NAME='op_name'
QUANTIZED_OP_NAME='quantized_op_name'
SG_PASS_NAME='MKLDNN'
QUANTIZE_SG_PASS_NAME='MKLDNN_QUANTIZE'
config =  {
  'conv': {
    OP_NAME: 'sg_mkldnn_conv',
    QUANTIZED_OP_NAME: 'quantized_sg_mkldnn_conv'
  },
  'fc': {
    OP_NAME: 'sg_mkldnn_fully_connected',
    QUANTIZED_OP_NAME: 'quantized_sg_mkldnn_fully_connected'
  }
}

DATA_SHAPE=[(64, 4, 10, 10), (4, 3, 24, 24), (1, 16, 32, 32)]
fc_post_ops_list=['relu', 'sigmoid', 'tanh', 'softrelu',
                  'square', 'square_root', 'abs', 'exp', 'bounded_relu']

def check_qsym_calibrated(qsym, out_type, name='conv'):
  quantized_op_name = 'quantized_' + name
  assert ''.join(qsym.attr_dict().keys()).find(quantized_op_name) != -1
  for k, v in qsym.attr_dict().items():
    if k.find('_quantize') != -1:
      assert v['out_type'] == out_type
    if k.find(quantized_op_name) != -1:
      if quantized_op_name.startswith("quantized_sg_mkldnn_fully_connected") and 'enable_float_output' in v:
        continue
      assert 'min_calib_range' in v
      assert 'max_calib_range' in v

def check_qsym_scale_align(qsym):
  assert ''.join(qsym.attr_dict().keys()).find('quantized_sg_mkldnn_conv') != -1
  init = False
  for k, v in qsym.attr_dict().items():
    if k.find('quantized_sg_mkldnn_conv') != -1:
      assert 'min_calib_range' in v
      assert 'max_calib_range' in v
      if not init:
        min_calib_range = v['min_calib_range']
        max_calib_range = v['max_calib_range']
        init = True
      else:
        assert min_calib_range == v['min_calib_range']
        assert max_calib_range == v['max_calib_range']


def check_qsym_gluon_forward(path, qsym, qarg_params, qaux_params, data_shape):
  # save qsym to JSON file
  _, json_path = tempfile.mkstemp(suffix='-symbol.json', dir=path)
  params_path = json_path.replace('-symbol.json', '-0000.params')
  qsym.save(json_path)
  # save params
  save_dict = {('arg:%s' % k): v.as_in_context(mx.current_context()) for k, v in qarg_params.items()}
  save_dict.update({('aux:%s' % k): v.as_in_context(mx.current_context()) for k, v in qaux_params.items()})
  mx.nd.save(params_path, save_dict)
  # load back with SymbolBlock
  net = mx.gluon.SymbolBlock.imports(json_path, ['data'], params_path)
  net.collect_params().reset_ctx(ctx = mx.current_context())
  net.hybridize()

  data = mx.random.uniform(-1.0, 1.0, shape=data_shape)
  net(data)

class CalibIter(mx.io.DataIter):
    def __init__(self, batch, data_shape, batch_size):
        super(CalibIter, self).__init__(batch_size)
        self.data_shape = data_shape
        self.label_shape = (batch_size,)
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = []
        self.batch = batch

    def __iter__(self):
        yield self.batch


def check_neg_fusion(syms, attrs_name=None, excluded_attrs=None,
                     date_shape=(4,4,10,10), name='conv'):
  op_name = config[name][OP_NAME]

  for sym, attrs, excluded_attr in zip(syms, attrs_name, excluded_attrs):
    sym_sg = sym.get_backend_symbol(SG_PASS_NAME)
    exe_sg = sym_sg.simple_bind(mx.cpu(), data=date_shape, grad_req='null')

    attrs_dict = sym_sg.attr_dict()
    for k, v in attrs_dict.items():
      if k.find(op_name) != -1:
        for attr in attrs:
          assert v[attr] == 'true'
        for exc_attr in excluded_attr:
          assert exc_attr not in v.keys()

def head_symbol(data_shape):
  data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
  weight = mx.symbol.Variable('weight', dtype='float32')
  return data, weight


# conv + bn fusion case
def conv_bn(no_bias, data_shape):
  attr = {'conv': {'with_bn': 'true'}}
  data, weight = head_symbol(data_shape)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  bn1 = mx.symbol.BatchNorm(data=conv, name="bn1")
  return bn1, attr

# conv + act fusion case
def conv_act(no_bias, data_shape, alg):
  attr = {'conv': {'with_act': 'true'}}
  data, weight = head_symbol(data_shape)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  if alg == "relu6":
    relu = mx.symbol.clip(data=conv, name='relu6', a_min=0, a_max=6)
  elif alg == "leakyrelu":
    relu = mx.symbol.LeakyReLU(data=conv, slope=0.25, act_type='leaky')
  elif alg == "gelu":
    relu = mx.symbol.LeakyReLU(data=conv, act_type='gelu')
  else:
    relu = mx.symbol.Activation(data=conv, name=alg, act_type=alg)
  return relu, attr

# conv + act + sum fusion case
def conv_act_sum(no_bias, data_shape, alg):
  attr = {'conv': {'with_act': 'true', 'with_sum': 'true'}}
  data, weight = head_symbol(data_shape)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  if alg == "relu6":
    relu = mx.symbol.clip(data=conv, name='relu6', a_min=0, a_max=6)
  elif alg == "leakyrelu":
    relu = mx.symbol.LeakyReLU(data=conv, slope=0.25, act_type='leaky')
  elif alg == "gelu":
    relu = mx.symbol.LeakyReLU(data=conv, act_type='gelu')
  else:
    relu = mx.symbol.Activation(data=conv, name=alg, act_type=alg)
  conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  sum = relu + conv1
  return sum, attr


# mobilenetv2 case
def mobilenetv2_struct(data_shape):
  attr = {'sg_mkldnn_conv_bn_0' : {'with_bn': 'true'}}
  data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
  weight1 = mx.symbol.Variable('conv1_weight', dtype='float32')
  weight2 = mx.symbol.Variable('conv2_weight', dtype='float32')
  conv1 = mx.symbol.Convolution(data=data, weight=weight1, name='conv1', num_filter=64,
                               kernel=(1, 1), stride=(1, 1), no_bias=True)
  bn1 = mx.symbol.BatchNorm(data=conv1, name="bn1")
  conv2 = mx.symbol.Convolution(data=bn1, weight=weight2, name='conv2', num_filter=64,
                               kernel=(1, 1), stride=(1, 1), no_bias=True)
  bn2 = mx.symbol.BatchNorm(data=conv2, name="bn2")
  sum = bn1 + bn2
  return sum, attr

def single_fc(no_bias, data_shape, flatten=True):
  attr = {'fc': {}}
  data, weight = head_symbol(data_shape)
  fc = mx.symbol.FullyConnected(name='fc', data=data, weight=weight, num_hidden=64,
                                no_bias=no_bias, flatten=flatten)
  return fc, attr

# fc + eltwise fusion case
def fc_eltwise(no_bias, data_shape, flatten=True, alg='relu'):
  assert alg in fc_post_ops_list

  attr = {'fc': {'with_eltwise': 'true'}}
  data, weight = head_symbol(data_shape)
  fc = mx.symbol.FullyConnected(name='fc', data=data, weight=weight, num_hidden=64,
                                no_bias=no_bias, flatten=flatten)
  if alg in ['relu', 'sigmoid', 'tanh', 'softrelu']:
    sym = mx.symbol.Activation(data=fc, name='act', act_type=alg)
  elif alg == 'square':
    sym = mx.symbol.square(data=fc, name='square')
  elif alg == 'square_root':
    sym = mx.symbol.sqrt(data=fc, name='sqrt')
  elif alg == 'abs':
    sym = mx.symbol.abs(data=fc, name='abs')
  elif alg == 'exp':
    sym = mx.symbol.exp(data=fc, name='exp')
  else:
    sym = mx.symbol.clip(data=fc, name='bounded_relu', a_min=0, a_max=1.0)

  return sym, attr


def test_float64_fallback():
    sym = mx.sym.FullyConnected(
        mx.sym.Variable('in'),
        mx.sym.Variable('w'),
        mx.sym.Variable('b'),
        num_hidden=2
    )

    dtype = 'float64'
    ex = sym.bind(mx.cpu(),
                  {
        'in': mx.nd.array([[2, 3, 4]], dtype=dtype),
        'w': mx.nd.array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
        'b': mx.nd.array([7, 8], dtype=dtype)
    },
        args_grad=None,
        grad_req='write'
    )
    ex.forward()
    ex.outputs[0].wait_to_read()
