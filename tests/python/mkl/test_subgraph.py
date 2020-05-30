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



def check_qsym_forward(qsym, qarg_params, qaux_params, batch, data_shape):
def check_qsym_dummy_forward(qsym, batch, data_shape):
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

def tail_neg_symbol(sym1, sym2):
  fc1 = mx.sym.FullyConnected(data=sym1, num_hidden=10, flatten=True, name='fc1')
  fc2 = mx.sym.FullyConnected(data=sym2, num_hidden=10, flatten=True, name='fc2')
  concat = mx.sym.Concat(*[fc1, fc2], name="concat")
  sym = mx.sym.SoftmaxOutput(data=concat, name='softmax')
  return sym

# conv + bn can't be fusion case
# eg.1
# conv --------- > bn
#  |
#  |
#  -------------> [custom op]
def neg_conv_bn(data_shape):
  syms = []
  attrs = []
  excluded_attrs = []
  data, weight = head_symbol(data_shape)

  # eg.1 ([custom op] = pool)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn1 = mx.symbol.BatchNorm(data=conv, name="bn1")
  pool = mx.sym.Pooling(data=conv, kernel=(4, 4), pool_type='avg', name='pool')
  sym = tail_neg_symbol(bn1, pool)

  syms.append(sym)
  attrs.append([])
  excluded_attrs.append([])
  return syms, attrs, excluded_attrs

# conv + relu can't be fusion case
# eg.1
# conv -----------> relu
#  |
#  |
#  ---------------> [custom op]
def neg_conv_relu(data_shape):
  syms = []
  attrs = []
  excluded_attrs = []
  data, weight = head_symbol(data_shape)

  # eg.1 ([custom op] = pool)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  relu = mx.symbol.Activation(data=conv, name='relu', act_type="relu")
  pool = mx.sym.Pooling(data=conv, kernel=(4, 4), pool_type='avg', name='pool')
  sym = tail_neg_symbol(relu, pool)

  syms.append(sym)
  attrs.append([])
  excluded_attrs.append([])
  return syms, attrs, excluded_attrs

# conv + add can't be fusion case
# eg.1
#  ---------------> [custom op]
#  |
#  |
# conv -----------> add
#                   |
#                   |
# added ------------>
def neg_conv_add(data_shape):
  syms = []
  attrs = []
  excluded_attrs = []
  val = mx.symbol.Variable('addval')
  data, weight = head_symbol(data_shape)

  # eg.1 ([custom op] = pool, [added op] = val)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  sum1 = conv + val
  pool = mx.sym.Pooling(data=conv, kernel=(4, 4), pool_type='avg', name='pool')
  sym = tail_neg_symbol(sum1, pool)

  syms.append(sym)
  attrs.append([])
  excluded_attrs.append('with_sum')
  return syms, attrs, excluded_attrs

# conv + bn + relu can't be fusion case
# eg.1
#   --------------> [custom op]
#   |
# conv -----------> bn -----------> relu
#
# eg.2
#                   --------------> [custom op]
#                   |
# conv -----------> bn -----------> relu
def neg_conv_bn_relu(data_shape):
  syms = []
  attrs = []
  excluded_attrs = []
  data, weight = head_symbol(data_shape)

  # eg.1 ([custom op] = pool11)
  conv11 = mx.symbol.Convolution(data=data, weight=weight, name='conv11', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn11 = mx.symbol.BatchNorm(data=conv11, name="bn11")
  relu11 = mx.symbol.Activation(data=bn11, name='relu11', act_type="relu")
  pool11 = mx.sym.Pooling(data=conv11, kernel=(4, 4), pool_type='avg', name='pool11')
  sym1 = tail_neg_symbol(relu11, pool11)

  syms.append(sym1)
  attrs.append([])
  excluded_attrs.append([])

  # eg.2 ([custom op] = pool)
  conv21 = mx.symbol.Convolution(data=data, weight=weight, name='conv21', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn21 = mx.symbol.BatchNorm(data=conv21, name="bn21")
  relu21 = mx.symbol.Activation(data=bn21, name='relu21', act_type="relu")
  pool21 = mx.sym.Pooling(data=bn21, kernel=(4, 4), pool_type='avg', name='pool21')
  sym2 = tail_neg_symbol(relu21, pool21)

  syms.append(sym2)
  attrs.append(['with_bn'])
  excluded_attrs.append(['with_act'])
  return syms, attrs, excluded_attrs

# conv + bn + add + relu can't be fusion case
# eg.1
#   --------------> [custom op]
#   |
# conv -----------> bn -----------> add -----------> relu
#
# eg.2
#                    -------------> [custom op]
#                    |
# conv -----------> bn -----------> add -----------> relu
#
# eg.3
#                                    --------------> [custom op]
#                                    |
# conv -----------> bn -----------> add -----------> relu
def neg_conv_bn_add_relu(data_shape):
  syms = []
  attrs = []
  excluded_attrs = []
  addVal = mx.symbol.Variable('addval')
  data, weight = head_symbol(data_shape)

  # eg.1
  conv11 = mx.symbol.Convolution(data=data, weight=weight, name='conv11', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn11 = mx.symbol.BatchNorm(data=conv11, name="bn11")
  sum11 = bn11 + addVal
  relu11 = mx.symbol.Activation(data=sum11, name='relu11', act_type="relu")
  pool11 = mx.sym.Pooling(data=conv11, kernel=(4, 4), pool_type='avg', name='pool11')
  sym1 = tail_neg_symbol(relu11, pool11)

  syms.append(sym1)
  attrs.append([])
  excluded_attrs.append(['with_sum', 'with_postsum_act', 'with_bn'])

  # eg.2
  conv21 = mx.symbol.Convolution(data=data, weight=weight, name='conv21', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn21 = mx.symbol.BatchNorm(data=conv21, name="bn21")
  sum21 = bn21 + addVal
  relu21 = mx.symbol.Activation(data=sum21, name='relu21', act_type="relu")
  pool21 = mx.sym.Pooling(data=bn21, kernel=(4, 4), pool_type='avg', name='pool21')
  sym2 = tail_neg_symbol(relu21, pool21)

  syms.append(sym2)
  attrs.append(['with_bn'])
  excluded_attrs.append(['with_sum', 'with_postsum_act'])

  # eg.3
  conv31 = mx.symbol.Convolution(data=data, weight=weight, name='conv31', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn31 = mx.symbol.BatchNorm(data=conv31, name="bn31")
  sum31 = bn31 + addVal
  relu31 = mx.symbol.Activation(data=sum31, name='relu31', act_type="relu")
  pool31 = mx.sym.Pooling(data=sum31, kernel=(4, 4), pool_type='avg', name='pool31')
  sym3 = tail_neg_symbol(relu31, pool31)

  syms.append(sym3)
  attrs.append(['with_bn', 'with_sum'])
  excluded_attrs.append(['with_postsum_act'])
  return syms, attrs, excluded_attrs

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

# fc + relu can't be fusion case
# eg.1
# fc -----------> relu
#  |
#  |
#  ---------------> [custom op]
def neg_fc_relu(no_bias, data_shape, flatten=True):
  syms = []
  attrs = []
  excluded_attrs = []
  data, weight = head_symbol(data_shape)

  # eg.1 ([custom op] = pool)
  fc = mx.symbol.FullyConnected(name='fc', data=data, weight=weight, num_hidden=64,
                                no_bias=no_bias, flatten=flatten)
  relu = mx.symbol.Activation(data=fc, name='relu', act_type="relu")
  sigmoid = mx.symbol.Activation(data=fc, name='sigmoid', act_type="sigmoid")
  sym = tail_neg_symbol(relu, sigmoid)

  syms.append(sym)
  attrs.append([])
  excluded_attrs.append([])
  return syms, attrs, excluded_attrs

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn(data_shape):
    syms, attrs, excluded_attrs = neg_conv_bn(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_relu(data_shape):
    syms, attrs, excluded_attrs = neg_conv_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_add(data_shape):
    syms, attrs, excluded_attrs = neg_conv_add(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_relu(data_shape):
    syms, attrs, excluded_attrs = neg_conv_bn_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_add_relu(data_shape):
    syms, attrs, excluded_attrs = neg_conv_bn_add_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_single_fc(data_shape, no_bias, flatten, tmpdir):
    syms, attrs = single_fc(no_bias, data_shape, flatten)
    check_fusion(syms, data_shape, attrs, str(tmpdir), check_quantization=flatten)

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
@pytest.mark.parametrize('alg', fc_post_ops_list)
def test_fc_eltwise(data_shape, no_bias, flatten, alg, tmpdir):
    syms, attrs = fc_eltwise(no_bias, data_shape, flatten, alg)
    check_fusion(syms, data_shape, attrs, str(tmpdir), check_quantization=flatten)

@with_seed()
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_neg_fc_relu(data_shape, no_bias, flatten):
    syms, attrs, excluded_attrs = neg_fc_relu(no_bias, data_shape, flatten)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape, name='fc')

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


@with_seed()
@pytest.mark.parametrize('data_min,data_max,weight_min,weight_max', [
    (-1, 1, 0, 0),
    (-1, 1, -1e-6, +1e-6),
    (0, 0, 1, 1),
    (-1e-6, +1e-6, -1, 1),
    (-1e-6, +1e-6, -1e-6, +1e-6),
    (0, 0, 0, 0)
])
def test_quantized_conv_bias_overflow(data_min, data_max, weight_min, weight_max):
  data_shape = (1, 32, 2, 2)
  data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
  weight = mx.symbol.Variable('weight', dtype='float32')
  bias = mx.symbol.Variable('bias', dtype='float32')
  sym = mx.symbol.Convolution(data=data, weight=weight, bias=bias, name='conv', num_filter=64,
                               kernel=(1, 1), stride=(1, 1))
  data_nd = mx.random.uniform(data_min, data_max, shape=data_shape, ctx=mx.cpu())
  weight_nd = mx.random.uniform(weight_min, weight_max, shape=[64, 32, 1, 1], ctx=mx.cpu())
  bias_nd = mx.random.uniform(-1, +1, shape=[64], ctx=mx.cpu())
  arg_params = {
      'data': data_nd,
      'weight': weight_nd,
      'bias': bias_nd
  }

  ex = sym.bind(mx.cpu(), arg_params, args_grad=None)
  ex.forward()
  ex.outputs[0].wait_to_read()
  sym_sg = sym.get_backend_symbol(QUANTIZE_SG_PASS_NAME)
  batch = mx.io.DataBatch([data_nd], [])
  calib_data = CalibIter(batch, data_shape, 1)
  qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym_sg,
                                                                   arg_params={
                                                                       'weight': weight_nd,
                                                                       'bias': bias_nd
                                                                   },
                                                                   aux_params={},
                                                                   ctx=mx.cpu(),
                                                                   excluded_sym_names=None,
                                                                   excluded_op_names=None,
                                                                   quantized_dtype='int8',
                                                                   calib_mode='naive',
                                                                   calib_data=calib_data,
                                                                   label_names=None,
                                                                   num_calib_examples=1,
                                                                   quantize_mode='full')
  qsym = qsym.get_backend_symbol(QUANTIZE_SG_PASS_NAME)
  qarg_params['data'] = data_nd
  qex = qsym.bind(mx.cpu(), qarg_params, args_grad=None)
  qex.forward()
  qex.outputs[0].wait_to_read()
  assert_almost_equal_with_err(ex.outputs[0].asnumpy(), qex.outputs[0].asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)

@with_seed()
@pytest.mark.parametrize('data_min,data_max,weight_min,weight_max', [
    (-1, 1, 0, 0),
    (-1, 1, -1e-6, +1e-6),
    (0, 0, 1, 1),
    (-1e-6, +1e-6, -1, 1),
    (-1e-6, +1e-6, -1e-6, +1e-6),
    (0, 0, 0, 0)
])
def test_quantized_fc_bias_overflow(data_min, data_max, weight_min, weight_max):
  data_shape = (1, 32)
  data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
  weight = mx.symbol.Variable('weight', dtype='float32')
  bias = mx.symbol.Variable('bias', dtype='float32')
  sym = mx.symbol.FullyConnected(data=data, weight=weight, bias=bias, name='fc', num_hidden=64)
  data_nd = mx.random.uniform(data_min, data_max, shape=data_shape, ctx=mx.cpu())
  weight_nd = mx.random.uniform(weight_min, weight_max, shape=[64, 32], ctx=mx.cpu())
  bias_nd = mx.random.uniform(-1, +1, shape=[64], ctx=mx.cpu())
  arg_params = {
      'data': data_nd,
      'weight': weight_nd,
      'bias': bias_nd
  }

  ex = sym.bind(mx.cpu(), arg_params, args_grad=None)
  ex.forward()
  ex.outputs[0].wait_to_read()
  sym_sg = sym.get_backend_symbol(QUANTIZE_SG_PASS_NAME)
  batch = mx.io.DataBatch([data_nd], [])
  calib_data = CalibIter(batch, data_shape, 1)
  qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym_sg,
                                                                   arg_params={
                                                                       'weight': weight_nd,
                                                                       'bias': bias_nd
                                                                   },
                                                                   aux_params={},
                                                                   ctx=mx.cpu(),
                                                                   excluded_sym_names=None,
                                                                   excluded_op_names=None,
                                                                   quantized_dtype='int8',
                                                                   calib_mode='naive',
                                                                   calib_data=calib_data,
                                                                   label_names=None,
                                                                   num_calib_examples=1,
                                                                   quantize_mode='full')
  qarg_params['data'] = data_nd
  qsym = qsym.get_backend_symbol(QUANTIZE_SG_PASS_NAME)
  qex = qsym.bind(mx.cpu(), qarg_params, args_grad=None)
  qex.forward()
  qex.outputs[0].wait_to_read()
  assert_almost_equal_with_err(ex.outputs[0].asnumpy(), qex.outputs[0].asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)
