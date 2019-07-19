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
from mxnet.module import Module
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
  mod = Module(symbol=qsym, label_names=None, context=mx.current_context())
  mod.bind(for_training=False,
           data_shapes=[('data', data_shape)])
  mod.set_params(qarg_params, qaux_params)
  mod.forward(batch, is_train=False)
  for output in mod.get_outputs():
    output.wait_to_read()
  return mod.get_outputs()

def check_qsym_dummy_forward(qsym, batch, data_shape):
  mod = Module(symbol=qsym, label_names=None, context=mx.current_context())
  mod.bind(for_training=False,
           data_shapes=[('data', data_shape)])
  mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
  mod.forward(batch, is_train=False)
  for output in mod.get_outputs():
    output.wait_to_read()
  return mod.get_outputs()

def check_qsym_gluon_forward(qsym, qarg_params, qaux_params, data_shape):
  # save qsym to JSON file
  qsym.save('quantized-symbol.json')
  # save params
  save_dict = {('arg:%s' % k): v.as_in_context(mx.current_context()) for k, v in qarg_params.items()}
  save_dict.update({('aux:%s' % k): v.as_in_context(mx.current_context()) for k, v in qaux_params.items()})
  mx.nd.save('quantized-0000.params', save_dict)
  # load back with SymbolBlock
  net = mx.gluon.SymbolBlock.imports('quantized-symbol.json', ['data'], 'quantized-0000.params')
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


def check_quantize(sym, data_shape, out_type, name='conv',
                   check_calibration=True, gluon_forward=False, check_scale_align=False):
  if name in config:
    name = config[name][OP_NAME]
  sym_sg = sym.get_backend_symbol(QUANTIZE_SG_PASS_NAME)
  mod = Module(symbol=sym, label_names=None)
  mod.bind(for_training=False,
            data_shapes=[('data', data_shape)])
  mod.init_params(mx.init.Normal(0.5))
  arg_params, aux_params = mod.get_params()

  if out_type == 'uint8':
    data = [mx.random.uniform(0.0, 1.0, shape=shape, ctx=mx.current_context()) for _, shape in mod.data_shapes]
  else:
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=mx.current_context()) for _, shape in mod.data_shapes]
  batch = mx.io.DataBatch(data, [])

  mod.forward(batch, is_train=False)
  for output in mod.get_outputs():
      output.wait_to_read()
  ref_out = mod.get_outputs()

  excluded_sym_names = []
  if mx.current_context() == mx.cpu() and gluon_forward == True:
    excluded_sym_names += ['sg_mkldnn_fully_connected_0']
    excluded_sym_names += ['fc_softmax']

  calib_data = CalibIter(batch, data_shape, 1)

  qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym_sg,
                                                                   arg_params=arg_params,
                                                                   aux_params=aux_params,
                                                                   ctx=mx.current_context(),
                                                                   excluded_sym_names=excluded_sym_names,
                                                                   quantized_dtype=out_type,
                                                                   calib_mode='naive',
                                                                   calib_data=calib_data,
                                                                   calib_layer=None,
                                                                   label_names=None,
                                                                   num_calib_examples=1)
  qsym = qsym.get_backend_symbol(QUANTIZE_SG_PASS_NAME)
  if check_calibration:
    check_qsym_calibrated(qsym, out_type, name=name)
  if check_scale_align:
    check_qsym_scale_align(qsym)
  if gluon_forward == True:
    check_qsym_gluon_forward(qsym, qarg_params, qaux_params, data_shape)
  else:
    quantized_out = check_qsym_forward(qsym, qarg_params, qaux_params, batch, data_shape)
    for i in range(len(ref_out)):
      min_range = mx.nd.min(ref_out[i]).asscalar()
      max_range = mx.nd.max(ref_out[i]).asscalar()
      atol = 0.1 * max(abs(min_range), abs(max_range))
      assert_almost_equal_with_err(quantized_out[i].asnumpy(), ref_out[i].asnumpy(), rtol=0.1, atol=atol, etol=0.2)
    check_qsym_dummy_forward(qsym, batch, data_shape)

@with_seed()
def check_quantize_whole_model_with_forward():
  def check_qsym_forward(qsym, qarg_params, qaux_params, data_shape):
    mod = Module(symbol=qsym, label_names=None, context=mx.current_context())
    mod.bind(for_training=False,
             data_shapes=[('data', data_shape)])
    mod.set_params(qarg_params, qaux_params)
    data = [mx.random.uniform(-1.0, 1.0, shape=shape) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])
    mod.forward(batch, is_train=False)
    for output in mod.get_outputs():
        output.wait_to_read()

  def check_quantize_whole_model(out_type):
    batch_size = 4
    data_shape = (batch_size, 4, 10, 10)
    data = mx.sym.Variable('data')
    conv0 = mx.sym.Convolution(data, kernel=(1, 1), num_filter=16, name='conv0')
    sym = mx.sym.Convolution(conv0, kernel=(1, 1), num_filter=16, name='conv1')
    sym_sg = sym.get_backend_symbol('MKLDNN_QUANTIZE')
    mod = Module(symbol=sym, label_names=None)
    mod.bind(for_training=False,
             data_shapes=[('data', data_shape)])

    mod.init_params(mx.init.Normal(0.5))
    arg_params, aux_params = mod.get_params()

    excluded_sym_names = []

    calib_data = mx.nd.random.uniform(shape=data_shape)
    calib_data = mx.io.NDArrayIter(data=calib_data)
    calib_data = DummyIter(calib_data)
    calib_layer = lambda name: name.endswith('_output')
    qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym_sg,
                                                                     arg_params=arg_params,
                                                                     aux_params=aux_params,
                                                                     ctx=mx.current_context(),
                                                                     excluded_sym_names=excluded_sym_names,
                                                                     quantized_dtype=out_type,
                                                                     calib_mode='naive',
                                                                     calib_data=calib_data,
                                                                     calib_layer=calib_layer,
                                                                     label_names=None,
                                                                     num_calib_examples=1)
    qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
    check_qsym_forward(qsym, qarg_params, qaux_params, data_shape)

  for qdtype in ['uint8', 'int8', 'auto']:
    check_quantize_whole_model(qdtype)

@with_seed()
def check_fusion(sym, data_shape, attrs_dict, check_fp32_fusion=True, check_quantization=True, out_types=['uint8', 'int8', 'auto']):
  if check_fp32_fusion:
    sym_sg = sym.get_backend_symbol(SG_PASS_NAME)
    for name, attrs in attrs_dict.items():
      if name in config:
        op_name = config[name][OP_NAME]
      else:
        op_name = name
      assert ''.join(sym_sg.get_internals().list_outputs()).find(op_name) != -1
      if len(attrs):
          found = False
          for k, v in sym_sg.attr_dict().items():
            if k.find(op_name) != -1:
              found = True
              for attr_name, attr_value in attrs.items():
                assert v[attr_name].lower() == attr_value.lower()
          assert found

    arg_shapes, _, aux_shapes = sym.infer_shape()
    arg_array = [mx.nd.random.uniform(-1.0, 1.0, shape=shape) for shape in arg_shapes]
    aux_array = [mx.nd.random.uniform(shape=shape) for shape in aux_shapes]
    exe = sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
    exe.forward()
    os.environ['MXNET_SUBGRAPH_BACKEND'] = SG_PASS_NAME
    exe_sg = sym.bind(ctx=mx.current_context(), args=arg_array, aux_states=aux_array, grad_req='null')
    exe_sg.forward()
    del os.environ['MXNET_SUBGRAPH_BACKEND']
    for i in range(len(exe.outputs)):
      assert_almost_equal(exe.outputs[i].asnumpy(), exe_sg.outputs[i].asnumpy(), rtol=1e-3, atol=1e-1)

  if check_quantization:
    # fp32 to int8
    for out_type in out_types:
      check_quantize(sym, data_shape, out_type, name=op_name)
      # TODO(ciyong), since quantized fc save its params in int8, while gluon treat the default
      # variable from symbol file as fp32 which results in mismatch dtype of params.
      # Skip quantized fc in gluon pass.
      if name != 'fc':
        check_quantize(sym, data_shape, out_type, name=op_name, gluon_forward=True)

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

# single conv fuision case
def single_conv(no_bias, data_shape):
  attr = {'conv': []}
  data, weight = head_symbol(data_shape)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  return conv, attr

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
  else:
    relu = mx.symbol.Activation(data=conv, name=alg, act_type=alg)
  conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  sum = relu + conv1
  return sum, attr

# conv + add fusion case
def conv_add(no_bias, data_shape):
  attr = {'conv': {'with_sum': 'true'}}
  data, weight = head_symbol(data_shape)
  conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  conv2 = mx.symbol.Convolution(data=data, name='conv2', num_filter=64,
                               kernel=(3, 3), stride=(1, 1))
  pool = mx.sym.Pooling(data=conv2, kernel=(1, 1), pool_type='avg', name='pool')
  sum = conv1 + pool
  return sum, attr

# conv + add fusion case 2
def conv_add2(no_bias, data_shape):
  attr = {'conv': {'with_sum': 'true'}}
  data, weight = head_symbol(data_shape)
  conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  conv2 = mx.symbol.Convolution(data=data, name='conv2', num_filter=64,
                               kernel=(3, 3), stride=(1, 1))
  pool = mx.sym.Pooling(data=conv2, kernel=(1, 1), pool_type='avg', name='pool')
  sum = pool + conv1
  return sum, attr

# conv + bn + act fusion case
def conv_bn_act(no_bias, data_shape, alg):
  attr = {'conv': {'with_bn': 'true', 'with_act': 'true'}}
  data, weight = head_symbol(data_shape)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  bn1 = mx.symbol.BatchNorm(data=conv, name="bn1")
  if alg == "relu6":
    relu = mx.symbol.clip(data=bn1, name='relu6', a_min=0, a_max=6)
  else:
    relu = mx.symbol.Activation(data=bn1, name=alg, act_type=alg)
  return relu, attr

# conv + bn + add + act fusion case
def conv_bn_sum_act(no_bias, data_shape, alg):
  attr = {'conv': {'with_sum': 'true', 'with_postsum_act': 'true', 'with_bn': 'true'}}
  data, weight = head_symbol(data_shape)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  bn1 = mx.symbol.BatchNorm(data=conv, name="bn1")
  conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                                kernel=(3, 3), stride=(1, 1))
  sum1 = bn1 + conv1
  if alg == "relu6":
    relu = mx.symbol.clip(data=sum1, name='relu6', a_min=0, a_max=6)
  else:
    relu = mx.symbol.Activation(data=sum1, name=alg, act_type=alg)
  return relu, attr

# single concat case
def single_concat(data_shape, input_num, dim):
  data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
  inputs = []
  for i in range(input_num):
    inputs.append(data)
  concat = mx.symbol.Concat(*inputs, name="concat", dim=dim)
  return concat

# concat scale alignment case
def concat_scale_align(data_shape):
  data, weight = head_symbol(data_shape)
  conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=True)
  conv2 = mx.symbol.Convolution(data=data, weight=weight * 2, name='conv2', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=True)
  conv3 = mx.symbol.Convolution(data=data, weight=weight * 3, name='conv3', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=True)
  conv4 = mx.symbol.Convolution(data=data, weight=weight * 4, name='conv4', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=True)
  concat = mx.symbol.Concat(*[conv1, conv2, conv3, conv4], name="concat", dim=1)
  return concat


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

def fc_relu(no_bias, data_shape, flatten=True):
  attr = {'fc': {'with_relu': 'true'}}
  data, weight = head_symbol(data_shape)
  fc = mx.symbol.FullyConnected(name='fc', data=data, weight=weight, num_hidden=64,
                                no_bias=no_bias, flatten=flatten)
  relu = mx.symbol.Activation(data=fc, name='relu', act_type="relu")
  return relu, attr

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
def test_pos_single_conv():
  for data_shape in DATA_SHAPE:
    net, attrs = single_conv(False, data_shape)
    check_fusion(net, data_shape, attrs)
    net, attrs = single_conv(True, data_shape)
    check_fusion(net, data_shape, attrs)

@with_seed()
def test_pos_conv_act():
  act_list = {"relu": True,
              "sigmoid": True,
              "tanh": True,
              "softrelu": True,
              "relu6": True}
  for data_shape in DATA_SHAPE:
    for (alg, quantize) in act_list.items():
      net, attrs = conv_act(False, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)
      net, attrs = conv_act(True, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)

@with_seed()
def test_pos_conv_bn():
  for data_shape in DATA_SHAPE:
    net, attrs = conv_bn(False, data_shape)
    check_fusion(net, data_shape, attrs)
    net, attrs = conv_bn(True, data_shape)
    check_fusion(net, data_shape, attrs)

@with_seed()
def test_pos_conv_add():
  for data_shape in DATA_SHAPE:
    net, attrs = conv_add(False, data_shape)
    check_fusion(net, data_shape, attrs)
    net, attrs = conv_add(True, data_shape)
    check_fusion(net, data_shape, attrs)

@with_seed()
def test_pos_conv_add2():
  for data_shape in DATA_SHAPE:
    net, attrs = conv_add2(False, data_shape)
    check_fusion(net, data_shape, attrs)
    net, attrs = conv_add2(True, data_shape)
    check_fusion(net, data_shape, attrs)

@with_seed()
def test_pos_conv_bn_act():
  act_list = {"relu": True,
              "sigmoid": True,
              "tanh": True,
              "softrelu": True,
              "relu6": True}
  for data_shape in DATA_SHAPE:
    for (alg, quantize) in act_list.items():
      net, attrs = conv_bn_act(False, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)
      net, attrs = conv_bn_act(True, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)

@with_seed()
def test_pos_conv_bn_sum_act():
  act_list = {"relu": True,
              "sigmoid": True,
              "tanh": True,
              "softrelu": True,
              "relu6": False}
  for data_shape in DATA_SHAPE:
    for (alg, quantize) in act_list.items():
      net, attrs = conv_bn_sum_act(False, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)
      net, attrs = conv_bn_sum_act(True, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)

@with_seed()
def test_pos_single_concat():
  for data_shape in DATA_SHAPE:
    for out_type in ('int8', 'auto'):
      net = single_concat(data_shape, 2, -1)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False, gluon_forward=True)
      net = single_concat(data_shape, 2, 1)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False, gluon_forward=True)
      net = single_concat(data_shape, 4, 2)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False, gluon_forward=True)
      net = single_concat(data_shape, 4, 3)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False)
      check_quantize(net, data_shape, out_type, name='conv', check_calibration=False, gluon_forward=True)

@with_seed()
def test_pos_concat_scale_align():
  for data_shape in DATA_SHAPE:
    for out_type in ('int8', 'auto'):
      net = concat_scale_align(data_shape)
      check_quantize(net, data_shape, out_type, check_calibration=True, check_scale_align=True)
      check_quantize(net, data_shape, out_type, check_calibration=True, check_scale_align=True, gluon_forward=True)

@with_seed()
def test_mobilenetv2_struct():
  for data_shape in DATA_SHAPE:
      net, attrs = mobilenetv2_struct(data_shape)
      check_fusion(net, data_shape, attrs, out_types=['int8', 'auto'])

@with_seed()
def test_neg_conv_bn():
  for data_shape in DATA_SHAPE:
    syms, attrs, excluded_attrs = neg_conv_bn(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
def test_neg_conv_relu():
  for data_shape in DATA_SHAPE:
    syms, attrs, excluded_attrs = neg_conv_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
def test_neg_conv_add():
  for data_shape in DATA_SHAPE:
    syms, attrs, excluded_attrs = neg_conv_add(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
def test_neg_conv_bn_relu():
  for data_shape in DATA_SHAPE:
    syms, attrs, excluded_attrs = neg_conv_bn_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
def test_neg_conv_bn_add_relu():
  for data_shape in DATA_SHAPE:
    syms, attrs, excluded_attrs = neg_conv_bn_add_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)

@with_seed()
def test_single_fc():
  for dshape, no_bias, flatten in itertools.product(DATA_SHAPE, [True, False], [True, False]):
    syms, attrs = single_fc(no_bias, dshape, flatten)
    if flatten is True:
      check_fusion(syms, dshape, attrs, check_quantization=True)
    else:
      check_fusion(syms, dshape, attrs, check_quantization=False)


@with_seed()
def test_fc_relu():
  for dshape, no_bias, flatten in itertools.product(DATA_SHAPE, [True, False], [True, False]):
    syms, attrs = fc_relu(no_bias, dshape, flatten)
    if flatten is True:
      check_fusion(syms, dshape, attrs, check_quantization=True)
    else:
      check_fusion(syms, dshape, attrs, check_quantization=False)

@with_seed()
def test_neg_fc_relu():
  for dshape, no_bias, flatten in itertools.product(DATA_SHAPE, [True, False], [True, False]):
    syms, attrs, excluded_attrs = neg_fc_relu(no_bias, dshape, flatten)
    check_neg_fusion(syms, attrs, excluded_attrs, dshape, name='fc')

if __name__ == "__main__":
  import nose
  nose.runmodule()
