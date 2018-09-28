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
from mxnet.io import NDArrayIter
from mxnet.module import Module
from mxnet.symbol import Symbol
from importlib import import_module
from numpy.testing import assert_allclose
from mxnet.base import SymbolHandle, check_call, _LIB, mx_uint, c_str
from mxnet.test_utils import DummyIter
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../unittest/'))
from common import with_seed

def check_qsym_calibrated(qsym):
  attrs = qsym.attr_dict()
  if ''.join(qsym.attr_dict().keys()).find('quantized_pool') != -1:
    return 0, 0
  assert ''.join(qsym.attr_dict().keys()).find('quantized_') != -1
  for k, v in attrs.items():
    if k.find('_sg_mkldnn_conv') != -1:
      assert 'min_calib_range' in v
      assert 'max_calib_range' in v
      min_value = v['min_calib_range']
      max_value = v['max_calib_range']
    if k.find('_quantize') != -1:
      assert v['out_type'] == 'uint8'
  return float(min_value), float(max_value)

def check_qsym_forward(qsym, qarg_params, qaux_params, data_val, data_shape, label_shape):
  mod = mx.mod.Module(symbol=qsym, context=mx.current_context())
  mod.bind(for_training=False,
           data_shapes=[('data', data_shape)],
           label_shapes=[('softmax_label', label_shape)])
  mod.set_params(qarg_params, qaux_params)
  batch = mx.io.DataBatch(data_val, [])
  mod.forward(batch, is_train=False)
  for output in mod.get_outputs():
    output.wait_to_read()
  return output

def check_quantize(sym, data_shape, label_shape, data_val, sym_output):
    mod = Module(symbol=sym)
    mod.bind(data_shapes=[('data', data_shape)], label_shapes=[('softmax_label', label_shape)], for_training=False)
    mod.init_params()
    arg_params, aux_params = mod.get_params()
    excluded_sym_names = []
    if mx.current_context() == mx.cpu():
      excluded_sym_names += ['fc']
    calib_data = mx.nd.random.uniform(shape=data_shape)
    calib_data = NDArrayIter(data=calib_data)
    calib_data = DummyIter(calib_data)
    calib_layer = lambda name: name.endswith('_output')
    qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym,
                                                                     arg_params=arg_params,
                                                                     aux_params=aux_params,
                                                                     ctx=mx.current_context(),
                                                                     excluded_sym_names=excluded_sym_names,
                                                                     quantized_dtype='uint8',
                                                                     calib_mode='naive',
                                                                     calib_data=calib_data,
                                                                     calib_layer=calib_layer,
                                                                     #disable_requantize=True,
                                                                     calib_quantize_op=True,
                                                                     num_calib_examples=20)
    qsym = qsym.get_backend_symbol("MKLDNN_POST_QUANTIZE")
    minVar, maxVar = check_qsym_calibrated(qsym)
    rtol = (maxVar - minVar) / 256
    qsym_output = check_qsym_forward(qsym, qarg_params, qaux_params, data_val, data_shape, label_shape)
    assert_allclose(qsym_output[0].asnumpy(), sym_output[0].asnumpy(), rtol=rtol)

def check_fusion(sym, date_shape, label_shape, name, nofusion=False):
  exe = sym.simple_bind(mx.cpu(), data=date_shape, grad_req='null')
  sym_sg = sym.get_backend_symbol("MKLDNN")
  exe_sg = sym_sg.simple_bind(mx.cpu(), data=date_shape, grad_req='null')

  mx.random.seed(12345)
  for k, v in exe.arg_dict.items():
    v = mx.random.uniform(-1.0, 1.0, shape=v.shape)
  data_val = [exe.arg_dict['data']]

  fwd = exe.forward(is_train=False)
  fwd[0].wait_to_read()

  fwd_sg = exe_sg.forward(is_train=False)
  fwd_sg[0].wait_to_read()

  # Check the result accuracy based on fp32 fusion
  assert_allclose(fwd[0].asnumpy(), fwd_sg[0].asnumpy(), rtol=0)
  attrs=sym_sg.attr_dict()
  if not nofusion:
    assert ''.join(sym_sg.get_internals().list_outputs()).find('sg_mkldnn_conv') != -1
  for k, v in attrs.items():
    if k.find('sg_mkldnn_conv') != -1:
      for attr_op in name:
        assert v[attr_op] == 'true'

  # fp32 to uint8
  if nofusion:
    check_quantize(sym, date_shape, label_shape, data_val, fwd[0])
  else: check_quantize(sym_sg, date_shape, label_shape, data_val, fwd[0])

def single_conv():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  fc = mx.sym.FullyConnected(data=conv, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_bn():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn1')
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  fc = mx.sym.FullyConnected(data=bn, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_relu():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  relu = mx.symbol.Activation(data=conv, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_sum():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  conv1 = mx.symbol.Convolution(data=bn, weight=weight, name='conv1', num_filter=64, kernel=(3, 3), stride=(1, 1))
  sum1 = conv + conv1
  fc = mx.sym.FullyConnected(data=sum1, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_bn_relu():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn1')
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  relu = mx.symbol.Activation(data=bn, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def conv_bn_sum_relu():
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.symbol.BatchNorm(data=data, name="bn1")
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  conv1 = mx.symbol.Convolution(data=bn1, weight=weight, name='conv1', num_filter=64, kernel=(3, 3), stride=(1, 1))
  sum1 = bn + conv1
  relu = mx.symbol.Activation(data=sum1, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

def int8_pooling():
  data = mx.symbol.Variable('data')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  pool = mx.sym.Pooling(data=bn, kernel=(4, 4), pool_type='avg', name='pool')
  fc = mx.sym.FullyConnected(data=pool, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

@with_seed()
def test_sugbraph():
  def check_test_sugbraph():
    conv_attr = ['']
    conv_relu_attr = ['with_relu']
    conv_bn_attr = ['with_bn']
    conv_sum_attr = ['with_sum']
    conv_bn_relu_attr = ['with_bn', 'with_relu']
    conv_bn_sum_relu_attr = ['with_sum', 'with_postsum_relu', 'with_bn']

    shape = [(4, 4, 10, 10), (32, 3, 24, 24), (64, 8, 64, 64)]
    label = [(4, 10), (32, 10), (64, 10)]

    for date_shape, label_shape in zip(shape, label):
      net = conv_bn_sum_relu()
      check_fusion(net, date_shape, label_shape, conv_bn_sum_relu_attr)
      net = single_conv()
      check_fusion(net, date_shape, label_shape, conv_attr)
      net = conv_relu()
      check_fusion(net, date_shape, label_shape, conv_relu_attr)
      net = conv_bn()
      check_fusion(net, date_shape, label_shape, conv_bn_attr)
      net = conv_sum()
      check_fusion(net, date_shape, label_shape, conv_sum_attr)
      net = conv_bn_relu()
      check_fusion(net, date_shape, label_shape, conv_bn_relu_attr)
      net = int8_pooling()
      check_fusion(net, date_shape, label_shape, '', True)

  check_test_sugbraph()
