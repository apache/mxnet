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
    if k.find('requantize_sg_mkldnn_conv') != -1:
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
    minVar, maxVar = check_qsym_calibrated(qsym)
    rtol = (maxVar - minVar) / 256
    qsym_output = check_qsym_forward(qsym, qarg_params, qaux_params, data_val, data_shape, label_shape)
    assert_allclose(qsym_output[0].asnumpy(), sym_output[0].asnumpy(), rtol=rtol)

def check_fusion(sym, date_shape, label_shape, attrs_op, nofusion=False):
  exe = sym.simple_bind(mx.cpu(), data=date_shape, grad_req='null')
  out = SymbolHandle()
  backend = "MKLDNN"
  check_call(_LIB.MXGenBackendSubgraph(sym.handle, c_str(backend), ctypes.byref(out)))
  sym_sg = Symbol(out)
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
      for attr_op in attrs_op:
        assert v[attr_op] == 'true'

  # fp32 to uint8
  if nofusion:
    check_quantize(sym, date_shape, label_shape, data_val, fwd[0])
  else: check_quantize(sym_sg, date_shape, label_shape, data_val, fwd[0])


def check_neg_fusion(syms, attrs_name=None, excluded_attrs=None, date_shape=(4,4,10,10)):
  for sym, attrs, excluded_attr in zip(syms, attrs_name, excluded_attrs):
    out = SymbolHandle()
    backend = "MKLDNN"
    check_call(_LIB.MXGenBackendSubgraph(sym.handle, c_str(backend), ctypes.byref(out)))
    sym_sg = Symbol(out)
    exe_sg = sym_sg.simple_bind(mx.cpu(), data=date_shape, grad_req='null')

    attrs_dict = sym_sg.attr_dict()
    for k, v in attrs_dict.items():
      if k.find('sg_mkldnn_conv') != -1:
        for attr in attrs:
          assert v[attr] == 'true'
        for exc_attr in excluded_attr:
          assert exc_attr not in v.keys()

# single conv fuision case
def single_conv():
  conv_attr = ['']
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  fc = mx.sym.FullyConnected(data=conv, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym, conv_attr

# conv + bn fusion case
def conv_bn():
  conv_bn_attr = ['with_bn']
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn1')
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  fc = mx.sym.FullyConnected(data=bn, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym, conv_bn_attr

# conv + relu fusion case
def conv_relu():
  conv_relu_attr = ['with_relu']
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  relu = mx.symbol.Activation(data=conv, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym, conv_relu_attr

# conv + add fusion case
def conv_add():
  conv_add_attr = ['with_sum']
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  conv1 = mx.symbol.Convolution(data=bn, weight=weight, name='conv1', num_filter=64, kernel=(3, 3), stride=(1, 1))
  sum1 = conv + conv1
  fc = mx.sym.FullyConnected(data=sum1, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym, conv_add_attr

# conv + bn + relu fusion case
def conv_bn_relu():
  conv_bn_relu_attr = ['with_bn', 'with_relu']
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn1 = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn1')
  conv = mx.symbol.Convolution(data=bn1, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  relu = mx.symbol.Activation(data=bn, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym, conv_bn_relu_attr

# conv + bn + add + relu fusion case
def conv_bn_sum_relu():
  conv_bn_add_relu_attr = ['with_sum', 'with_postsum_relu', 'with_bn']
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
  return sym, conv_bn_add_relu_attr

# pooling op quantizion case
def int8_pooling():
  data = mx.symbol.Variable('data')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')
  pool = mx.sym.Pooling(data=bn, kernel=(4, 4), pool_type='avg', name='pool')
  fc = mx.sym.FullyConnected(data=pool, num_hidden=10, flatten=True, name='fc')
  sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  return sym

# conv + bn can't be fusion case
# eg.1
# conv --------- > bn
#  |
#  |
#  -------------> [custom op]
def neg_conv_bn():
  syms = []
  attrs = []
  excluded_attrs = []
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')

  # eg.1 ([custom op] = relu1)
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn1 = mx.symbol.BatchNorm(data=conv, name="bn1")
  fc1 = mx.sym.FullyConnected(data=bn1, num_hidden=10, flatten=True, name='fc')

  pool = mx.sym.Pooling(data=conv, kernel=(4, 4), pool_type='avg', name='pool')
  fc2 = mx.sym.FullyConnected(data=pool, num_hidden=10, flatten=True, name='fc2')

  concat = mx.sym.Concat(*[fc1, fc2], name="concat")
  sym = mx.sym.SoftmaxOutput(data=concat, name='softmax')

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
def neg_conv_relu():
  syms = []
  attrs = []
  excluded_attrs = []
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')

  # eg.1 ([custom op] = pool)
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  relu = mx.symbol.Activation(data=conv, name='relu', act_type="relu")
  fc = mx.sym.FullyConnected(data=relu, num_hidden=10, flatten=True, name='fc')

  pool = mx.sym.Pooling(data=conv, kernel=(4, 4), pool_type='avg', name='pool')
  fc2 = mx.sym.FullyConnected(data=pool, num_hidden=10, flatten=True, name='fc2')

  concat = mx.sym.Concat(*[fc, fc2], name="concat")
  sym = mx.sym.SoftmaxOutput(data=concat, name='softmax')

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
def neg_conv_add():
  syms = []
  attrs = []
  excluded_attrs = []
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  val = mx.symbol.Variable('addval')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')

  # eg.1 ([custom op] = pool, [added op] = val)
  conv = mx.symbol.Convolution(data=bn, weight=weight, name='conv', num_filter=64, kernel=(3, 3), stride=(1, 1))
  sum1 = conv + val
  fc = mx.sym.FullyConnected(data=sum1, num_hidden=10, flatten=True, name='fc')

  pool = mx.sym.Pooling(data=conv, kernel=(4, 4), pool_type='avg', name='pool')
  fc2 = mx.sym.FullyConnected(data=pool, num_hidden=10, flatten=True, name='fc2')
  concat = mx.sym.Concat(*[fc, fc2], name="concat")

  sym1 = mx.sym.SoftmaxOutput(data=concat, name='softmax')
  syms.append(sym1)
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
def neg_conv_bn_relu():
  syms = []
  attrs = []
  excluded_attrs = []
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')

  # eg.1 ([custom op] = pool)
  conv11 = mx.symbol.Convolution(data=bn, weight=weight, name='conv11', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn11 = mx.symbol.BatchNorm(data=conv11, name="bn11")
  relu11 = mx.symbol.Activation(data=bn11, name='relu11', act_type="relu")
  fc11 = mx.sym.FullyConnected(data=relu11, num_hidden=10, flatten=True, name='fc11')

  pool11 = mx.sym.Pooling(data=conv11, kernel=(4, 4), pool_type='avg', name='pool11')
  fc12 = mx.sym.FullyConnected(data=pool11, num_hidden=10, flatten=True, name='fc12')
  concat11 = mx.sym.Concat(*[fc11, fc12], name="concat11")
  sym1 = mx.sym.SoftmaxOutput(data=concat11, name='softmax11')
  syms.append(sym1)
  attrs.append([])
  excluded_attrs.append([])

  # eg.2 ([custom op] = pool)
  conv21 = mx.symbol.Convolution(data=bn, weight=weight, name='conv21', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn21 = mx.symbol.BatchNorm(data=conv21, name="bn21")
  relu21 = mx.symbol.Activation(data=bn21, name='relu21', act_type="relu")
  fc21 = mx.sym.FullyConnected(data=relu21, num_hidden=10, flatten=True, name='fc21')

  pool21 = mx.sym.Pooling(data=bn21, kernel=(4, 4), pool_type='avg', name='pool21')
  fc22 = mx.sym.FullyConnected(data=pool21, num_hidden=10, flatten=True, name='fc22')
  concat21 = mx.sym.Concat(*[fc21, fc22], name="concat21")
  sym2 = mx.sym.SoftmaxOutput(data=concat21, name='softmax21')
  syms.append(sym2)
  attrs.append(['with_bn'])
  excluded_attrs.append(['with_relu'])
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
#
#
# eg.4
#                 [custom op] ------->
#                                    |
# conv -----------> bn -----------> add -----------> relu
def neg_conv_bn_add_relu():
  syms = []
  attrs = []
  excluded_attrs = []
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  addVal = mx.symbol.Variable('addval')
  bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn')

  # eg.1
  conv11 = mx.symbol.Convolution(data=bn, weight=weight, name='conv11', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn11 = mx.symbol.BatchNorm(data=conv11, name="bn11")
  sum11 = bn11 + addVal
  relu11 = mx.symbol.Activation(data=sum11, name='relu11', act_type="relu")
  fc11 = mx.sym.FullyConnected(data=relu11, num_hidden=10, flatten=True, name='fc11')

  pool11 = mx.sym.Pooling(data=conv11, kernel=(4, 4), pool_type='avg', name='pool11')
  fc12 = mx.sym.FullyConnected(data=pool11, num_hidden=10, flatten=True, name='fc12')
  concat11 = mx.sym.Concat(*[fc11, fc12], name="concat11")
  sym1 = mx.sym.SoftmaxOutput(data=concat11, name='softmax11')

  syms.append(sym1)
  attrs.append([])
  excluded_attrs.append(['with_sum', 'with_postsum_relu', 'with_bn'])

  # eg.2
  conv21 = mx.symbol.Convolution(data=bn, weight=weight, name='conv21', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn21 = mx.symbol.BatchNorm(data=conv21, name="bn21")
  sum21 = bn21 + addVal
  relu21 = mx.symbol.Activation(data=sum21, name='relu21', act_type="relu")
  fc21 = mx.sym.FullyConnected(data=relu21, num_hidden=10, flatten=True, name='fc21')

  pool21 = mx.sym.Pooling(data=bn21, kernel=(4, 4), pool_type='avg', name='pool21')
  fc22 = mx.sym.FullyConnected(data=pool21, num_hidden=10, flatten=True, name='fc22')
  concat21 = mx.sym.Concat(*[fc21, fc22], name="concat21")
  sym2 = mx.sym.SoftmaxOutput(data=concat21, name='softmax21')

  syms.append(sym2)
  attrs.append(['with_bn'])
  excluded_attrs.append(['with_sum', 'with_postsum_relu'])

  # eg.3
  conv31 = mx.symbol.Convolution(data=bn, weight=weight, name='conv31', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn31 = mx.symbol.BatchNorm(data=conv31, name="bn31")
  sum31 = bn31 + addVal
  relu31 = mx.symbol.Activation(data=sum31, name='relu31', act_type="relu")
  fc31 = mx.sym.FullyConnected(data=relu31, num_hidden=10, flatten=True, name='fc31')

  pool31 = mx.sym.Pooling(data=sum31, kernel=(4, 4), pool_type='avg', name='pool31')
  fc32 = mx.sym.FullyConnected(data=pool31, num_hidden=10, flatten=True, name='fc32')
  concat31 = mx.sym.Concat(*[fc31, fc32], name="concat31")
  sym3 = mx.sym.SoftmaxOutput(data=concat31, name='softmax21')

  syms.append(sym3)
  attrs.append(['with_bn', 'with_sum'])
  excluded_attrs.append(['with_postsum_relu'])

  # eg.4
  addVal1 = mx.symbol.Variable('addval1')
  conv41 = mx.symbol.Convolution(data=bn, weight=weight, name='conv41', num_filter=64, kernel=(3, 3), stride=(1, 1))
  bn41 = mx.symbol.BatchNorm(data=conv41, name="bn41")
  sum41 = bn41 + addVal + addVal1
  relu41 = mx.symbol.Activation(data=sum41, name='relu41', act_type="relu")
  fc41 = mx.sym.FullyConnected(data=relu41, num_hidden=10, flatten=True, name='fc41')
  sym4 = mx.sym.SoftmaxOutput(data=fc41, name='softmax41')

  syms.append(sym4)
  attrs.append(['with_bn', 'with_sum'])
  excluded_attrs.append(['with_postsum_relu'])
  return syms, attrs, excluded_attrs

@with_seed()
def check_positive_fusion():
  def positive_fusion():
    shape = [(4, 4, 10, 10), (32, 3, 24, 24), (64, 8, 64, 64)]
    label = [(4, 10), (32, 10), (64, 10)]

    for date_shape, label_shape in zip(shape, label):
      # conv + bn + add + relu
      net, attrs = conv_bn_sum_relu()
      check_fusion(net, date_shape, label_shape, attrs)
      # single conv replace
      net, attrs = single_conv()
      check_fusion(net, date_shape, label_shape, attrs)
      # conv + relu
      net, attrs = conv_relu()
      check_fusion(net, date_shape, label_shape, attrs)
      # conv + bn
      net, attrs = conv_bn()
      check_fusion(net, date_shape, label_shape, attrs)
      # conv + add
      net, attrs = conv_add()
      check_fusion(net, date_shape, label_shape, attrs)
      # conv + bn + relu
      net, attrs = conv_bn_relu()
      check_fusion(net, date_shape, label_shape, attrs)
      # pool quantlizion
      net = int8_pooling()
      check_fusion(net, date_shape, label_shape, '', True)

  positive_fusion()

@with_seed()
def check_negative_fusion():
  #shape = [(4, 4, 10, 10), (32, 3, 24, 24), (64, 8, 64, 64)]
  shape = [(4,4,10,10)]
  for data_shape in shape:
    # conv + bn
    syms, attrs, excluded_attrs = neg_conv_bn()
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)
    # conv + relu
    syms, attrs, excluded_attrs = neg_conv_relu()
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)
    # conv + add
    syms, attrs, excluded_attrs = neg_conv_add()
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)
    # conv + bn + relu
    syms, attrs, excluded_attrs = neg_conv_bn_relu()
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)
    # conv + bn + add + relu
    syms, attrs, excluded_attrs = neg_conv_bn_add_relu()
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)


@with_seed()
def test_sugbraph():
  check_positive_fusion()
  check_negative_fusion()

if __name__ == "__main__":
  import nose
  nose.runmodule()
