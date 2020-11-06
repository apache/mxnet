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
import pytest
import ctypes

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../unittest/'))
from mxnet.contrib import quant
from mxnet.test_utils import assert_almost_equal, assert_almost_equal_with_err, DummyIter

def test_float64_fallback():
    sym = mx.sym.FullyConnected(
        mx.sym.Variable('in'),
        mx.sym.Variable('w'),
        mx.sym.Variable('b'),
        num_hidden=2)

    dtype = 'float64'
    args = {'in': mx.nd.array([[2, 3, 4]], dtype=dtype),
        'w': mx.nd.array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
        'b': mx.nd.array([7, 8], dtype=dtype)}
    ex = sym._bind(mx.cpu(), args, args_grad=None, grad_req='write')
    ex.forward()
    ex.outputs[0].wait_to_read()


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


def initialize_block_params(block, initializer):
  for name, param in block.collect_params('.*gamma|.*moving_var|.*running_var').items():
      param.initialize(mx.init.Constant(1))
  for name, param in block.collect_params('.*beta|.*moving_mean|.*running_mean|.*bias').items():
      param.initialize(mx.init.Constant(0))
  for name, param in block.collect_params('.*weight').items():
      param.initialize(initializer)

def collect_block_args_aux(block, sym):
  arg_params, aux_params = dict(), dict()
  for k, v in block.collect_params().items():
    if k in sym.list_arguments():
      arg_params[k]= v._reduce()
    elif k in sym.list_auxiliary_states():
      aux_params[k]= v._reduce()
  return arg_params, aux_params
  
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


def check_qsym_dummy_forward(qsym, data, data_shape):
  inputs = mx.sym.var('data', dtype='float32')
  sym_block = mx.gluon.SymbolBlock(qsym, inputs)
  initialize_block_params(sym_block, mx.init.One())

  outputs = sym_block(data)
  for output in outputs:
    output.wait_to_read()
  return outputs

def check_qsym_gluon_forward(qsym, qarg_params, qaux_params, data):
  param_dict = {('arg:%s' % k): v.as_in_context(mx.current_context()) for k, v in qarg_params.items()}
  param_dict.update({('aux:%s' % k): v.as_in_context(mx.current_context()) for k, v in qaux_params.items()})

  # create SymbolBlock
  net = mx.gluon.SymbolBlock(qsym, mx.sym.var('data'))
  net.load_dict(param_dict, cast_dtype=True, dtype_source='saved')
  net.reset_ctx(ctx = mx.current_context())
  net.hybridize()
  outputs = net(data)
  for output in outputs:
    output.wait_to_read()
  return outputs

def check_quantize(sym, data_shape, out_type, name='conv',
                   check_calibration=True, check_scale_align=False):
  quantize_granularity_list = ['tensor-wise']
  if name == 'fc':
    quantize_granularity_list += ['channel-wise']

  if name in config:
    name = config[name][OP_NAME]
  sym_sg = sym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)

  inputs = mx.sym.var('data', dtype='float32')
  sym_block = mx.gluon.SymbolBlock(sym, inputs)
  initialize_block_params(sym_block, mx.init.Normal(0.5))

  min_value = -1 if out_type != 'uint8' else 0
  data = mx.random.uniform(min_value, 1.0, shape=data_shape, dtype='float32', ctx=mx.current_context())

  outputs = sym_block(data)
  for output in outputs:
      output.wait_to_read()
  ref_out = outputs
  arg_params, aux_params = collect_block_args_aux(sym_block, sym)

  excluded_sym_names = []
  excluded_op_names = []

  calib_data = mx.gluon.data.DataLoader(data, batch_size=1)
  for quantize_granularity in quantize_granularity_list:
    qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym_sg,
                                                                    arg_params=arg_params,
                                                                    aux_params=aux_params,
                                                                    ctx=mx.current_context(),
                                                                    excluded_sym_names=excluded_sym_names,
                                                                    excluded_op_names=excluded_op_names,
                                                                    quantized_dtype=out_type,
                                                                    calib_mode='naive',
                                                                    calib_data=calib_data,
                                                                    label_names=None,
                                                                    num_calib_batches=1,
                                                                    quantize_mode='full',
                                                                    quantize_granularity=quantize_granularity)
    qsym = qsym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
    if check_calibration:
      check_qsym_calibrated(qsym, out_type, name=name)
    if check_scale_align:
      check_qsym_scale_align(qsym)

    quantized_out = check_qsym_gluon_forward(qsym, qarg_params, qaux_params, data)
    for i in range(len(ref_out)):
      min_range = mx.nd.min(ref_out[i]).asscalar()
      max_range = mx.nd.max(ref_out[i]).asscalar()
      atol = 0.1 * max(abs(min_range), abs(max_range))
      assert_almost_equal_with_err(quantized_out[i].asnumpy(), ref_out[i].asnumpy(), rtol=0.1, atol=atol, etol=0.2)
    check_qsym_dummy_forward(qsym, data, data_shape)


@pytest.mark.parametrize('qdtype', ['uint8', 'int8', 'auto'])
def test_quantize_whole_model_with_forward(qdtype):
    batch_size = 4
    data_shape = (batch_size, 4, 10, 10)
    data = mx.sym.Variable('data')
    conv0 = mx.sym.Convolution(data, kernel=(1, 1), num_filter=16, name='conv0')
    sym = mx.sym.Convolution(conv0, kernel=(1, 1), num_filter=16, name='conv1')

    sym_block = mx.gluon.SymbolBlock(outputs=sym, inputs=data)
    initialize_block_params(sym_block, mx.init.Normal(0.5))
    
    in_data = mx.random.uniform(0.0 if qdtype=='uint8' else -1.0, 1.0, shape=data_shape)
    ref_out = sym_block(in_data)

    excluded_layers = []

    calib_data = mx.nd.random.uniform(0.0 if qdtype=='uint8' else -1.0, 1.0, shape=data_shape)
    calib_data = mx.gluon.data.DataLoader(calib_data, batch_size=batch_size)
    qsym = mx.contrib.quantization.quantize_net_v2(sym_block,
                                                   ctx=mx.current_context(),
                                                   exclude_layers=excluded_layers,
                                                   quantized_dtype=qdtype,
                                                   calib_mode='naive',
                                                   calib_data=calib_data,
                                                   num_calib_batches=1,
                                                   quantize_mode='full')

    outputs = qsym(in_data)
    for output in outputs:
        output.wait_to_read()

    for i in range(len(ref_out)):
        min_range = mx.nd.min(ref_out[i]).asscalar()
        max_range = mx.nd.max(ref_out[i]).asscalar()
        atol = 0.1 * max(abs(min_range), abs(max_range))
        assert_almost_equal_with_err(outputs[i].asnumpy(), ref_out[i].asnumpy(), rtol=0.1, atol=atol, etol=0.2)



def check_fusion(sym, data_shape, attrs_dict, check_fp32_fusion=True, check_quantization=True,
                 out_types=['uint8', 'int8', 'auto'], dedup_subgraph=True):
  if check_fp32_fusion:
    data_min = -1.0
    data_max = 1.0
    if ''.join(sym.get_internals().list_outputs()).find('sqrt') != -1:
      check_quantization = False
      data_min = 0

    sym_sg = sym.optimize_for(SG_PASS_NAME, dedup_subgraph=dedup_subgraph, skip_infer=True)
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
    aux_names = sym.list_auxiliary_states()
    arg_names = sym.list_arguments()
    arg_dict = {name : mx.nd.random.uniform(data_min, data_max, shape=shape, dtype='float32')
                  for shape, name in zip(arg_shapes, arg_names)}

    aux_dict = {name : mx.nd.random.uniform(shape=shape, dtype='float32') for shape, name in zip(aux_shapes, aux_names)}
    
    exe = sym._bind(ctx=mx.current_context(), args=arg_dict, aux_states=aux_dict, grad_req='null')
    exe.forward()

    exe_sg = sym_sg._bind(ctx=mx.current_context(), args=arg_dict, aux_states=aux_dict, grad_req='null')
    exe_sg.forward()
    for i in range(len(exe.outputs)):
      assert_almost_equal(exe.outputs[i].asnumpy(), exe_sg.outputs[i].asnumpy(), rtol=1e-3, atol=1e-1)

  if check_quantization:
    # fp32 to int8
    for out_type in out_types:
      check_quantize(sym, data_shape, out_type, name=name)

def check_neg_fusion(syms, attrs_name=None, excluded_attrs=None,
                     date_shape=(4,4,10,10), name='conv'):
  op_name = config[name][OP_NAME]

  for sym, attrs, excluded_attr in zip(syms, attrs_name, excluded_attrs):
    sym_sg = sym.optimize_for(SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
    exe_sg = sym_sg._simple_bind(mx.cpu(), data=date_shape, grad_req='null')

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


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
def test_pos_single_conv(no_bias, data_shape):
# single conv fusion case
    attr = {'conv': []}
    data, weight = head_symbol(data_shape)
    conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                                 kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
    check_fusion(conv, data_shape, attr)

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


# conv + add fusion case
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
def test_pos_conv_add(no_bias, data_shape):
    attr = {'conv': {'with_sum': 'true'}}
    data, weight = head_symbol(data_shape)
    conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
    conv2 = mx.symbol.Convolution(data=data, name='conv2', num_filter=64,
                               kernel=(3, 3), stride=(1, 1))
    pool = mx.sym.Pooling(data=conv2, kernel=(1, 1), pool_type='avg', name='pool')
    sum = conv1 + pool
    check_fusion(sum, data_shape, attr)


# conv + add fusion case 2
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
def test_pos_conv_add2(no_bias, data_shape):
    attr = {'conv': {'with_sum': 'true'}}
    data, weight = head_symbol(data_shape)
    conv1 = mx.symbol.Convolution(data=data, weight=weight, name='conv1', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
    conv2 = mx.symbol.Convolution(data=data, name='conv2', num_filter=64,
                               kernel=(3, 3), stride=(1, 1))
    pool = mx.sym.Pooling(data=conv2, kernel=(1, 1), pool_type='avg', name='pool')
    sum = pool + conv1
    check_fusion(sum, data_shape, attr)


# conv + bn + act fusion case
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", True),
    ("sigmoid", True),
    ("tanh", True),
    ("softrelu", True),
    ("relu6", True),
    ("leakyrelu", True),
    ("gelu", True)
])
@pytest.mark.parametrize('no_bias', [True, False])
def test_pos_conv_bn_act(no_bias, data_shape, alg, quantize):
  attr = {'conv': {'with_bn': 'true', 'with_act': 'true'}}
  data, weight = head_symbol(data_shape)
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=64,
                               kernel=(3, 3), stride=(1, 1), no_bias=no_bias)
  bn1 = mx.symbol.BatchNorm(data=conv, name="bn1")
  if alg == "relu6":
    relu = mx.symbol.clip(data=bn1, name='relu6', a_min=0, a_max=6)
  elif alg == "leakyrelu":
    relu = mx.symbol.LeakyReLU(data=bn1, slope=0.25, act_type='leaky')
  elif alg == "gelu":
    relu = mx.symbol.LeakyReLU(data=bn1, act_type='gelu')
  else:
    relu = mx.symbol.Activation(data=bn1, name=alg, act_type=alg)
  check_fusion(relu, data_shape, attr, check_quantization=quantize)


# conv + bn + add + act fusion case
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", True),
    ("sigmoid", True),
    ("tanh", True),
    #("softrelu", True), #TODO(bgawrych): failing fusion check - difference in random single element
    ("relu6", False),
    ("leakyrelu", True),
    ("gelu", False)
])
@pytest.mark.parametrize('no_bias', [True, False])
def test_pos_conv_bn_sum_act(no_bias, data_shape, alg, quantize):
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
  elif alg == "leakyrelu":
    relu = mx.symbol.LeakyReLU(data=sum1, slope=0.25, act_type='leaky')
  elif alg == "gelu":
    relu = mx.symbol.LeakyReLU(data=sum1, act_type='gelu')
  else:
    relu = mx.symbol.Activation(data=sum1, name=alg, act_type=alg)
  check_fusion(relu, data_shape, attr, check_quantization=quantize)

def run_sym_block(model, data_nd, dedup_subgraph):
  data = mx.symbol.Variable('data', shape=data_nd.shape, dtype='float32')
  sym = model.optimize_for(backend='MKLDNN', dedup_subgraph = dedup_subgraph, skip_infer = True)
  sym_block = mx.gluon.SymbolBlock(sym, data)
  initialize_block_params(sym_block, mx.init.One())
  return sym_block(data_nd)

def conv_bn_sum(data_shape, reverse_sum_order):
  attr = {'sg_mkldnn_conv_bn_add_0' : {'with_bn': 'true'}}
  data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
  weight = mx.symbol.Variable('conv_weight', dtype='float32')
  conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=4,
                            kernel=(1, 1), stride=(1, 1), no_bias=True)
  bn = mx.symbol.BatchNorm(data=conv, name="bn")
  sum = bn + data if reverse_sum_order else data + bn
  return sum, attr


@pytest.mark.parametrize('reverse_sum_order', [True, False])
@pytest.mark.parametrize('dedup_subgraph', [True, False])
def test_conv_bn_sum(reverse_sum_order, dedup_subgraph):
  data_shape=(64, 4, 10, 10)
  net, attrs = conv_bn_sum(data_shape=data_shape, reverse_sum_order=reverse_sum_order)
  check_fusion(net, data_shape, attrs, out_types=['int8', 'auto'], dedup_subgraph=dedup_subgraph)


@pytest.mark.parametrize('reverse_sum_order', [False, True])
@pytest.mark.parametrize('model_name', ['conv_bn_sum', 'mobilenetv2_struct'])
def test_dedup(reverse_sum_order, model_name):
  shape = (64, 4, 10, 10)
  data = mx.symbol.Variable('data', shape=shape, dtype='float32')
  data_nd = mx.random.uniform(-1, 1, shape=shape, ctx=mx.cpu())
  if (model_name == 'mobilenetv2_struct'):
    model, _ = mobilenetv2_struct(data_shape=shape, reverse_sum_order=reverse_sum_order)
  else:
    model, _ = conv_bn_sum(data_shape=shape, reverse_sum_order=reverse_sum_order)
  out = run_sym_block(model, data_nd, dedup_subgraph = False)
  out_dedup = run_sym_block(model, data_nd, dedup_subgraph = True)
  assert_almost_equal(out.asnumpy(), out_dedup.asnumpy(), rtol=1e-3, atol=1e-1)


# single concat case
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('input_num,dim', [
    (2, -1),
    (2, 1),
    (4, 2),
    (4, 3)
])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_single_concat(data_shape, input_num, dim, out_type):
    data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
    inputs = []
    for i in range(input_num):
        inputs.append(data)
        concat = mx.symbol.Concat(*inputs, name="concat", dim=dim)
    check_quantize(concat, data_shape, out_type, name='conv',
                   check_calibration=False)

@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_single_concat_pos_neg(data_shape, out_type):
    data, weight = head_symbol(data_shape)
    conv = mx.symbol.Convolution(data=data, weight=weight, name='conv', num_filter=4,
                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
    relu = mx.symbol.Activation(data=conv, name='relu', act_type='relu')
    inputs = [data, relu]
    concat = mx.symbol.Concat(*inputs, name="concat", dim=1)
    check_quantize(concat, data_shape, out_type, name='', check_calibration=False)


# concat scale alignment case
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_concat_scale_align(data_shape, out_type):
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
    check_quantize(concat, data_shape, out_type, check_calibration=True,
                   check_scale_align=True)


# mobilenetv2 case
def mobilenetv2_struct(data_shape, reverse_sum_order=False):
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
  sum = bn2 + bn1 if reverse_sum_order else bn1 + bn2
  return sum, attr

def tail_neg_symbol(sym1, sym2):
  fc1 = mx.sym.FullyConnected(data=sym1, num_hidden=10, flatten=True, name='fc1')
  fc2 = mx.sym.FullyConnected(data=sym2, num_hidden=10, flatten=True, name='fc2')
  concat = mx.sym.Concat(*[fc1, fc2], name="concat")
  sym = mx.sym.softmax(data=concat, name='softmax')
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


def test_pos_conv_act():
  act_list = {"relu": True,
              "sigmoid": True,
              "tanh": True,
              "softrelu": True,
              "relu6": True,
              "leakyrelu": True,
              "gelu": True}
  for data_shape in DATA_SHAPE:
    for (alg, quantize) in act_list.items():
      net, attrs = conv_act(False, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)
      net, attrs = conv_act(True, data_shape, alg)
      check_fusion(net, data_shape, attrs, check_quantization=quantize)


def test_pos_conv_bn():
  for data_shape in DATA_SHAPE:
    net, attrs = conv_bn(False, data_shape)
    check_fusion(net, data_shape, attrs)
    net, attrs = conv_bn(True, data_shape)
    check_fusion(net, data_shape, attrs)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('reverse_sum_order', [True, False])
@pytest.mark.parametrize('dedup_subgraph', [True, False])
def test_mobilenetv2_struct(data_shape, reverse_sum_order, dedup_subgraph):
      net, attrs = mobilenetv2_struct(data_shape, reverse_sum_order=reverse_sum_order)
      check_fusion(net, data_shape, attrs, out_types=['int8', 'auto'], dedup_subgraph=dedup_subgraph)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn(data_shape):
    syms, attrs, excluded_attrs = neg_conv_bn(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_relu(data_shape):
    syms, attrs, excluded_attrs = neg_conv_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_add(data_shape):
    syms, attrs, excluded_attrs = neg_conv_add(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_relu(data_shape):
    syms, attrs, excluded_attrs = neg_conv_bn_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_add_relu(data_shape):
    syms, attrs, excluded_attrs = neg_conv_bn_add_relu(data_shape)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_single_fc(data_shape, no_bias, flatten):
    syms, attrs = single_fc(no_bias, data_shape, flatten)
    check_fusion(syms, data_shape, attrs, check_quantization=flatten)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
@pytest.mark.parametrize('alg', fc_post_ops_list)
def test_fc_eltwise(data_shape, no_bias, flatten, alg):
    syms, attrs = fc_eltwise(no_bias, data_shape, flatten, alg)
    check_fusion(syms, data_shape, attrs, check_quantization=flatten)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_neg_fc_relu(data_shape, no_bias, flatten):
    syms, attrs, excluded_attrs = neg_fc_relu(no_bias, data_shape, flatten)
    check_neg_fusion(syms, attrs, excluded_attrs, data_shape, name='fc')


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
      'weight': weight_nd,
      'bias': bias_nd
  }

  ex = sym._bind(mx.cpu(), arg_params, args_grad=None)
  ex.forward(data = data_nd)
  ex.outputs[0].wait_to_read()
  sym_sg = sym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
  
  calib_data = mx.gluon.data.DataLoader(data_nd, batch_size=data_shape[0])
  qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym_sg,
                                                                   arg_params=arg_params,
                                                                   aux_params={},
                                                                   ctx=mx.cpu(),
                                                                   excluded_sym_names=None,
                                                                   excluded_op_names=None,
                                                                   quantized_dtype='int8',
                                                                   calib_mode='naive',
                                                                   calib_data=calib_data,
                                                                   label_names=None,
                                                                   num_calib_batches=1,
                                                                   quantize_mode='full')
  qsym = qsym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
  qarg_params['data'] = data_nd
  qex = qsym._bind(mx.cpu(), qarg_params, args_grad=None)
  qex.forward()
  qex.outputs[0].wait_to_read()
  assert_almost_equal_with_err(ex.outputs[0].asnumpy(), qex.outputs[0].asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)


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
      'weight': weight_nd,
      'bias': bias_nd
  }

  ex = sym._bind(mx.cpu(), arg_params, args_grad=None)
  ex.forward(data = data_nd)
  ex.outputs[0].wait_to_read()
  sym_sg = sym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
  
  calib_data = mx.gluon.data.DataLoader(data_nd, batch_size=1)
  qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym_sg,
                                                                   arg_params=arg_params,
                                                                   aux_params={},
                                                                   ctx=mx.cpu(),
                                                                   excluded_sym_names=None,
                                                                   excluded_op_names=None,
                                                                   quantized_dtype='int8',
                                                                   calib_mode='naive',
                                                                   calib_data=calib_data,
                                                                   label_names=None,
                                                                   num_calib_batches=1,
                                                                   quantize_mode='full')
  qarg_params['data'] = data_nd
  qsym = qsym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
  qex = qsym._bind(mx.cpu(), qarg_params, args_grad=None)
  qex.forward()
  qex.outputs[0].wait_to_read()
  assert_almost_equal_with_err(ex.outputs[0].asnumpy(), qex.outputs[0].asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)