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
import copy


from mxnet.contrib import quantization
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal, assert_almost_equal_with_err

def test_float64_fallback():
    dtype = 'float64'
    net = nn.Dense(2, dtype=dtype,)
    in_data = mx.nd.array([[2, 3, 4]], dtype=dtype)
    net.initialize()
    out = net(in_data)
    out.wait_to_read()
    assert in_data.dtype == out.dtype

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


# Helpers
class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNetV2."""

    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")

class CustomNormalInit(mx.init.Initializer):
    """Initializes weights with random values sampled from a normal distribution
    with a custom mean and standard deviation of `sigma`.
    """
    def __init__(self, mean=0, sigma=0.01):
        super(CustomNormalInit, self).__init__(mean=mean, sigma=sigma)
        self.mean = mean
        self.sigma = sigma

    def _init_weight(self, _, arr):
        mx.random.normal(self.mean, self.sigma, arr.shape, dtype=arr.dtype, out=arr)


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


def check_quantize(net_original, data_shape, out_type, name='conv',
                   check_calibration=True, check_scale_align=False):
  quantize_granularity_list = ['tensor-wise']
  if name == 'fc':
    quantize_granularity_list += ['channel-wise']

  if name in config:
    name = config[name][OP_NAME]

  net_original.initialize(init=mx.init.Normal(0.5), force_reinit=True)
  min_value = -1 if out_type != 'uint8' else 0
  data = mx.random.uniform(min_value, 1.0, shape=data_shape, dtype='float32', ctx=mx.current_context())

  outputs = net_original(data)
  for output in outputs:
      output.wait_to_read()
  ref_out = outputs

  excluded_layers = []
  excluded_operators = []

  calib_data = mx.gluon.data.DataLoader(data, batch_size=1)
  for quantize_granularity in quantize_granularity_list:
    qnet = quantization.quantize_net(net_original,
                                     ctx=mx.current_context(),
                                     exclude_layers=excluded_layers,
                                     exclude_operators=excluded_operators,
                                     quantized_dtype=out_type,
                                     calib_mode='naive',
                                     calib_data=calib_data,
                                     num_calib_batches=1,
                                     quantize_mode='full',
                                     quantize_granularity=quantize_granularity)
    qsym, _ = qnet.export(None)
    if check_calibration:
      check_qsym_calibrated(qsym, out_type, name=name)
    if check_scale_align:
      check_qsym_scale_align(qsym)

    quantized_out = qnet(data)
    for i in range(len(ref_out)):
      min_range = mx.nd.min(ref_out[i]).asscalar()
      max_range = mx.nd.max(ref_out[i]).asscalar()
      atol = 0.1 * max(abs(min_range), abs(max_range))
      assert_almost_equal_with_err(quantized_out.asnumpy(), ref_out.asnumpy(), rtol=0.1, atol=atol, etol=0.2)


def check_fusion(net_original, data_shape, attrs_dict, check_fp32_fusion=True, check_quantization=True,
                 out_types=['uint8', 'int8', 'auto'], dedup_subgraph=True):
  net_original.initialize()
  net_original.hybridize(static_alloc=False, static_shape=False)
  data = mx.nd.random.uniform(shape=data_shape)
  net_original(data)
  net_fusion = copy.copy(net_original)
  sym, params = net_original.export(None)

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

    data = mx.nd.random.uniform(shape=data_shape, low=data_min, high=data_max)
    out_unfused = net_original(data)

    net_fusion.optimize_for(data, backend=SG_PASS_NAME)
    out_fused = net_fusion(data)

    for i in range(len(out_fused)):
      assert_almost_equal(out_unfused.asnumpy(), out_fused.asnumpy(), rtol=1e-3, atol=1e-1)

  if check_quantization:
    # fp32 to int8
    for out_type in out_types:
      check_quantize(net_original, data_shape, out_type, name=name)

def check_neg_fusion(net, attrs_name=None, excluded_attrs=None,
                     data_shapes=(4,4,10,10), name='conv'):
  net.initialize()
  net.hybridize()
  if isinstance(data_shapes, tuple):
    data_nd = [mx.nd.random.uniform(shape=data_shapes)]
  else:
    data_nd = [mx.nd.random.uniform(shape=dshape) for dshape in data_shapes]

  net(*data_nd)
  op_name = config[name][OP_NAME]
  sym, _ = net.export(None)
  sym_sg = sym.optimize_for(SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)

  attrs_dict = sym_sg.attr_dict()
  for k, v in attrs_dict.items():
    if k.find(op_name) != -1:
      for attr in attrs_name:
        assert v[attr] == 'true'
      for exc_attr in excluded_attrs:
        assert exc_attr not in v.keys()


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_single_conv(use_bias, data_shape):
  # single conv fusion case
  class Conv(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Conv, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)

    def hybrid_forward(self, F, x):
        out = self.conv0(x)
        return out

  attr = {'conv': []}
  net = Conv()
  check_fusion(net, data_shape, attr)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_add(use_bias, data_shape):
  # conv + add fusion case
  class ConvAdd(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1)
        self.pool = nn.AvgPool2D(pool_size=(1,1))

    def hybrid_forward(self, F, x):
      out = self.conv0(x) + self.pool(self.conv1(x))
      return out
    
  attr = {'conv': {'with_sum': 'true'}}
  net = ConvAdd(use_bias=use_bias)
  check_fusion(net, data_shape, attr)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
def test_pos_conv_add2(no_bias, data_shape):
  # conv + add fusion case 2
  class ConvAdd(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1)
        self.pool = nn.AvgPool2D(pool_size=(1,1))

    def hybrid_forward(self, F, x):
      out = self.pool(self.conv1(x)) + self.conv0(x)
      return out

  attr = {'conv': {'with_sum': 'true'}}
  net = ConvAdd(use_bias=True)
  check_fusion(net, data_shape, attr)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", False), #TODO(bgawrych): investigate
    ("sigmoid", True),
    ("tanh", False), #TODO(bgawrych): investigate
    #("softrelu", True), #TODO(bgawrych): bug in oneDNN with AVX
    ("relu6", False), #TODO(bgawrych): investigate
    ("leakyrelu", True),
    ("gelu", True)
])
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_act_add(data_shape, alg, quantize, use_bias):
# conv + act + add fusion case
  class ConvActAdd(nn.HybridBlock):
    def __init__(self, use_bias, alg, **kwargs):
        super(ConvActAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias,
                               weight_initializer=mx.init.Xavier(magnitude=2.24))
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        else:
          self.act = nn.Activation(activation = alg)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1.share_parameters(self.conv0.collect_params())

    def hybrid_forward(self, F, x):
        out = self.act(self.conv0(x)) + self.conv1(x)
        return out

  attrs = {'sg_mkldnn_conv_act_0': {'with_act': 'true'},
           'sg_mkldnn_conv_add_1': {'with_sum': 'true'}}

  net = ConvActAdd(use_bias, alg)
  check_fusion(net, data_shape, attrs, check_quantization=quantize)


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
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_bn_act(use_bias, data_shape, alg, quantize):
# conv + bn + act fusion case
  class ConvBNAct(nn.HybridBlock):
    def __init__(self, alg, use_bias, **kwargs):
        super(ConvBNAct, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.bn = nn.BatchNorm()
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        else:
          self.act = nn.Activation(activation = alg)

    def hybrid_forward(self, F, x):
      out = self.act(self.bn(self.conv0(x)))
      return out

  attr = {'conv': {'with_bn': 'true', 'with_act': 'true'}}
  net = ConvBNAct(alg, use_bias)
  check_fusion(net, data_shape, attr, check_quantization=quantize)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", True),
    ("sigmoid", True),
    ("tanh", True),
    #("softrelu", True), #TODO(bgawrych): failing fusion check - difference in random single element
    ("relu6", True),
    ("leakyrelu", True),
    ("gelu", False) #TODO: for True we get assert instead of not fusing pattern
])
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_bn_sum_act(use_bias, data_shape, alg, quantize):
  # conv + bn + add + act fusion case
  class ConvBNSumAct(nn.HybridBlock):
    def __init__(self, alg, use_bias, **kwargs):
        super(ConvBNSumAct, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1)
        self.conv1.share_parameters(self.conv0.collect_params())
        self.bn = nn.BatchNorm()
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        else:
          self.act = nn.Activation(activation = alg)

    def hybrid_forward(self, F, x):
        out = self.bn(self.conv0(x)) + self.conv1(x)
        out = self.act(out)
        return out

  attr = {'conv': {'with_sum': 'true', 'with_postsum_act': 'true', 'with_bn': 'true'}}
  net = ConvBNSumAct(alg, use_bias)
  check_fusion(net, data_shape, attr, check_quantization=quantize)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('input_num,dim', [
    (2, -1),
    (2, 1),
    (4, 2),
    (4, 3)
])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_single_concat(data_shape, input_num, dim, out_type):
  # single concat case
  class SingleConcat(nn.HybridBlock):
    def __init__(self, input_num, dim, **kwargs):
        super(SingleConcat, self).__init__(**kwargs)
        self.concat = nn.HybridConcatenate(axis=dim)
        for i in range(input_num):
            self.concat.add(nn.Identity())

    def hybrid_forward(self, F, x):
        out = self.concat(x)
        return out

  concat = SingleConcat(input_num, dim)
  check_quantize(concat, data_shape, out_type, name='conv',
                  check_calibration=False)

@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_single_concat_pos_neg(data_shape, out_type):
  class ConvDataConcat(nn.HybridBlock):
    def __init__(self, dim, **kwargs):
        super(ConvDataConcat, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=4, kernel_size=(1, 1), strides=1, use_bias=False)
        self.act = nn.Activation(activation = 'relu')
        self.concat_dim = dim

    def hybrid_forward(self, F, x):
        relu_out = self.act(self.conv0(x))
        out = F.concat(x, relu_out, dim=self.concat_dim)
        return out

  concat = ConvDataConcat(dim=1)
  check_quantize(concat, data_shape, out_type, name='', check_calibration=False)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_concat_scale_align(data_shape, out_type):
  # concat scale alignment case
  class ConcatScaleAlign(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(ConcatScaleAlign, self).__init__(**kwargs)
      self.shared_weight = mx.gluon.Parameter('shared_weight', init=mx.init.Xavier(magnitude=2.24),
                                              dtype='float32', allow_deferred_init=True)

    def hybrid_forward(self, F, x, shared_weight):
        conv1 = F.Convolution(x, kernel=(3,3), num_filter=64, weight=shared_weight,   no_bias=True)
        conv2 = F.Convolution(x, kernel=(3,3), num_filter=64, weight=shared_weight*2, no_bias=True)
        conv3 = F.Convolution(x, kernel=(3,3), num_filter=64, weight=shared_weight*3, no_bias=True)
        conv4 = F.Convolution(x, kernel=(3,3), num_filter=64, weight=shared_weight*4, no_bias=True)
        return F.concat(conv1, conv2, conv3, conv4, dim=1)

  concat = ConcatScaleAlign()
  check_quantize(concat, data_shape, out_type, check_calibration=True,
                  check_scale_align=True)


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
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_act(use_bias, data_shape, alg, quantize):
  # conv + act fusion case
  class ConvAct(nn.HybridBlock):
    def __init__(self, use_bias, alg, **kwargs):
        super(ConvAct, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        else:
          self.act = nn.Activation(activation = alg)

    def hybrid_forward(self, F, x):
        out = self.act(self.conv0(x))
        return out

  attrs = {'conv': {'with_act': 'true'}}
  net = ConvAct(False, alg)
  check_fusion(net, data_shape, attrs, check_quantization=quantize)
  net = ConvAct(True, alg)
  check_fusion(net, data_shape, attrs, check_quantization=quantize)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_bn(use_bias, data_shape):
  # conv + bn fusion case
  class ConvBN(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvBN, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.bn = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        out = self.bn(self.conv0(x))
        return out

  attr = {'conv': {'with_bn': 'true'}}
  net = ConvBN(use_bias)
  check_fusion(net, data_shape, attr)


# used in multiple tests
class ConvBNSum(nn.HybridBlock):
  def __init__(self, reverse_sum_order, **kwargs):
      super(ConvBNSum, self).__init__(**kwargs)
      self.conv0 = nn.Conv2D(channels=4, kernel_size=(1, 1), strides=1, use_bias=False)
      self.bn = nn.BatchNorm()
      self.reverse = reverse_sum_order

  def hybrid_forward(self, F, x):
      if self.reverse:
        return self.bn(self.conv0(x)) + x
      else:
        return x + self.bn(self.conv0(x))


@pytest.mark.parametrize('reverse_sum_order', [True, False])
@pytest.mark.parametrize('dedup_subgraph', [True, False])
def test_conv_bn_sum(reverse_sum_order, dedup_subgraph):
  data_shape=(64, 4, 10, 10)
  attr = {'sg_mkldnn_conv_bn_add_0' : {'with_bn': 'true'}}
  net = ConvBNSum(reverse_sum_order=reverse_sum_order)
  check_fusion(net, data_shape, attr, out_types=['int8', 'auto'], dedup_subgraph=dedup_subgraph)


# used in multiple tests
class MobileNetV2Struct(nn.HybridBlock):
  def __init__(self, reverse_sum_order, **kwargs):
      super(MobileNetV2Struct, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(1, 1), strides=(1,1), use_bias=False)
      self.conv2 = nn.Conv2D(channels=64, kernel_size=(1, 1), strides=(1,1), use_bias=False)
      self.bn1 = nn.BatchNorm()
      self.bn2 = nn.BatchNorm()
      self.reverse = reverse_sum_order

  def hybrid_forward(self, F, x):
      out = self.bn1(self.conv1(x))
      if self.reverse:
        return self.bn2(self.conv2(out)) + out
      else:
        return out + self.bn2(self.conv2(out))

@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('reverse_sum_order', [True, False])
@pytest.mark.parametrize('dedup_subgraph', [True, False])
def test_mobilenetv2_struct(data_shape, reverse_sum_order, dedup_subgraph):
  attr = {'sg_mkldnn_conv_bn_0' : {'with_bn': 'true'}}
  net = MobileNetV2Struct(reverse_sum_order=reverse_sum_order)
  check_fusion(net, data_shape, attr, out_types=['int8', 'auto'], dedup_subgraph=dedup_subgraph)


@pytest.mark.parametrize('reverse_sum_order', [False, True])
@pytest.mark.parametrize('model_name', ['conv_bn_sum', 'mobilenetv2_struct'])
def test_deduplication(reverse_sum_order, model_name):
  shape = (64, 4, 10, 10)
  data_nd = mx.random.uniform(-1, 1, shape=shape, ctx=mx.cpu())
  if (model_name == 'mobilenetv2_struct'):
    model_dedup = MobileNetV2Struct(reverse_sum_order=reverse_sum_order)
  else:
    model_dedup = ConvBNSum(reverse_sum_order=reverse_sum_order)

  model_dedup.initialize()
  model_no_dedup = copy.copy(model_dedup)

  model_dedup.optimize_for(data_nd, backend='MKLDNN', dedup_subgraph = True, skip_infer = True)
  out = model_dedup(data_nd)

  model_dedup.optimize_for(data_nd, backend='MKLDNN', dedup_subgraph = False, skip_infer = True)
  out_dedup = model_no_dedup(data_nd)

  assert_almost_equal(out.asnumpy(), out_dedup.asnumpy(), rtol=1e-3, atol=1e-1)


class TailNegBlock(nn.HybridBlock):
  def __init__(self, **kwargs):
    super(TailNegBlock, self).__init__(**kwargs)
    self.fc1 = nn.Dense(10, flatten=True)
    self.fc2 = nn.Dense(10, flatten=True)

  def hybrid_forward(self, F, x1, x2):
    out_fc1 = self.fc1(x1)
    out_fc2 = self.fc2(x2)
    out = F.concat(out_fc1, out_fc2)
    out = F.softmax(out)
    return out

@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn(data_shape):
  # conv + bn can't be fusion case
  # eg.1
  # conv --------- > bn
  #  |
  #  |
  #  -------------> [custom op]
  class NegConvBN(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(NegConvBN, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.bn1 = nn.BatchNorm()
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()

    def hybrid_forward(self, F, x):
      conv = self.conv1(x)
      bn = self.bn1(conv)
      pool = self.pool(conv)

      return self.tailneg(bn, pool)

  attrs = []
  excluded_attrs = []
  net = NegConvBN()
  check_neg_fusion(net, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_relu(data_shape):
  # conv + relu can't be fusion case
  # eg.1
  # conv -----------> relu
  #  |
  #  |
  #  ---------------> [custom op]
  class NegConvReLU(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(NegConvReLU, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()

    def hybrid_forward(self, F, x):
      conv = self.conv1(x)
      bn = self.act(conv)
      pool = self.pool(conv)
      return self.tailneg(bn, pool)

  attrs = []
  excluded_attrs = []
  net = NegConvReLU()
  check_neg_fusion(net, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_add(data_shape):
  # conv + add can't be fusion case
  # eg.1
  #  ---------------> [custom op]
  #  |
  #  |
  # conv -----------> add
  #                   |
  #                   |
  # added ------------>
  class NegConvAdd(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(NegConvAdd, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()
      self.add_value = mx.gluon.Parameter('add_value', init=mx.init.Xavier(magnitude=2.24),
                                          dtype='float32', allow_deferred_init=True)

    def hybrid_forward(self, F, x, add_value):
      conv = self.conv1(x)
      sum1 = conv + add_value
      pool = self.pool(conv)
      return self.tailneg(sum1, pool)

  attrs = []
  excluded_attrs = ['with_sum']
  net = NegConvAdd()
  check_neg_fusion(net, attrs, excluded_attrs, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_relu(data_shape):
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
  class NegConvBNRelu(nn.HybridBlock):
    def __init__(self, batchnorm_pool = False, **kwargs):
      super(NegConvBNRelu, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.bn = nn.BatchNorm()
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()
      self.batchnorm_pool = batchnorm_pool

    def hybrid_forward(self, F, x):
      conv = self.conv1(x)
      bn = self.bn(conv)
      relu = self.act(bn)
      pool = self.pool(bn) if self.batchnorm_pool else self.pool(conv)
      return self.tailneg(relu, pool)

  # eg.1 ([custom op] = pool11)
  net1 = NegConvBNRelu()
  attrs1 = []
  excluded_attrs1 = []
  check_neg_fusion(net1, attrs1, excluded_attrs1, data_shape)

  # eg.2 ([custom op] = pool)
  net2 = NegConvBNRelu(batchnorm_pool=True)
  attrs2 = ['with_bn']
  excluded_attrs2 = ['with_act']
  check_neg_fusion(net2, attrs2, excluded_attrs2, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_add_relu(data_shape):
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

  class NegConvBNAddRelu(nn.HybridBlock):
    def __init__(self, connect_mode = "conv_customop", **kwargs):
      super(NegConvBNAddRelu, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.bn = nn.BatchNorm()
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()
      self.connect_mode = connect_mode
      self.add_value = mx.gluon.Parameter('add_value', init=mx.init.Xavier(magnitude=2.24),
                                          dtype='float32', allow_deferred_init=True)

    def hybrid_forward(self, F, x, add_value):
      conv = self.conv1(x)
      bn = self.bn(conv)
      sum1 = bn + add_value
      relu = self.act(sum1)
      if self.connect_mode == "conv_customop":
        pool = self.pool(conv)
      elif self.connect_mode == "bn_customop":
        pool = self.pool(bn)
      else:
        pool = self.pool(sum1)
      return self.tailneg(relu, pool)

  # eg.1
  net1 = NegConvBNAddRelu(connect_mode = "conv_customop")
  attrs1 = []
  excluded_attrs1 = ['with_sum', 'with_postsum_act', 'with_bn']
  check_neg_fusion(net1, attrs1, excluded_attrs1, data_shape)

  # eg.2
  net2 = NegConvBNAddRelu(connect_mode = "bn_customop")
  attrs2 = ['with_bn']
  excluded_attrs2 = ['with_sum', 'with_postsum_act']
  check_neg_fusion(net2, attrs2, excluded_attrs2, data_shape)

  # eg.3
  net3 = NegConvBNAddRelu(connect_mode = "add_customop")
  attrs3 = ['with_bn', 'with_sum']
  excluded_attrs3 = ['with_postsum_act']
  check_neg_fusion(net3, attrs3, excluded_attrs3, data_shape)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_single_fc(data_shape, use_bias, flatten):

  class SingleFC(nn.HybridBlock):
    def __init__(self, use_bias, flatten, **kwargs):
      super(SingleFC, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten)

    def hybrid_forward(self, F, x):
      return self.fc(x)

  attrs = {'fc': {}}
  net = SingleFC(use_bias, flatten)
  check_fusion(net, data_shape, attrs, check_quantization=flatten)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
@pytest.mark.parametrize('alg', fc_post_ops_list)
def test_fc_eltwise(data_shape, use_bias, flatten, alg):
  # fc + eltwise fusion case
  class FCEltwise(nn.HybridBlock):
    def __init__(self, use_bias, flatten, alg, **kwargs):
      super(FCEltwise, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten,
                          weight_initializer=CustomNormalInit(mean=0.5, sigma=0.1) if alg == 'square_root' else None)
                                            #avoid calculating square root of negative values
      self.alg = alg

    def hybrid_forward(self, F, x):
      fc_out = self.fc(x)
      if self.alg in ['relu', 'sigmoid', 'tanh', 'softrelu']:
        out = F.Activation(fc_out, act_type=self.alg)
      elif self.alg == 'square':
        out = F.square(fc_out)
      elif self.alg == 'square_root':
        out = F.sqrt(fc_out)
      elif self.alg == 'abs':
        out = F.abs(fc_out)
      elif self.alg == 'exp':
        out = F.exp(fc_out)
      else:
        out = F.clip(fc_out, 0, 1.0)
      return out

  attrs = {'fc': {'with_eltwise': 'true'}}
  net = FCEltwise(use_bias, flatten, alg)
  check_fusion(net, data_shape, attrs, check_quantization=flatten)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_neg_fc_relu(data_shape, use_bias, flatten):
  # fc + relu can't be fusion case
  # eg.1
  # fc -----------> relu
  #  |
  #  |
  #  ---------------> [custom op]
  class NegFCReLU(nn.HybridBlock):
    def __init__(self, use_bias, flatten, **kwargs):
      super(NegFCReLU, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten)
      self.act1 = nn.Activation('relu')
      self.act2 = nn.Activation('sigmoid')
      self.tail_neg = TailNegBlock()

    def hybrid_forward(self, F, x):
      fc_out = self.fc(x)
      return self.tail_neg(self.act1(fc_out), self.act2(fc_out))


  attrs, excluded_attrs = [], []
  net = NegFCReLU(use_bias, flatten)
  check_neg_fusion(net, attrs, excluded_attrs, [data_shape], name='fc')


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
  qsym, qarg_params, qaux_params = quantization.quantize_model(sym=sym_sg,
                                                               arg_params=arg_params,
                                                               aux_params={},
                                                               ctx=mx.cpu(),
                                                               excluded_sym_names=None,
                                                               excluded_op_names=None,
                                                               quantized_dtype='int8',
                                                               calib_mode='naive',
                                                               calib_data=calib_data,
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
  qsym, qarg_params, qaux_params = quantization.quantize_model(sym=sym_sg,
                                                               arg_params=arg_params,
                                                               aux_params={},
                                                               ctx=mx.cpu(),
                                                               excluded_sym_names=None,
                                                               excluded_op_names=None,
                                                               quantized_dtype='int8',
                                                               calib_mode='naive',
                                                               calib_data=calib_data,
                                                               num_calib_batches=1,
                                                               quantize_mode='full')
  qarg_params['data'] = data_nd
  qsym = qsym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
  qex = qsym._bind(mx.cpu(), qarg_params, args_grad=None)
  qex.forward()
  qex.outputs[0].wait_to_read()
  assert_almost_equal_with_err(ex.outputs[0].asnumpy(), qex.outputs[0].asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)

@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_bn_relu_fusion(axis):
    dummy_data = mx.nd.uniform(-1.0, 1.0, shape=(32, 3, 224, 224))

    net = mx.gluon.nn.HybridSequential()
    net.add(mx.gluon.nn.BatchNorm(axis=axis))
    net.add(mx.gluon.nn.Activation('relu'))
    net.initialize()

    out1 = net(dummy_data)
    out1.wait_to_read()
    net.optimize_for(dummy_data, backend='MKLDNN')
    out2 = net(dummy_data)

    assert_almost_equal(out1, out2)
