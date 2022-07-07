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

import mxnet as mx
import numpy as np
import unittest
import pytest
import ctypes
import copy


from mxnet.contrib import quantization
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal, assert_almost_equal_with_err

OP_NAME='op_name'
QUANTIZED_OP_NAME='quantized_op_name'
SG_PASS_NAME='ONEDNN'
QUANTIZE_SG_PASS_NAME='ONEDNN_QUANTIZE'
config =  {
  'conv': {
    OP_NAME: 'sg_onednn_conv',
    QUANTIZED_OP_NAME: 'quantized_sg_onednn_conv'
  },
  'fc': {
    OP_NAME: 'sg_onednn_fully_connected',
    QUANTIZED_OP_NAME: 'quantized_sg_onednn_fully_connected'
  }
}

DATA_SHAPE=[(64, 4, 10, 10), (4, 3, 24, 24), (1, 16, 32, 32)]

# Helpers
class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNetV2."""

    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def forward(self, x):
        return mx.np.clip(x, 0, 6)

class TailNegBlock(nn.HybridBlock):
  def __init__(self, **kwargs):
    super(TailNegBlock, self).__init__(**kwargs)
    self.fc1 = nn.Dense(10, flatten=True)
    self.fc2 = nn.Dense(10, flatten=True)

  def forward(self, x1, x2):
    out_fc1 = self.fc1(x1)
    out_fc2 = self.fc2(x2)
    out = mx.np.concatenate([out_fc1, out_fc2])
    out = mx.npx.softmax(out)
    return out

class CustomNormalInit(mx.init.Initializer):
    """Initializes weights with random values sampled from a normal distribution
    with a custom mean and standard deviation of `sigma`.
    """
    def __init__(self, mean=0, sigma=0.01, bounded=False):
        super(CustomNormalInit, self).__init__(mean=mean, sigma=sigma, bounded=bounded)
        self.mean = mean
        self.sigma = sigma
        self.bounded = bounded

    def _init_weight(self, _, arr):
        mx.np.random.normal(self.mean, self.sigma, arr.shape, dtype=arr.dtype, out=arr)
        if self.bounded:
            mx.np.abs(arr, out=arr)


def check_qsym_calibrated(qsym, out_type, name='conv'):
  quantized_op_name = 'quantized_' + name
  assert ''.join(qsym.attr_dict().keys()).find(quantized_op_name) != -1
  for k, v in qsym.attr_dict().items():
    if k.find('_quantize') != -1:
      assert v['out_type'] == out_type
    if k.find(quantized_op_name) != -1:
      if (quantized_op_name.startswith("quantized_sg_onednn_fully_connected") 
          or quantized_op_name.startswith("quantized_sg_onednn_conv")) and 'enabled_float_output' in v:
        continue
      assert 'min_calib_range' in v
      assert 'max_calib_range' in v

def check_qsym_scale_align(qsym):
  assert ''.join(qsym.attr_dict().keys()).find('quantized_sg_onednn_conv') != -1
  init = False
  for k, v in qsym.attr_dict().items():
    if k.find('quantized_sg_onednn_conv') != -1:
      assert 'min_calib_range' in v
      assert 'max_calib_range' in v
      if not init:
        min_calib_range = v['min_calib_range']
        max_calib_range = v['max_calib_range']
        init = True
      else:
        assert min_calib_range == v['min_calib_range']
        assert max_calib_range == v['max_calib_range']


def check_fusion_parameter(sym, attrs_dict):
  for name, attrs in attrs_dict.items():
    if name in config:
      op_name = config[name][OP_NAME]
    else:
      op_name = name
    assert ''.join(sym.get_internals().list_outputs()).find(op_name) != -1
    if len(attrs):
      found = False
      for k, v in sym.attr_dict().items():
        if k.find('_quantize') != -1:
          continue
        if k.find(op_name) != -1:
          found = True
          for attr_name, attr_value in attrs.items():
            assert v[attr_name].lower() == attr_value.lower()
      assert found

def check_quantize(net_original, data_shapes, out_type, name='conv',
                   check_calibration=True, check_scale_align=False, quantize_mode='full',
                   attrs_dict={}, calib_mode='naive', check_fusion=True):
  quantize_granularity_list = ['tensor-wise']
  if name == 'fc':
    quantize_granularity_list += ['channel-wise']

  if name in config:
    name = config[name][OP_NAME]

  sigma = 0.01 if hasattr(net_original, 'alg') is True and net_original.alg == 'exp' else 0.5
  if out_type == 'uint8':
    # Initialize weights and tensors only with positive values to be sure
    # that results are always positive
    init = CustomNormalInit(sigma=sigma, bounded=True)
    min_value = 0
  else:
    init = mx.init.Normal(sigma)
    min_value = -1

  net_original.initialize(init=init, force_reinit=True)

  one_shape = isinstance(data_shapes, tuple)
  if one_shape:
    # replace one shape with list of shapes with one element inside to follow later the same schema
    data_shapes=[data_shapes]
  data = []
  for shape in data_shapes:
    data.append(mx.np.random.uniform(min_value, 1.0, size=shape, dtype='float32', device=mx.cpu()))

  outputs = net_original(*data)
  for output in outputs:
      output.wait_to_read()
  ref_out = outputs
  one_output = not isinstance(ref_out, list)
  if one_output:
      # make a list to have a common path for one and multiple outputs
      ref_out = [ref_out]

  class TestDataLoader(mx.gluon.data.DataLoader):
    def __init__(self, data):
      self.data = data
      self.finish = False

    def __iter__(self):
      self.finish = False
      return self

    def __next__(self):
      if self.finish:
        raise StopIteration
      self.finish = True
      return self.data

    def __del__(self):
      pass

  calib_data = TestDataLoader(data)

  for quantize_granularity in quantize_granularity_list:
    qnet = quantization.quantize_net(net_original,
                                     device=mx.cpu(),
                                     exclude_layers=None,
                                     exclude_operators=None,
                                     quantized_dtype=out_type,
                                     calib_mode=calib_mode,
                                     calib_data=calib_data,
                                     num_calib_batches=1,
                                     quantize_mode=quantize_mode,
                                     quantize_granularity=quantize_granularity)
    qsym, _ = qnet.export(None)
    if check_fusion:
      check_fusion_parameter(qsym, attrs_dict)
    if check_calibration:
      check_qsym_calibrated(qsym, out_type, name=name)
    if check_scale_align:
      check_qsym_scale_align(qsym)

    quantized_out = qnet(*data)
    if one_output:
      quantized_out = [quantized_out]
    for i in range(len(ref_out)):
      min_range = mx.np.min(ref_out[i]).item()
      max_range = mx.np.max(ref_out[i]).item()
      atol = 0.1 * max(abs(min_range), abs(max_range))
      assert_almost_equal_with_err(quantized_out[i].asnumpy(), ref_out[i].asnumpy(),
                                   rtol=0.1, atol=atol, etol=0.2)


def check_fusion(net_original, data_shapes, attrs_dict, input_type='float32', check_fusion=True,
                 check_quantization=True, out_types=['uint8', 'int8', 'auto'], dedup_subgraph=True,
                 quantize_mode='full'):
  net_original.initialize()
  net_original.hybridize(static_alloc=False, static_shape=False)
  one_shape = isinstance(data_shapes, tuple)
  data_min = -1.0 if input_type == 'float32' else -10
  data_max = 1.0 if input_type == 'float32' else 10

  if one_shape:
    # replace one shape with list of shapes with one element to follow later the same schema
    data_shapes=[data_shapes]
  data = []
  for shape in data_shapes:
    data.append(mx.np.random.uniform(size=shape, dtype=input_type, device=mx.cpu(),
                low=data_min, high=data_max))
  net_original(*data)
  net_fusion = copy.copy(net_original)
  sym, _ = net_original.export(None)

  if check_fusion:
    if ''.join(sym.get_internals().list_outputs()).find('sqrt') != -1:
      check_quantization = False
      data_min = 0

    sym_sg = sym.optimize_for(SG_PASS_NAME, dedup_subgraph=dedup_subgraph, skip_infer=True)
    check_fusion_parameter(sym_sg, attrs_dict)
    if data_min == 0 and mx.npx.is_np_default_dtype():
      # regenerate inputs if they have different range or data type
      data = []
      for shape in data_shapes:
        data.append(mx.np.random.uniform(size=shape, device=mx.cpu(), low=data_min, high=data_max))
    out_unfused = net_original(*data)

    net_fusion.optimize_for(*data, backend=SG_PASS_NAME)
    out_fused = net_fusion(*data)

    assert_almost_equal(out_unfused.asnumpy(), out_fused.asnumpy(), rtol=1e-3, atol=1e-1)

  if check_quantization:
    # fp32 to int8
    for out_type in out_types:
      check_quantize(net_original, data_shapes, out_type, name=list(attrs_dict.keys())[0],
                     quantize_mode=quantize_mode, attrs_dict=attrs_dict)

def check_neg_fusion(net_original, attrs_name=None, excluded_attrs=None,
                     data_shapes=[(4,4,10,10)], name='conv'):
  op_name = config[name][OP_NAME]
  one_shape = isinstance(data_shapes, tuple)
  if one_shape:
    # replace one shape with list of shapes with one element to follow later the same schema
    data_shapes = [data_shapes]
  data = []
  for shape in data_shapes:
    data.append(mx.np.random.uniform(size=shape))

  net_original.initialize()
  net_original.hybridize()
  net_original(*data)

  sym, _ = net_original.export(None)
  sym_sg = sym.optimize_for(SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)

  attrs_dict = sym_sg.attr_dict()
  for k, v in attrs_dict.items():
    if k.find(op_name) != -1:
      for attr in attrs_name:
        assert v[attr] == 'true'
      for exc_attr in excluded_attrs:
        assert exc_attr not in v.keys(), exc_attr + " atribute shouldn't exist"



def check_neg_fusion_quantized(net_original, attrs_name=None, excluded_attrs=None,
                     data_shapes=[(4,4,10,10)], name='conv'):
  op_name = config[name][OP_NAME]
  net_original.initialize(init=mx.init.Normal(0.5), force_reinit=True)
  one_shape = isinstance(data_shapes, tuple)
  if one_shape:
    # replace one shape with list of shapes with one element inside to follow later the same schema
    data_shapes=[data_shapes]
  data = []
  for shape in data_shapes:
    data.append(mx.np.random.uniform(size=shape, dtype='float32', device=mx.cpu()))

  dataArray= mx.gluon.data.ArrayDataset(*data)
  calib_data = mx.gluon.data.DataLoader(dataArray, batch_size=1)

  qnet = quantization.quantize_net(net_original,
                                    device=mx.cpu(),
                                    exclude_layers=None,
                                    exclude_operators=None,
                                    quantized_dtype='int8',
                                    calib_mode='naive',
                                    calib_data=calib_data,
                                    num_calib_batches=1,
                                    quantize_mode='full',
                                    quantize_granularity='tensor-wise')
  qsym, _ = qnet.export(None)
  attrs_dict = qsym.attr_dict()
  for k, v in attrs_dict.items():
    if k.find(op_name) != -1:
      for attr in attrs_name:
        assert v[attr] == 'true'
      for exc_attr in excluded_attrs:
        assert exc_attr not in v.keys(), exc_attr + " atribute shouldn't exist"

