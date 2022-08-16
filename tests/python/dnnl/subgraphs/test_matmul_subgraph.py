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

import copy
import mxnet as mx
import numpy as np
import pytest
from mxnet.contrib import quantization
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal, assert_almost_equal_with_err
from mxnet.util import use_np
import math


class MultiHeadAttention(nn.HybridBlock):
  def __init__(self, units, num_heads, batch_size=-1, seq_length=-1, dtype='float32', negative_case=False, no_split_case = False, **kwargs):
      super(MultiHeadAttention, self).__init__(**kwargs)
      self._units = units
      self._num_heads = num_heads
      self._fc = nn.Dense(in_units=self._units, units=3*self._units, flatten=False, dtype=dtype)
      self._scale = math.sqrt(self._units // self._num_heads)
      self.negative_case = negative_case
      self.no_split_case = no_split_case
      self.batch_size = batch_size
      self.seq_length = seq_length

  def forward(self, x, mask):
      out = self._fc(x)
      query, key, value = mx.np.split(out, 3, axis=-1)
      if self.no_split_case:
        key = mx.np.concat((key, key), axis = 1)
        value = mx.np.concat((value, value), axis = 1)
      query = mx.np.reshape(query, (-2, -2, self._num_heads, -1))
      if self.negative_case:
        query = query * 2
      key = mx.np.reshape(key, (-2, -2, self._num_heads, -1))
      value = mx.np.reshape(value, (-2, -2, self._num_heads, -1))
      scores = mx.npx.batch_dot(mx.np.swapaxes(query, 1, 2), mx.np.swapaxes(key, 1, 2),
                                transpose_b=True)
      mask = mx.np.expand_dims(mask, axis=1).astype(np.bool)
      attn_weights = mx.npx.masked_softmax(scores, mask=mask, axis=-1, temperature=self._scale)
      attn_weights = mx.npx.dropout(attn_weights, p=0.1)
      context_vec = mx.npx.batch_dot(attn_weights,
                                     mx.np.swapaxes(value, 1, 2)).transpose((0, 2, 1, 3))
      context_vec = mx.npx.reshape(context_vec, (-2, -2, -1))
      return context_vec

@use_np
@pytest.mark.parametrize('batch_size', [1, 32])
@pytest.mark.parametrize('seq_length', [124, 384])
@pytest.mark.parametrize('units', [256, 768])
@pytest.mark.parametrize('num_heads', [4, 8])
@pytest.mark.parametrize('split', [True, False])
def test_self_attention(batch_size, seq_length, units, num_heads, split):
  net = MultiHeadAttention(units, num_heads, no_split_case=not split)
  in_data = mx.np.random.uniform(size=[batch_size, seq_length, units], dtype='float32')
  if (split):
    mask = mx.np.random.uniform(low=0, high=2, size=[batch_size, seq_length, seq_length], dtype='int32')
  else:
    # key dimension will be expanded by num_heads value to simulate gpt-2 model
    # mask needs to be expanded as well
    mask = mx.np.random.uniform(low=0, high=2, size=[batch_size, seq_length, seq_length * 2], dtype='int32')

  net.initialize()
  fused_net = net
  net.hybridize()
  ref_out = net(in_data, mask)

  fused_net.optimize_for(in_data, mask, backend="ONEDNN")
  out = fused_net(in_data, mask)
  mx.nd.waitall()

  assert_almost_equal(out.asnumpy(), ref_out.asnumpy())

  calib_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(in_data, mask), batch_size=batch_size)
  qnet = mx.contrib.quant.quantize_net(net, quantized_dtype='auto',
                                            exclude_layers=None,
                                            exclude_layers_match=None,
                                            calib_data=calib_data,
                                            calib_mode='naive',
                                            num_calib_batches=batch_size,
                                            ctx=mx.cpu())

  qout = qnet(in_data, mask)
  mx.nd.waitall()

  min_range = np.min(ref_out.asnumpy())
  max_range = np.max(ref_out.asnumpy())
  atol = 0.1 * max(abs(min_range), abs(max_range))
  assert_almost_equal_with_err(qout.asnumpy(), ref_out.asnumpy(), rtol=0.1, atol=atol, etol=0.2)

@use_np
@pytest.mark.parametrize('batch_size', [1, 32])
@pytest.mark.parametrize('seq_length', [124, 384])
@pytest.mark.parametrize('units', [256, 768])
@pytest.mark.parametrize('num_heads', [4, 8])
def test_self_attention_negative(batch_size, seq_length, units, num_heads):
  net = MultiHeadAttention(units, num_heads, batch_size, seq_length, negative_case=True)
  in_data = mx.np.random.uniform(size=[batch_size, seq_length, units], dtype='float32')
  mask = mx.np.random.uniform(low=0, high=2, size=[batch_size, seq_length, seq_length], dtype='int32')

  net.initialize()
  fused_net = net
  net.hybridize()
  ref_out = net(in_data, mask)

  fused_net.optimize_for(in_data, mask, backend="ONEDNN")
  out = fused_net(in_data, mask)
  mx.nd.waitall()

  assert_almost_equal(out.asnumpy(), ref_out.asnumpy())

  calib_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(in_data, mask), batch_size=batch_size)
  qnet = mx.contrib.quant.quantize_net(net, quantized_dtype='auto',
                                            exclude_layers=None,
                                            exclude_layers_match=None,
                                            calib_data=calib_data,
                                            calib_mode='naive',
                                            num_calib_batches=batch_size,
                                            ctx=mx.cpu())

  qout = qnet(in_data, mask)
  mx.nd.waitall()
  min_range = np.min(ref_out.asnumpy())
  max_range = np.max(ref_out.asnumpy())
  atol = 0.1 * max(abs(min_range), abs(max_range))
  assert_almost_equal_with_err(qout.asnumpy(), ref_out.asnumpy(), rtol=0.1, atol=atol, etol=0.2)

@use_np
@pytest.mark.parametrize('batch_size', [1, 32])
@pytest.mark.parametrize('seq_length', [124, 384])
@pytest.mark.parametrize('units', [256, 768])
@pytest.mark.parametrize('num_heads', [4, 8])
def test_batch_dot(batch_size, seq_length, units, num_heads):
  class BatchDotBlock(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(BatchDotBlock, self).__init__(**kwargs)

    def forward(self, lhs, rhs):
      x = mx.npx.batch_dot(lhs, rhs)
      return x

  lhs_data = mx.np.random.uniform(low=-1, high=1, size=[batch_size, units, seq_length], dtype='float32')
  rhs_data = mx.np.random.uniform(low=-1, high=1, size=[batch_size, seq_length, seq_length], dtype='float32')

  net = BatchDotBlock()
  net.initialize()
  fused_net = net
  net.hybridize()
  ref_out = net(lhs_data, rhs_data)

  fused_net.optimize_for(lhs_data, rhs_data, backend="ONEDNN")
  out = fused_net(lhs_data, rhs_data)
  mx.nd.waitall()

  for i in range(len(out)):
    assert_almost_equal(out[i].asnumpy(), ref_out[i].asnumpy())

  calib_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(lhs_data, rhs_data), batch_size=1)
  qnet = mx.contrib.quant.quantize_net(net, quantized_dtype='auto',
                                            exclude_layers=None,
                                            exclude_layers_match=None,
                                            calib_data=calib_data,
                                            calib_mode='naive',
                                            num_calib_batches=1,
                                            ctx=mx.cpu())

  qout = qnet(lhs_data, rhs_data)
  mx.nd.waitall()

  min_range = np.min(ref_out.asnumpy())
  max_range = np.max(ref_out.asnumpy())
  atol = 0.1 * max(abs(min_range), abs(max_range))
  assert_almost_equal_with_err(qout.asnumpy(), ref_out.asnumpy(), rtol=0.1, atol=atol, etol=0.1)
