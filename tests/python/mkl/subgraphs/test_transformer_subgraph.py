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

@use_np
@pytest.mark.parametrize('batch_size', [1, 32])
@pytest.mark.parametrize('seq_length', [124, 384])
@pytest.mark.parametrize('units', [256, 768])
@pytest.mark.parametrize('num_heads', [4, 8])
def test_self_attention(batch_size, seq_length, units, num_heads):
  class MultiHeadAttention(nn.HybridBlock):
    def __init__(self, units, num_heads, dtype='float32', **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._units = units
        self._num_heads = num_heads
        self._fc = nn.Dense(in_units=self._units, units=3*self._units, flatten=False, dtype=dtype)
        self._scale = math.sqrt(self._units // self._num_heads)

    def forward(self, x, mask):
        x = mx.np.copy(x)
        out = self._fc(x)
        query, key, value = mx.np.split(out, 3, axis=-1)
        query = mx.npx.reshape(query, (-2, -2, self._num_heads, -1))
        key = mx.npx.reshape(key, (-2, -2, self._num_heads, -1))
        value = mx.npx.reshape(value, (-2, -2, self._num_heads, -1))
        scores = mx.npx.batch_dot(mx.np.swapaxes(query, 1, 2), mx.np.swapaxes(key, 1, 2),
                               transpose_b=True)
        mask = mx.np.expand_dims(mask, axis=1).astype(np.bool)
        attn_weights = mx.npx.masked_softmax(scores, mask=mask.astype(np.bool),
                                            axis=-1, temperature=self._scale)
        attn_weights = mx.npx.dropout(attn_weights, p=0.1)
        context_vec = mx.npx.batch_dot(attn_weights,
                                     mx.np.swapaxes(value, 1, 2)).transpose((0, 2, 1, 3))
        context_vec = mx.npx.reshape(context_vec, (-2, -2, -1))

        return context_vec

  net = MultiHeadAttention(units, num_heads)
  in_data = mx.np.random.uniform(size=[batch_size, seq_length, units], dtype='float32')
  mask = mx.np.random.uniform(low=0, high=2, size=[batch_size, seq_length, seq_length], dtype='int32')

  net.initialize()
  fused_net = net
  net.hybridize()
  ref_out = net(in_data, mask)

  fused_net.optimize_for(in_data, mask, backend="MKLDNN")
  out = fused_net(in_data, mask)
  mx.nd.waitall()

  for i in range(len(out)):
    assert_almost_equal(out[i].asnumpy(), ref_out[i].asnumpy())


  calib_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(in_data, mask), batch_size=1)
  qnet = mx.contrib.quant.quantize_net(net, quantized_dtype='auto',
                                            exclude_layers=None,
                                            exclude_layers_match=None,
                                            calib_data=calib_data,
                                            calib_mode='naive',
                                            num_calib_batches=1,
                                            ctx=mx.cpu())

  qout = qnet(in_data, mask)
  mx.nd.waitall()

  for i in range(len(ref_out)):
      min_range = np.min(ref_out[i].asnumpy())
      max_range = np.max(ref_out[i].asnumpy())
      atol = 0.1 * max(abs(min_range), abs(max_range))
      assert_almost_equal_with_err(qout[i].asnumpy(), ref_out[i].asnumpy(), rtol=0.1, atol=atol, etol=0.2)
