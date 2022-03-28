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

import os
import sys
import mxnet as mx
import numpy as np
from random import randint
import warnings
import collections
import ctypes
from mxnet import amp
import pytest
from mxnet.test_utils import set_default_device, same_symbol_structure
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
from mxnet.operator import get_all_registered_operators_grouped
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import assert_raises_cudnn_not_satisfied
sys.path.insert(0, os.path.join(curr_path, '../train'))
set_default_device(mx.gpu(0))

@pytest.fixture()
def amp_tests(request):
    def teardown():
        mx.nd.waitall()

    request.addfinalizer(teardown)

def test_amp_coverage(amp_tests):
    conditional = [item[0] for item in amp.lists.symbol_fp16.CONDITIONAL_FP32_FUNCS]

    # Check for duplicates
    for a in [amp.lists.symbol_fp16.FP16_FUNCS,
          amp.lists.symbol_fp16.FP16_FP32_FUNCS,
          amp.lists.symbol_fp16.FP32_FUNCS,
          amp.lists.symbol_fp16.WIDEST_TYPE_CASTS,
          conditional]:
        ret = [item for item, count in collections.Counter(a).items() if count > 1]
        assert ret == [], "Elements " + str(ret) + " are duplicated in the AMP lists."

    t = []
    for a in [amp.lists.symbol_fp16.FP16_FUNCS,
              amp.lists.symbol_fp16.FP16_FP32_FUNCS,
              amp.lists.symbol_fp16.FP32_FUNCS,
              amp.lists.symbol_fp16.WIDEST_TYPE_CASTS,
              conditional]:
        t += a
    ret = [item for item, count in collections.Counter(t).items() if count > 1]
    assert ret == [], "Elements " + str(ret) + " exist in more than 1 AMP list."

    # Check the coverage
    covered = set(t)
    ops = get_all_registered_operators_grouped()
    required = set(k for k in ops
                   if not k.startswith(("_backward", "_contrib_backward", "_npi_backward")) and
                   not k.endswith("_backward"))

    extra = covered - required
    assert not extra, f"{len(extra)} operators are not needed in the AMP lists: {sorted(extra)}"

    guidelines = """Please follow these guidelines for choosing a proper list:
    - if your operator is not to be used in a computational graph
      (e.g. image manipulation operators, optimizers) or does not have
      inputs, put it in FP16_FP32_FUNCS list,
    - if your operator requires FP32 inputs or is not safe to use with lower
      precision, put it in FP32_FUNCS list,
    - if your operator supports both FP32 and lower precision, has
      multiple inputs and expects all inputs to be of the same
      type, put it in WIDEST_TYPE_CASTS list,
    - if your operator supports both FP32 and lower precision and has
      either a single input or supports inputs of different type,
      put it in FP16_FP32_FUNCS list,
    - if your operator is both safe to use in lower precision and
      it is highly beneficial to use it in lower precision, then
      put it in FP16_FUNCS (this is unlikely for new operators)
    - If you are not sure which list to choose, FP32_FUNCS is the
                     safest option"""
    diff = required - covered
    assert not diff, f"{len(diff)} operators {sorted(diff)} do not exist in AMP lists (in " \
        f"python/mxnet/amp/lists/symbol_fp16.py) - please add them. " \
        f"\n{guidelines}"

@pytest.mark.skip(reason='Error during waitall(). Tracked in #18099')
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_amp_conversion_rnn(amp_tests):
    with mx.Device(mx.gpu(0)):
        model = nn.HybridSequential()
        model.add(rnn.LSTM(hidden_size=10, num_layers=2, bidirectional=True))
        model.add(nn.Dense(2))
        model.initialize()
        model.hybridize()
        out = model(mx.nd.ones((2, 3, 4)))
        new_model = amp.convert_hybrid_block(model)
        out2 = new_model(mx.nd.ones((2, 3, 4)))
        mx.test_utils.assert_almost_equal(out.asnumpy(), out2.asnumpy(), atol=1e-2, rtol=1e-2)


@mx.util.use_np
def test_fp16_offline_casting():
  class TestNet(nn.HybridBlock):
    def __init__(self):
      super().__init__()
      self.lp16_op1 = nn.Conv2D(4, 3)
      self.lp16_op2 = nn.Conv2DTranspose(4, 3)
      self.fp32_op = nn.Dense(4)

    def forward(self, x):
      x = self.lp16_op1(x)
      x = self.lp16_op2(x)
      x = x.reshape(x.shape[0], -1)
      x = self.fp32_op(x)
      return x

  net = TestNet()
  net.initialize()
  data_example = mx.np.random.uniform(-1, 1, (4, 3, 16, 16))
  lp_net = amp.convert_hybrid_block(net, data_example, target_dtype='float16',
                                    target_dtype_ops=['Convolution'], fp32_ops=['FullyConnected'],
                                    cast_params_offline=True, device=mx.current_context())
  lp_net(data_example)
  for name, data in lp_net.collect_params().items():
    assert data.dtype == (np.float32 if 'fp32_op' in name else 'float16')


@mx.util.use_np
def test_fp16_offline_casting_shared_params():
  COMMON_SIZE = 4

  class TestNet(nn.HybridBlock):
    def __init__(self):
      super().__init__()
      self.lp16_op1 = nn.Dense(COMMON_SIZE)
      self.lp16_op2 = nn.Dense(COMMON_SIZE)
      self.lp16_op2.share_parameters({'weight': self.lp16_op1.weight})
      self.fp32_op = nn.Conv1D(COMMON_SIZE, 3)
      self.fp32_op.share_parameters({'bias': self.lp16_op2.bias})

    def forward(self, x):
      x = self.lp16_op1(x)
      x1 = self.lp16_op2(x)
      x2 = mx.np.expand_dims(x, 1)
      x2 = self.fp32_op(x2)
      x2 = nn.Flatten()(x2)
      x = mx.np.concat((x1, x2), axis=1)
      return x

  net = TestNet()
  net.initialize()
  data_example = mx.np.random.uniform(-1, 1, (4, COMMON_SIZE))
  lp_net = amp.convert_hybrid_block(net, data_example, target_dtype='float16',
                                    target_dtype_ops=['FullyConnected'], fp32_ops=['Convolution'],
                                    cast_params_offline=True, device=mx.current_context())
  lp_net(data_example)
  for name, data in lp_net.collect_params().items():
    assert data.dtype == (np.float32 if 'fp32_op' in name else 'float16')
