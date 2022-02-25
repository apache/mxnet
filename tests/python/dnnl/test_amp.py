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
import warnings
import collections
import ctypes
from mxnet import amp
from mxnet.amp.amp import bfloat16
from mxnet.gluon import nn
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))


def test_amp_coverage():
    conditional = [item[0] for item in amp.lists.symbol_bf16.CONDITIONAL_FP32_FUNCS]

    # Check for duplicates
    for a in [amp.lists.symbol_bf16.BF16_FUNCS,
              amp.lists.symbol_bf16.BF16_FP32_FUNCS,
              amp.lists.symbol_bf16.FP32_FUNCS,
              amp.lists.symbol_bf16.WIDEST_TYPE_CASTS,
              conditional]:
        ret = [item for item, count in collections.Counter(a).items() if count > 1]
        assert ret == [], "Elements " + str(ret) + " are duplicated in the AMP lists."

    t = []
    for a in [amp.lists.symbol_bf16.BF16_FUNCS,
              amp.lists.symbol_bf16.BF16_FP32_FUNCS,
              amp.lists.symbol_bf16.FP32_FUNCS,
              amp.lists.symbol_bf16.WIDEST_TYPE_CASTS,
              conditional]:
        t += a
    ret = [item for item, count in collections.Counter(t).items() if count > 1]
    assert ret == [], "Elements " + str(ret) + " exist in more than 1 AMP list."

    # Check the coverage
    py_str = lambda x: x.decode('utf-8')

    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    mx.base._LIB.MXListAllOpNames(ctypes.byref(size),
                                  ctypes.byref(plist))
    op_names = []
    for i in range(size.value):
        s = py_str(plist[i])
        if not s.startswith("_backward") \
           and not s.startswith("_contrib_backward_"):
            op_names.append(s)

    ret1 = set(op_names) - set(t)

    if ret1 != set():
        warnings.warn("Operators " + str(ret1) + " do not exist in AMP lists (in "
                      "python/mxnet/amp/lists/symbol_bf16.py) - please add them. "
                      """Please follow these guidelines for choosing a proper list:
                       - if your operator is not to be used in a computational graph
                         (e.g. image manipulation operators, optimizers) or does not have
                         inputs, put it in BF16_FP32_FUNCS list,
                       - if your operator requires FP32 inputs or is not safe to use with lower
                         precision, put it in FP32_FUNCS list,
                       - if your operator supports both FP32 and lower precision, has
                         multiple inputs and expects all inputs to be of the same
                         type, put it in WIDEST_TYPE_CASTS list,
                       - if your operator supports both FP32 and lower precision and has
                         either a single input or supports inputs of different type,
                         put it in BF16_FP32_FUNCS list,
                       - if your operator is both safe to use in lower precision and
                         it is highly beneficial to use it in lower precision, then
                         put it in BF16_FUNCS (this is unlikely for new operators)
                       - If you are not sure which list to choose, FP32_FUNCS is the
                         safest option""")


@mx.util.use_np
def test_bf16_offline_casting():
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
  lp_net = amp.convert_hybrid_block(net, data_example, target_dtype=bfloat16,
                                    target_dtype_ops=['Convolution'], fp32_ops=['FullyConnected'],
                                    cast_params_offline=True, device=mx.current_context())
  lp_net(data_example)
  for name, data in lp_net.collect_params().items():
    assert data.dtype == (np.float32 if 'fp32_op' in name else bfloat16)


@mx.util.use_np
def test_bf16_offline_casting_shared_params():
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
  lp_net = amp.convert_hybrid_block(net, data_example, target_dtype=bfloat16,
                                    target_dtype_ops=['FullyConnected'], fp32_ops=['Convolution'],
                                    cast_params_offline=True, device=mx.current_context())
  lp_net(data_example)
  for name, data in lp_net.collect_params().items():
    assert data.dtype == (np.float32 if 'fp32_op' in name else bfloat16)
