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

import warnings
import collections
import mxnet as mx
from mxnet import amp
from mxnet.gluon import nn
from mxnet.operator import get_all_registered_operators_grouped

if 'cpu' in mx.current_device().device_type:
  AMP_DTYPE = 'bfloat16'
  LP_NAME = 'BF16'
elif 'gpu' in mx.current_device().device_type:
  AMP_DTYPE = 'float16'
  LP_NAME = 'FP16'
else:
  raise mx.MXNetError('Unsupported device')


def test_amp_coverage():
    conditional = [item[0] for item in amp.list_conditional_fp32_ops(AMP_DTYPE)]
    lp16_ops = amp.list_lp16_ops(AMP_DTYPE)
    lp16_fp32_ops = amp.list_lp16_fp32_ops(AMP_DTYPE)
    fp32_ops = amp.list_fp32_ops(AMP_DTYPE)
    widest_ops = amp.list_widest_type_cast(AMP_DTYPE)
    all_lp_lists = [lp16_ops, lp16_fp32_ops, fp32_ops, widest_ops, conditional]

    # Check for duplicates
    for op_list in all_lp_lists:
        ret = [op for op, count in collections.Counter(op_list).items() if count > 1]
        assert ret == [], "Elements " + str(ret) + " are duplicated in the AMP lists."

    all_lp_ops = [op for op_list in all_lp_lists for op in op_list]
    ret = [op for op, count in collections.Counter(all_lp_ops).items() if count > 1]
    assert ret == [], "Elements " + str(ret) + " exist in more than 1 AMP list."

    # Check the coverage
    covered_ops = set(all_lp_ops)
    all_mxnet_ops = get_all_registered_operators_grouped()
    required_ops = {op for op in all_mxnet_ops if not "backward" in op}

    extra_ops = covered_ops - required_ops
    assert not extra_ops, f"{len(extra_ops)} operators are not needed in the AMP lists: {sorted(extra_ops)}"

    guidelines = f"""Please follow these guidelines for choosing a proper list:
    - if your operator is not to be used in a computational graph
      (e.g. image manipulation operators, optimizers) or does not have
      inputs, put it in {LP_NAME}_FP32_FUNCS list,
    - if your operator requires FP32 inputs or is not safe to use with lower
      precision, put it in FP32_FUNCS list,
    - if your operator supports both FP32 and lower precision, has
      multiple inputs and expects all inputs to be of the same
      type, put it in WIDEST_TYPE_CASTS list,
    - if your operator supports both FP32 and lower precision and has
      either a single input or supports inputs of different type,
      put it in {LP_NAME}_FP32_FUNCS list,
    - if your operator is both safe to use in lower precision and
      it is highly beneficial to use it in lower precision, then
      put it in {LP_NAME}_FUNCS (this is unlikely for new operators)
    - If you are not sure which list to choose, FP32_FUNCS is the
      safest option"""
    missing_ops = required_ops - covered_ops

    if len(missing_ops) > 0:
      warnings.warn(f"{len(missing_ops)} operators {sorted(missing_ops)} do not exist in AMP lists "
                    f"(in python/mxnet/amp/lists/symbol_{LP_NAME.lower()}.py) - please add them. \n{guidelines}")


def check_amp_net_stats(net, data_example, lp16_tensors_num, lp16_casts_num, other_casts_num):
  def inspect_output(tensor_name, op_name, tensor):
    nonlocal lp16_tensors_num, lp16_casts_num, other_casts_num

    dtype = mx.nd.get_dtype_name(tensor.dtype)
    if op_name == 'amp_cast':
      if dtype == AMP_DTYPE:
        lp16_casts_num -= 1
      else:
        other_casts_num -= 1
    if dtype == AMP_DTYPE:
      lp16_tensors_num -= 1

  net.register_op_hook(inspect_output)
  net(data_example)

  assert lp16_tensors_num == 0
  assert lp16_casts_num == 0
  assert other_casts_num == 0


def test_amp_basic_use():
  class TestNet(nn.HybridBlock):
    def __init__(self):
      super().__init__()
      self.fc1 = nn.Dense(4)
      self.fc2 = nn.Dense(4)

    def forward(self, x):
      x = self.fc1(x)
      x = self.fc2(x)
      return x.reshape((-1, 2, 2))

  data_example = mx.np.random.uniform(-1, 1, (4, 4))

  net = TestNet()
  net.initialize()
  net = amp.convert_hybrid_block(net, data_example, AMP_DTYPE)

  lp16_casts = 1  # net_data_cast
  lp16_casts += 2  # fc1_weights_cast, fc1_bias_cast
  lp16_casts += 2  # fc2_weights_cast, fc2_bias_cast

  other_casts = 1  # net_output_cast

  lp16_tensors = 1  # net_data_cast_output
  lp16_tensors += 3  # fc1_weights_cast_output, fc1_bias_cast_output, fc1_output
  lp16_tensors += 3  # fc2_weights_cast_output, fc2_bias_cast_output, fc2_output
  lp16_tensors += 1  # reshape_output
  check_amp_net_stats(net, data_example, lp16_tensors_num=lp16_tensors, lp16_casts_num=lp16_casts,
                      other_casts_num=other_casts)


@mx.util.use_np
def test_amp_offline_casting():
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
  lp_net = amp.convert_hybrid_block(net, data_example, AMP_DTYPE, target_dtype_ops=['Convolution'],
                                    fp32_ops=['FullyConnected'], cast_params_offline=True)

  check_amp_net_stats(lp_net, data_example, lp16_tensors_num=4, lp16_casts_num=1, other_casts_num=1)
  for name, data in lp_net.collect_params().items():
    assert mx.nd.get_dtype_name(data.dtype) == ('float32' if 'fp32_op' in name else AMP_DTYPE)


@mx.util.use_np
def test_amp_offline_casting_shared_params():
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
      x2 = mx.npx.batch_flatten(x2)
      x = mx.np.concat((x1, x2), axis=1)
      return x

  net = TestNet()
  net.initialize()
  data_example = mx.np.random.uniform(-1, 1, (4, COMMON_SIZE))
  lp_net = amp.convert_hybrid_block(net, data_example, AMP_DTYPE, target_dtype_ops=['FullyConnected'],
                                    fp32_ops=['Convolution'], cast_params_offline=True)

  check_amp_net_stats(lp_net, data_example, lp16_tensors_num=5, lp16_casts_num=2, other_casts_num=2)
  for name, data in lp_net.collect_params().items():
    assert mx.nd.get_dtype_name(data.dtype) == ('float32' if 'fp32_op' in name else AMP_DTYPE)


@mx.util.use_np
def test_lp16_fp32_ops_order_independence():
  class TestNet(nn.HybridBlock):
    def __init__(self, lp16_fp32_is_first):
      super().__init__()
      if lp16_fp32_is_first:
        self.first = nn.Activation('relu')  # lp16_fp32_op
        self.second = nn.Dense(4)
      else:
        self.first = nn.Dense(4)
        self.second = nn.Activation('relu')  # lp16_fp32_op

    def forward(self, x):
      x = 2**x
      x1 = self.first(x)
      x2 = self.second(x)
      return x1, x2

  data_example = mx.np.random.uniform(-1, 1, (4, 16))

  for lp16_fp32_is_second in [False, True]:
    net = TestNet(lp16_fp32_is_second)
    net.initialize()
    net = amp.convert_hybrid_block(net, data_example, AMP_DTYPE, cast_params_offline=True)
    check_amp_net_stats(net, data_example, lp16_tensors_num=3, lp16_casts_num=1, other_casts_num=2)
