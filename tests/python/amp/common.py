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


def test_amp_coverage(lp_dtype, lp_name):
    conditional = [item[0] for item in amp.list_conditional_fp32_ops(lp_dtype)]
    lp16_ops = amp.list_lp16_ops(lp_dtype)
    lp16_fp32_ops = amp.list_lp16_fp32_ops(lp_dtype)
    fp32_ops = amp.list_fp32_ops(lp_dtype)
    widest_ops = amp.list_widest_type_cast(lp_dtype)
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
      inputs, put it in {lp_name.upper()}_FP32_FUNCS list,
    - if your operator requires FP32 inputs or is not safe to use with lower
      precision, put it in FP32_FUNCS list,
    - if your operator supports both FP32 and lower precision, has
      multiple inputs and expects all inputs to be of the same
      type, put it in WIDEST_TYPE_CASTS list,
    - if your operator supports both FP32 and lower precision and has
      either a single input or supports inputs of different type,
      put it in {lp_name.upper()}_FP32_FUNCS list,
    - if your operator is both safe to use in lower precision and
      it is highly beneficial to use it in lower precision, then
      put it in {lp_name.upper()}_FUNCS (this is unlikely for new operators)
    - If you are not sure which list to choose, FP32_FUNCS is the
      safest option"""
    missing_ops = required_ops - covered_ops

    if len(missing_ops) > 0:
      warnings.warn(f"{len(missing_ops)} operators {sorted(missing_ops)} do not exist in AMP lists "
                    f"(in python/mxnet/amp/lists/symbol_{lp_name.lower()}.py) - please add them. \n{guidelines}")


def test_amp_basic_use(lp_dtype):
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
  net = amp.convert_hybrid_block(net, data_example, lp_dtype)

  lp16_casts = 1  # cast for network input
  lp16_casts += 2  # cast for weights and bias of `fc1`
  lp16_casts += 2  # cast for weights and bias of `fc2`

  other_casts = 1  # cast for the network output (from lp16 to f32)

  lp16_tensors = 1  # cast network input
  lp16_tensors += 3  # cast weights and bias of `fc1`, `fc1` output
  lp16_tensors += 3  # cast weights and bias of `fc2`, `fc2` output
  lp16_tensors += 1  # reshape output
  check_amp_net_stats(lp_dtype, net, data_example, lp16_tensors_num=lp16_tensors, lp16_casts_num=lp16_casts,
                      other_casts_num=other_casts)


def test_amp_offline_casting(lp_dtype):
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
      with nn.HybridBlock.OptConstraint.disable_amp():
        x = self.fp32_op(x)
      return x

  net = TestNet()
  net.initialize()
  data_example = mx.np.random.uniform(-1, 1, (4, 3, 16, 16))
  lp_net = amp.convert_hybrid_block(net, data_example, lp_dtype, cast_params_offline=True)

  check_amp_net_stats(lp_dtype, lp_net, data_example, lp16_tensors_num=4,
                      lp16_casts_num=1, other_casts_num=1)
  for name, data in lp_net.collect_params().items():
    assert mx.nd.get_dtype_name(data.dtype) == ('float32' if 'fp32_op' in name else lp_dtype)


def test_amp_offline_casting_shared_params(lp_dtype):
  COMMON_SIZE = 4

  class TestNet(nn.HybridBlock):
    def __init__(self):
      super().__init__()
      self.lp16_op1 = nn.Dense(COMMON_SIZE)
      self.lp16_op2 = nn.Dense(COMMON_SIZE)
      self.lp16_op2.share_parameters({'weight': self.lp16_op1.weight})
      self.fp32_op = nn.Dense(COMMON_SIZE)
      self.fp32_op.share_parameters({'bias': self.lp16_op2.bias})

    def forward(self, x):
      x = self.lp16_op1(x)
      x1 = self.lp16_op2(x)
      with nn.HybridBlock.OptConstraint.disable_amp():
        x2 = self.fp32_op(x)
      x = mx.np.concat((x1, x2), axis=1)
      return x

  net = TestNet()
  net.initialize()
  data_example = mx.np.random.uniform(-1, 1, (4, COMMON_SIZE))
  lp_net = amp.convert_hybrid_block(net, data_example, lp_dtype, cast_params_offline=True)

  check_amp_net_stats(lp_dtype, lp_net, data_example, lp16_tensors_num=4,
                      lp16_casts_num=2, other_casts_num=2)
  for name, data in lp_net.collect_params().items():
    assert mx.nd.get_dtype_name(data.dtype) == ('float32' if 'fp32_op' in name else lp_dtype)


def test_lp16_fp32_ops_order_independence(lp_dtype):
  class TestNet(nn.HybridBlock):
    def __init__(self, lp16_fp32_is_first):
      super().__init__()
      if lp16_fp32_is_first:
        self.first = mx.npx.batch_flatten  # lp16_fp32_op
        self.second = nn.Dense(4)
      else:
        self.first = nn.Dense(4)
        self.second = mx.npx.batch_flatten  # lp16_fp32_op

    def forward(self, x):
      x = 2**x
      x1 = self.first(x)
      x2 = self.second(x)
      return x1, x2

  data_example = mx.np.random.uniform(-1, 1, (4, 16))

  for lp16_fp32_is_second in [False, True]:
    net = TestNet(lp16_fp32_is_second)
    net.initialize()
    net = amp.convert_hybrid_block(net, data_example, lp_dtype, cast_params_offline=True)
    check_amp_net_stats(lp_dtype, net, data_example, lp16_tensors_num=3,
                        lp16_casts_num=1, other_casts_num=2)


def test_amp_node_excluding(lp_dtype):
  DISABLE_AMP_ATTR_DICT = {'__opt_constraint__': str(
      mx.gluon.HybridBlock.OptConstraint.Flag.DisableAMP.value)}

  data = mx.sym.var('data')
  wei = mx.sym.var('weights')
  bias = mx.sym.var('bias')
  # manually excluded
  fc1 = mx.sym.FullyConnected(data, wei, bias, num_hidden=4, name='fc1', attr=DISABLE_AMP_ATTR_DICT)
  # to be excluded using the conversion API
  fc2 = mx.sym.FullyConnected(data, wei, bias, num_hidden=4, name='fc2')
  symnet = mx.sym.Group([fc1, fc2])

  net = mx.gluon.SymbolBlock(symnet, [data])
  net.initialize()

  # exclude only nodes with set attribute (only 1 node - `fc1`)
  data_example = mx.np.random.uniform(-1, 1, (4, 16))
  net_1_excluded = amp.convert_hybrid_block(net, data_example, lp_dtype)

  lp16_tensors = 4  # cast `data`, weights and bias of `fc1`, `fc1` output
  lp16_casts = 3  # `data` cast, casts for weights and bias of `fc1`
  other_casts = 1  # cast for the network output (from lp16 to f32)
  check_amp_net_stats(lp_dtype, net_1_excluded, data_example, lp16_tensors_num=lp16_tensors,
                      lp16_casts_num=lp16_casts, other_casts_num=other_casts)

  # exclude using the `excluded_sym_names` argument (both nodes)
  net_2_excluded = amp.convert_hybrid_block(net, data_example, lp_dtype,
                                            excluded_sym_names=['fc1', 'fc2'])
  check_amp_net_stats(lp_dtype, net_2_excluded, data_example, lp16_tensors_num=0,
                      lp16_casts_num=0, other_casts_num=0)


def check_amp_net_stats(lp_dtype, net, data_example, lp16_tensors_num, lp16_casts_num, other_casts_num):
  lp16_tensors = set()
  lp16_casts = set()
  other_casts = set()

  def inspect_output(tensor_name, op_name, tensor):
    dtype = mx.nd.get_dtype_name(tensor.dtype)
    if op_name == 'amp_cast':
      if dtype == lp_dtype:
        lp16_casts.add(tensor_name)
      else:
        other_casts.add(tensor_name)
    if dtype == lp_dtype:
      lp16_tensors.add(tensor_name)

  net.register_op_hook(inspect_output)
  net(data_example)

  assert len(lp16_tensors) == lp16_tensors_num, f'Bad lp16 tensors! Present tensors: {sorted(lp16_tensors)}'
  assert len(lp16_casts) == lp16_casts_num, f'Bad lp16 casts! Present casts: {sorted(lp16_casts)}'
  assert len(other_casts) == other_casts_num, f'Bad casts! Present casts: {sorted(other_casts)}'
