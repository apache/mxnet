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

import json
import mxnet as mx
from mxnet import amp
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal
from subgraph_common import SG_PASS_NAME, QUANTIZE_SG_PASS_NAME
from test_matmul_subgraph import MultiHeadAttention

import os
import sys
from pathlib import Path
curr_path = Path(__file__).resolve().parent
sys.path.insert(0, str(curr_path.parent.parent))

from amp.common import check_amp_net_stats

AMP_SG_PASS_NAME = 'ONEDNN_AMP'
AMP_DTYPE = 'bfloat16'



def same_graph_structure(symnet_observed, symnet_expected, expected):
  nodes_obs = json.loads(symnet_observed.tojson(remove_amp_cast=False))['nodes']
  nodes_exp = json.loads(symnet_expected.tojson(remove_amp_cast=False))['nodes']
  nodes_obs = [(node['op'], node['inputs']) for node in nodes_obs]
  nodes_exp = [(node['op'], node['inputs']) for node in nodes_exp]
  assert (len(nodes_obs) == len(nodes_exp)) == expected
  for node_obs, node_exp in zip(nodes_obs, nodes_exp):
    if node_obs != node_exp:
      assert expected == False, '\n'.join([f'{n1} vs {n2}' for n1, n2 in zip(nodes_obs, nodes_exp)])
      break


def check_amp_fuse(net, data_example, expected_sym=None, quantized_nodes=[], rtol=0.05):
  net.hybridize()
  out_ref = net(*data_example)

  net.optimize_for(data_example, backend=SG_PASS_NAME)  # amp pass works only on oneDNN nodes
  lp_net = amp.convert_hybrid_block(net, data_example, target_dtype=AMP_DTYPE,
                                    excluded_sym_names=quantized_nodes, cast_params_offline=True,
                                    device=mx.current_context())
  lp_net.optimize_for(data_example, backend=AMP_SG_PASS_NAME)
  out_lp_net = lp_net(*data_example)

  # check outputs
  out_ref = [out_ref] if not isinstance(out_ref, list) else out_ref
  out_lp_net = [out_lp_net] if not isinstance(out_ref, list) else out_lp_net
  for ref_out, lp_out in zip(out_ref, out_lp_net):
    assert_almost_equal(ref_out, lp_out, rtol=rtol, atol=1.0)

  # check graph
  if expected_sym is not None:
    lp_symnet = lp_net.export(None, remove_amp_cast=False)[0]
    same_graph_structure(lp_symnet, expected_sym, True)


@mx.util.use_np
def test_amp_fc():
  class TestNet(nn.HybridBlock):
    def __init__(self):
      super(TestNet, self).__init__()
      self.fc1 = nn.Dense(16)
      self.fc2 = nn.Dense(16)

    def forward(self, x):
      x = self.fc1(x)
      x = self.fc2(x)
      return x

  net = TestNet()
  net.initialize()

  exp_data = mx.symbol.Variable('data')
  exp_weight = [mx.symbol.Variable('weight{}'.format(i)) for i in range(2)]
  exp_bias = [mx.symbol.Variable('bias{}'.format(i)) for i in range(2)]
  exp_sym = mx.symbol.amp_cast(exp_data, dtype=AMP_DTYPE)
  for weight, bias in zip(exp_weight, exp_bias):
    exp_sym = mx.symbol.FullyConnected(exp_sym, weight, bias, num_hidden=1)
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)

  data_example = mx.np.random.uniform(-1, 1, (1, 8))
  check_amp_fuse(net, [data_example], exp_sym)
  check_amp_fuse(net, [data_example], exp_sym, ['sg_onednn_fully_connected_1'])


@mx.util.use_np
def test_amp_conv():
  class TestNet(nn.HybridBlock):
    def __init__(self):
      super(TestNet, self).__init__()
      self.conv1 = nn.Conv2D(16, (3, 3))
      self.conv2 = nn.Conv2D(16, (3, 3))

    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      return x

  net = TestNet()
  net.initialize()

  data_example = mx.np.random.uniform(-1, 1, (1, 3, 8, 8))

  exp_data = mx.symbol.Variable('data')
  exp_weight = [mx.symbol.Variable('weight{}'.format(i)) for i in range(2)]
  exp_bias = [mx.symbol.Variable('bias{}'.format(i)) for i in range(2)]
  exp_sym = mx.symbol.amp_cast(exp_data, dtype=AMP_DTYPE)
  for weight, bias in zip(exp_weight, exp_bias):
    exp_sym = mx.symbol.Convolution(exp_sym, weight, bias, kernel=(3, 3), num_filter=1)
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)
  check_amp_fuse(net, [data_example], exp_sym)

  exp_sym = mx.symbol.amp_cast(exp_data, dtype=AMP_DTYPE)
  for weight, bias in zip(exp_weight, exp_bias):
    exp_sym = mx.symbol.Convolution(exp_sym, weight, bias, kernel=(3, 3), num_filter=1)
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)
  check_amp_fuse(net, [data_example], exp_sym, ['sg_onednn_conv_1'])


@mx.util.use_np
def test_amp_transformers():
  batch_size = 16
  seq_length = 32
  units = 8
  num_heads = 8

  in_data = mx.np.random.uniform(size=(batch_size, seq_length, units), dtype='float32')
  mask = mx.np.random.randint(0, 2, (batch_size, seq_length, seq_length), dtype='int32')

  net = MultiHeadAttention(units, num_heads)
  net.initialize()

  check_amp_fuse(net, [in_data, mask], None)
  check_amp_fuse(net, [in_data, mask], None, ['sg_onednn_fully_connected_0'])


@mx.util.use_np
def test_amp_concat():
  class TestNet(nn.HybridBlock):
    def __init__(self):
      super(TestNet, self).__init__()
      self.fc1 = nn.Dense(16)
      self.fc2 = nn.Dense(16)

    def forward(self, x):
      x1 = self.fc1(x)
      x2 = self.fc2(x)
      x = mx.np.concat((x1, x2), axis=1)
      return x

  net = TestNet()
  net.initialize()

  data_example = mx.np.random.uniform(-1, 1, (1, 16))

  exp_data = mx.symbol.Variable('data')
  exp_amp_data = mx.symbol.amp_cast(exp_data, dtype=AMP_DTYPE)

  exp_weight = [mx.symbol.Variable(f"fc{i}_weight") for i in range(2)]
  exp_bias = [mx.symbol.Variable(f"fc{i}_bias") for i in range(2)]
  exp_fc = [mx.symbol.FullyConnected(exp_amp_data, exp_weight[i], exp_bias[i], num_hidden=1)
            for i in range(2)]
  exp_sym = mx.symbol.Concat(*exp_fc)
  exp_sym = mx.symbol.amp_cast(exp_sym, dtype='float32')
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)
  check_amp_fuse(net, [data_example], exp_sym)

  exp_fc[0] = mx.symbol.FullyConnected(exp_amp_data, exp_weight[0], exp_bias[0], num_hidden=1)
  exp_fc[1] = mx.symbol.FullyConnected(exp_data, exp_weight[1], exp_bias[1], num_hidden=1)
  exp_sym = mx.symbol.Concat(*exp_fc)
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)
  check_amp_fuse(net, [data_example], exp_sym, ['sg_onednn_fully_connected_1'])


@mx.util.use_np
def test_amp_fuse_with_branch():
  class TestNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(TestNet, self).__init__(**kwargs)
        self.fc1 = nn.Dense(16)
        self.fc2 = nn.Dense(16)

    def forward(self, x, *args):
        out = self.fc1(x)
        out1 = self.fc2(out)
        out1 = nn.Activation('relu')(out1)
        with nn.HybridBlock.OptConstraint.disable_amp():
          out2 = mx.npx.softmax(out)
        return out1, out2

  net = TestNet()
  net.initialize()

  data_example = mx.np.ones((10,))

  #              |---> lp16_op_2
  # lp16_op_1 ---|
  #              |---> f32_amp_cast ---> f32_op
  #
  # `lp16_op_1` cannot fuse the `f32_amp_cast` node, since `lp16_op_2` already uses its lp16 output

  exp_data = mx.sym.Variable('data')
  exp_weight = [mx.symbol.Variable('weight{}'.format(i)) for i in range(2)]
  exp_bias = [mx.symbol.Variable('bias{}'.format(i)) for i in range(2)]
  amp_data = mx.sym.amp_cast(exp_data, dtype=AMP_DTYPE)
  lp16_op_1 = mx.sym.FullyConnected(amp_data, exp_weight[0], exp_bias[0], num_hidden=16)
  lp16_op_2 = mx.sym.FullyConnected(lp16_op_1, exp_weight[1], exp_bias[1], num_hidden=16)
  f32_amp_cast = mx.sym.amp_cast(lp16_op_1, dtype='float32')
  f32_op = mx.sym.softmax(f32_amp_cast)
  exp_sym = mx.sym.Group([lp16_op_2, f32_op])
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)
  check_amp_fuse(net, [data_example], exp_sym)


def test_amp_excluding_after_graph_pass():
  class TestNet(nn.HybridBlock):
    def __init__(self):
      super(TestNet, self).__init__()
      self.fc1 = nn.Dense(16)
      self.fc2 = nn.Dense(16)

    def forward(self, x):
      x = self.fc1(x)
      with nn.HybridBlock.OptConstraint.disable_amp():
        x = self.fc2(x)
      return x

  data_example = mx.np.random.uniform(-1, 1, (1, 8))
  net = TestNet()
  net.initialize()

  net_before = amp.convert_hybrid_block(net, data_example, AMP_DTYPE, cast_params_offline=True)
  check_amp_net_stats(AMP_DTYPE, net_before, data_example, lp16_tensors_num=2,
                      lp16_casts_num=1, other_casts_num=1)

  net.optimize_for(data_example, backend=SG_PASS_NAME)  # introduces new nodes
  net_after = amp.convert_hybrid_block(net, data_example, AMP_DTYPE, cast_params_offline=True)
  check_amp_net_stats(AMP_DTYPE, net_after, data_example, lp16_tensors_num=2,
                      lp16_casts_num=1, other_casts_num=1)
