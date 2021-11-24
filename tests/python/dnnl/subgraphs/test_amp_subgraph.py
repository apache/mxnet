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
import pytest
import mxnet as mx
from mxnet import amp, use_np
from mxnet.amp.amp import bfloat16
from mxnet.test_utils import assert_almost_equal
from subgraph_common import SG_PASS_NAME
from test_matmul_subgraph import MultiHeadAttention

AMP_SG_PASS_NAME = 'ONEDNN_AMP'


def check_amp_with_quantization(sym_fp32, args, quantized_nodes, amp_dtype):
  nodes = {n['name'] for n in json.loads(sym_fp32[0].tojson())['nodes'] if n['op'] != 'null'}
  quant_excluded_nodes = list(nodes - set(quantized_nodes))

  # Check if amp (after the fuse) changes the name of tensors for calibration

  sym_fp32 = sym_fp32.get_backend_symbol(SG_PASS_NAME)
  a, calib_tensors1 = mx.contrib.quantization._quantize_symbol(
      sym_fp32, mx.cpu(), excluded_symbols=quant_excluded_nodes)

  sym_lp, _, _ = amp.convert_model(sym_fp32, args, {}, target_dtype=amp_dtype,
                                   excluded_sym_names=quantized_nodes,
                                   cast_optional_params=True)
  sym_sg_lp = sym_lp.get_backend_symbol(AMP_SG_PASS_NAME)  # fuse amp casts
  b, calib_tensors2 = mx.contrib.quantization._quantize_symbol(
      sym_sg_lp, mx.cpu(), excluded_symbols=quant_excluded_nodes)

  outputs = {out for sym in sym_sg_lp.get_internals() for out in sym.list_outputs()}
  assert calib_tensors1 == calib_tensors2


def same_graph_structure(symnet1, symnet2, expected):
    nodes1 = json.loads(symnet1.tojson(remove_amp_cast=False))['nodes']
    nodes2 = json.loads(symnet2.tojson(remove_amp_cast=False))['nodes']
    assert (len(nodes1) == len(nodes2)) == expected
    for node1, node2 in zip(nodes1, nodes2):
        if node1['op'] != node2['op'] or node1['inputs'] != node2['inputs']:
          assert expected == False
          break


def check_amp_fuse(sym_fp32, args, dtype, expected_sym=None, quantized_nodes=[],
                   rtol=0.05, inputs_casted=True):
  args = args.copy()
  ex_ref = sym_fp32._bind(mx.cpu(), args)
  out_ref = ex_ref.forward()[0]

  sym_sg_fp32 = sym_fp32.get_backend_symbol(SG_PASS_NAME)  # amp pass works only on onednn nodes
  sym_sg_lp, args_sg_lp, _ = amp.convert_model(sym_sg_fp32, args, {}, target_dtype=dtype,
                                               excluded_sym_names=quantized_nodes,
                                               cast_optional_params=True)
  sym_sg_lp_no_args, _, _ = amp.convert_model(sym_sg_fp32, {}, {}, target_dtype=dtype,
                                              excluded_sym_names=quantized_nodes,
                                              cast_optional_params=True)
  sym_sg_lp = sym_sg_lp.get_backend_symbol(AMP_SG_PASS_NAME)
  sym_sg_lp_no_args = sym_sg_lp_no_args.get_backend_symbol(AMP_SG_PASS_NAME)
  ex_sg_lp = sym_sg_lp._bind(mx.cpu(), args_sg_lp)

  out_sg_lp = ex_sg_lp.forward()[0]

  # check outputs
  assert_almost_equal(out_ref, out_sg_lp, rtol=rtol, atol=1.0)

  # check graph
  if expected_sym is not None:
    same_graph_structure(sym_sg_lp, expected_sym, True)
    if inputs_casted:
      # when cast_optional_params == True, inputs should not be casted when they are not
      # provieded in amp.convert_model call
      same_graph_structure(sym_sg_lp_no_args, expected_sym, False)

  # check amp with quantization
  check_amp_with_quantization(sym_fp32, args, quantized_nodes, dtype)


def test_amp_fc():
  data_nd = mx.random.uniform(-1, 1, shape=(1, 8), dtype='float32')
  weight1_nd = mx.random.normal(-0.1, 0.1, shape=(16, data_nd.shape[1]), dtype='float32')
  weight2_nd = mx.random.normal(-0.1, 0.1, shape=(8, weight1_nd.shape[0]), dtype='float32')
  bias1_nd = mx.random.normal(-0.1, 0.1, shape=(weight1_nd.shape[0],), dtype='float32')
  data = mx.symbol.Variable('data')
  weight1 = mx.symbol.Variable('weight1')
  weight2 = mx.symbol.Variable('weight2')
  bias1 = mx.symbol.Variable('bias1')
  fc1 = mx.symbol.FullyConnected(data=data, weight=weight1, bias=bias1,
                                 num_hidden=weight1_nd.shape[0])
  sym = mx.symbol.FullyConnected(data=fc1, weight=weight2, no_bias=True,
                                 num_hidden=weight2_nd.shape[0])
  args = {
      'data': data_nd,
      'weight1': weight1_nd,
      'weight2': weight2_nd,
      'bias1': bias1_nd
  }

  exp_fc1 = mx.symbol.FullyConnected(data=data, weight=weight1, bias=bias1,
                                     num_hidden=weight1_nd.shape[0])
  exp_sym = mx.symbol.FullyConnected(data=exp_fc1, weight=weight2, no_bias=True,
                                     num_hidden=weight2_nd.shape[0])
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)

  check_amp_fuse(sym, args, 'bfloat16', exp_sym, ['sg_onednn_fully_connected_1'])


def test_amp_conv():
  data_nd = mx.random.uniform(-1, 1, shape=(1, 3, 8, 8), dtype='float32')
  weight1_nd = mx.random.normal(-0.1, 0.1, shape=(8, 3, 3, 3), dtype='float32')
  weight2_nd = mx.random.normal(-0.1, 0.1, shape=(4, 8, 3, 3), dtype='float32')
  bias1_nd = mx.random.normal(-0.1, 0.1, shape=(weight1_nd.shape[0],), dtype='float32')
  data = mx.symbol.Variable('data')
  weight1 = mx.symbol.Variable('weight1')
  weight2 = mx.symbol.Variable('weight2')
  bias1 = mx.symbol.Variable('bias1')
  conv1 = mx.symbol.Convolution(data=data, weight=weight1, bias=bias1, kernel=weight1_nd.shape[2:],
                                num_filter=weight1_nd.shape[0])
  sym = mx.symbol.Convolution(data=conv1, weight=weight2, no_bias=True, kernel=weight2_nd.shape[2:],
                              num_filter=weight2_nd.shape[0])
  args = {
      'data': data_nd,
      'weight1': weight1_nd,
      'weight2': weight2_nd,
      'bias1': bias1_nd
  }

  exp_conv1 = mx.symbol.Convolution(data=data, weight=weight1, bias=bias1, kernel=weight1_nd.shape[2:],
                                    num_filter=weight1_nd.shape[0])
  # this should be deleted after conv is fixed in oneDNN
  exp_amp_cast = mx.symbol.amp_cast(exp_conv1, dtype=bfloat16)
  exp_sym = mx.symbol.Convolution(data=exp_amp_cast, weight=weight2, no_bias=True, kernel=weight2_nd.shape[2:],
                                  num_filter=weight2_nd.shape[0])
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)

  check_amp_fuse(sym, args, 'bfloat16', exp_sym, ['sg_onednn_conv_1'])


@use_np
def test_amp_transformers():
  batch_size = 16
  seq_length = 32
  units = 8
  num_heads = 8

  in_data = mx.np.random.uniform(size=[batch_size, seq_length, units], dtype='float32')
  mask = mx.np.random.randint(0, 2, [batch_size, seq_length, seq_length], dtype='int32')

  net = MultiHeadAttention(units, num_heads)
  net.initialize()
  net.hybridize()
  net.optimize_for(in_data, mask, backend=SG_PASS_NAME)
  net(in_data, mask)
  params = {name: param.data() for name, param in net.collect_params().items()}
  params['data0'] = in_data
  params['data1'] = mask
  net.export('test', 0, False)
  sym, _ = net.export(None)
  check_amp_fuse(sym, params, 'bfloat16', None)
  check_amp_fuse(sym, params, 'bfloat16', None, ['sg_onednn_fully_connected_0'])


def test_amp_common_params():
  data_nd = mx.random.uniform(-1, 1, shape=(1, 8), dtype='float32')
  weight_nd = mx.random.normal(-0.1, 0.1, shape=(16, data_nd.shape[1]), dtype='float32')
  bias_nd = mx.random.normal(-0.1, 0.1, shape=(weight_nd.shape[0],), dtype='float32')
  data = mx.symbol.Variable('data')
  weight = mx.symbol.Variable('weight')
  bias = mx.symbol.Variable('bias')
  fc1 = mx.symbol.FullyConnected(data=data, weight=weight, bias=bias,
                                 num_hidden=weight_nd.shape[0])
  fc2 = mx.symbol.FullyConnected(data=data, weight=weight, bias=bias,
                                 num_hidden=weight_nd.shape[0])
  fc3 = mx.symbol.FullyConnected(data=data, weight=weight, bias=bias,
                                 num_hidden=weight_nd.shape[0])
  sym = mx.symbol.Concat(fc1, fc2, fc3)

  args = {
      'data': data_nd,
      'weight': weight_nd,
      'bias': bias_nd
  }

  exp_sym = mx.symbol.amp_cast(sym, 'float32')
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)
  check_amp_fuse(sym, args, 'bfloat16', exp_sym)

  exp_amp_data = mx.symbol.amp_cast(data, dtype=bfloat16)
  exp_amp_wieght = mx.symbol.amp_cast(weight, dtype=bfloat16)
  exp_amp_bias = mx.symbol.amp_cast(bias, dtype=bfloat16)
  exp_fc1 = mx.symbol.FullyConnected(data=exp_amp_data, weight=exp_amp_wieght, bias=exp_amp_bias,
                                     num_hidden=weight_nd.shape[0])
  exp_fc2 = mx.symbol.FullyConnected(data=exp_amp_data, weight=exp_amp_wieght, bias=exp_amp_bias,
                                     num_hidden=weight_nd.shape[0])
  exp_fc3 = mx.symbol.FullyConnected(data=data, weight=weight, bias=bias,
                                     num_hidden=weight_nd.shape[0])
  exp_mc = mx.symbol.amp_multicast(exp_fc1, exp_fc2, exp_fc3, num_outputs=3)
  exp_sym = mx.symbol.Concat(*exp_mc[:3])
  exp_sym = exp_sym.get_backend_symbol(SG_PASS_NAME)

  check_amp_fuse(sym, args, 'bfloat16', exp_sym, ['sg_onednn_fully_connected_2'],
                 inputs_casted=False)


def test_amp_fuse_with_branch():
  class TestNet(mx.gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(TestNet, self).__init__(**kwargs)
        self.fc1 = mx.gluon.nn.Dense(16)
        self.fc2 = mx.gluon.nn.Dense(16)

    def forward(self, x, *args):
        out = self.fc1(x)
        out1 = self.fc2(out)
        out2 = mx.npx.softmax(out)
        return out1, out2

  net = TestNet()
  net.initialize()
  net.hybridize()
  net.optimize_for(mx.np.ones((10,)), backend='ONEDNN')
  net_lp = amp.convert_hybrid_block(net, target_dtype=bfloat16,
                                    cast_optional_params=True, device=mx.cpu())
  net_lp.optimize_for(mx.np.ones((10,)), backend='ONEDNN_AMP')

  exp_data = mx.sym.Variable('data')
  exp_weights1 = mx.sym.Variable('fc1.weight')
  exp_bias1 = mx.sym.Variable('fc1.bias')
  exp_weights2 = mx.sym.Variable('fc2.weight')
  exp_bias2 = mx.sym.Variable('fc2.bias')
  exp_amp_cast1 = mx.sym.amp_cast(exp_data, dtype=bfloat16)
  exp_fc1 = mx.sym.FullyConnected(exp_amp_cast1, exp_weights1, exp_bias1, num_hidden=16)
  exp_fc2 = mx.sym.FullyConnected(exp_fc1, exp_weights2, exp_bias2, num_hidden=16)
  exp_amp_cast2 = mx.sym.amp_cast(exp_fc1, dtype='float32')
  exp_softmax = mx.sym.softmax(exp_amp_cast2)
  exp_symnet = mx.sym.Group([exp_fc2, exp_softmax])
  exp_symnet = exp_symnet.get_backend_symbol(SG_PASS_NAME)

  symnet, _ = net_lp.export(None, remove_amp_cast=False)
  same_graph_structure(symnet, exp_symnet, expected=True)
