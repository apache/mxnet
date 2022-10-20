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
import pytest
from subgraph_common import check_fusion, check_neg_fusion, check_neg_fusion_quantized, check_quantize
from subgraph_common import CustomNormalInit, DATA_SHAPE, TailNegBlock
from mxnet.contrib import quantization
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal_with_err

fc_post_ops_list=['relu', 'sigmoid', 'log_sigmoid', 'mish', 'tanh', 'softrelu', 'gelu', 'elu', 'leaky',
                  'square', 'square_root', 'abs', 'exp', 'bounded_relu']

def test_float64_fallback():
  dtype = 'float64'
  net = nn.Dense(units=3, dtype=dtype)
  in_data = mx.np.random.normal(size=[3,3,3,3], dtype=dtype)
  net.initialize()
  out = net(in_data)
  out.wait_to_read()
  assert in_data.dtype == out.dtype

@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_single_fc(data_shape, use_bias, flatten):

  class SingleFC(nn.HybridBlock):
    def __init__(self, use_bias, flatten, **kwargs):
      super(SingleFC, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten)

    def forward(self, x):
      return self.fc(x)

  attrs = {'fc': {}}
  net = SingleFC(use_bias, flatten)
  check_fusion(net, data_shape, attrs, check_quantization=flatten)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
@pytest.mark.parametrize('module', [mx.npx, mx.nd])
def test_fc_reshape(data_shape, use_bias, out_type, flatten, module):

  class FC_Reshape(nn.HybridBlock):
    def __init__(self, use_bias, flatten, **kwargs):
      super(FC_Reshape, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten)

    def forward(self, x):
      out = self.fc(x)
      if module == mx.npx:
        attrs = {"newshape": (1,-1)}
      else:
        attrs = {"shape": (1,-1)}
        out = out.as_nd_ndarray()
      out = getattr(module, "reshape")(out, **attrs)
      return out.as_np_ndarray()

  net = FC_Reshape(use_bias, flatten)
  check_quantize(net, data_shape, out_type, name='fc')

@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
@pytest.mark.parametrize('module', [mx.np, mx.nd])
def test_fc_transpose(data_shape, use_bias, out_type, module):

  class FC_Transpose(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
      super(FC_Transpose, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias)

    def forward(self, x):
      out = self.fc(x)
      if module == mx.nd:
        out = out.as_nd_ndarray()
      out = module.transpose(out)
      return out.as_np_ndarray()

  net = FC_Transpose(use_bias)
  check_quantize(net, data_shape, out_type, name='fc')

@mx.util.use_np
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
                         weight_initializer=CustomNormalInit(mean=0.5, sigma=0.1, bounded=True) if alg == 'square_root' else None)
                                            #avoid calculating square root of negative values
      self.alg = alg

    def forward(self, x):
      if self.alg == 'square_root':
        x = abs(x)
      fc_out = self.fc(x)
      if self.alg in ['relu', 'sigmoid', 'log_sigmoid', 'mish', 'tanh', 'softrelu']:
        out = mx.npx.activation(fc_out, act_type=self.alg)
      elif self.alg in ['gelu', 'gelu_tanh', 'elu', 'leaky']:
        out = mx.npx.leaky_relu(fc_out, act_type=self.alg)
      elif self.alg == 'square':
        out = mx.np.square(fc_out)
      elif self.alg == 'square_root':
        out = mx.np.sqrt(fc_out)
      elif self.alg == 'abs':
        out = mx.np.abs(fc_out)
      elif self.alg == 'exp':
        out = mx.np.exp(fc_out)
      else:
        out = mx.np.clip(fc_out, 0, 1.0)
      return out

  attrs = {'fc': {'with_eltwise': 'true'}}
  net = FCEltwise(use_bias, flatten, alg)
  check_fusion(net, data_shape, attrs, check_quantization=flatten)


@mx.util.use_np
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

    def forward(self, x):
      fc_out = self.fc(x)
      return self.tail_neg(self.act1(fc_out), self.act2(fc_out))


  attrs, excluded_attrs = [], []
  net = NegFCReLU(use_bias, flatten)
  check_neg_fusion(net, attrs, excluded_attrs, data_shape, name='fc')


@mx.util.use_np
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
  data_nd = mx.np.random.uniform(data_min, data_max, size=data_shape, device=mx.cpu())
  weight_nd = mx.np.random.uniform(weight_min, weight_max, size=[64, 32], device=mx.cpu())
  bias_nd = mx.np.random.uniform(-1, +1, size=[64], device=mx.cpu())

  class FCBiasOverflow(nn.HybridBlock):
    def __init__(self, dtype='float32', **kwargs):
        super(FCBiasOverflow, self).__init__(**kwargs)
        self.weight = mx.gluon.Parameter('weight', dtype=dtype, allow_deferred_init=True)
        self.bias = mx.gluon.Parameter('bias', dtype=dtype, allow_deferred_init=True)

    def forward(self, x):
        conv1 = mx.npx.fully_connected(x, num_hidden=64, weight=self.weight.data(x.device),
                                       no_bias=False, bias=self.bias.data(x.device))
        return conv1

    def infer_shape(self, x, *args):
        self.weight.shape = (64, x.shape[x.ndim-1])
        self.bias.shape = (64,)

  net = FCBiasOverflow()
  net.initialize()
  net(data_nd) # dummy run

  net.weight.data()[:] = weight_nd
  net.bias.data()[:] = bias_nd
  out = net(data_nd)

  calib_data = mx.gluon.data.DataLoader(data_nd, batch_size=1)
  qnet = quantization.quantize_net(net,
                                   device=mx.cpu(),
                                   exclude_layers=None,
                                   exclude_operators=None,
                                   quantized_dtype='int8',
                                   calib_mode='naive',
                                   calib_data=calib_data,
                                   num_calib_batches=1,
                                   quantize_mode='full')
  out_quantized = qnet(data_nd)
  assert_almost_equal_with_err(out.asnumpy(), out_quantized.asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('flatten', [True, False])
def test_fc_int8_and_fp32_outputs(data_shape, flatten):

#                 /---> Quantizable op
# Input ---> FC -|
#                 \---> Non quantizable op

  class MultiOutputFC(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(MultiOutputFC, self).__init__(**kwargs)
      self.dense0 = nn.Dense(64, flatten=flatten)
      self.dense1 = nn.Dense(64, flatten=flatten)

    def forward(self, x):
      x = self.dense0(x)
      y = self.dense1(x)      # quantizable
      z = mx.npx.softmax(x)   # non quantizable
      return y + z

  attrs = {'fc': {}}
  net = MultiOutputFC()
  check_fusion(net, data_shape, attrs, check_quantization=flatten)


@mx.util.use_np
@pytest.mark.parametrize('identity_node', ['dropout', 'copy'])
def test_fc_identity_eltwise(identity_node):
  class FCIdentityEltwise(nn.HybridBlock):
    def __init__(self, identity_node, **kwargs):
      super(FCIdentityEltwise, self).__init__(**kwargs)
      self.fc1 = nn.Dense(units=64, use_bias=False, weight_initializer=None, flatten=True)
      self.fc2 = nn.Dense(units=64, use_bias=False, weight_initializer=None, flatten=True)
      self.identity_node = identity_node

    def forward(self, x):
      out = self.fc1(x)
      if self.identity_node == 'copy':
        out = mx.np.copy(out)
      else:
        out = mx.npx.dropout(out)
      out = mx.npx.activation(out, act_type='relu')
      out = self.fc2(out)
      if self.identity_node == 'copy':
        out = mx.np.copy(out)
      else:
        out = mx.npx.dropout(out)
      out = mx.npx.activation(out, act_type='relu')
      return out

  data_shape = (64, 4, 10, 10)
  attrs = {'sg_onednn_fully_connected_eltwise_0' : {'with_eltwise': 'true'},
           'sg_onednn_fully_connected_eltwise_1' : {'with_eltwise': 'true'}}
  net = FCIdentityEltwise(identity_node)
  check_fusion(net, data_shape, attrs, check_quantization=False)


def function_fc_add(data_shape, add_op, quantize_mode, fc_out_add, flatten, relu, out_type):
  class FCWithSumExample(nn.HybridBlock):
    def __init__(self,  num_hidden, add_op, fc_out_add, **kwargs):
      super(FCWithSumExample, self).__init__(**kwargs)
      self.fca = nn.Dense(units=num_hidden, flatten=flatten)
      self.elemwise_add = (add_op == 'elemwise_add')
      self.fc_out_as_rhs = (fc_out_add == 'rhs')
      self.relu = (relu == 'leaky_relu')

    def forward(self, data1a, data2):
      fc_out = self.fca(data1a)
      if self.relu:
        fc_out = mx.npx.leaky_relu(fc_out, act_type='gelu')
      if self.fc_out_as_rhs:
        if  self.elemwise_add:
          sum1 = mx.nd.elemwise_add(data2.as_nd_ndarray(), fc_out.as_nd_ndarray()).as_np_ndarray()
        else:
          sum1 = data2 + fc_out
      else:
        if  self.elemwise_add:
          sum1 = mx.nd.elemwise_add(fc_out.as_nd_ndarray(), data2.as_nd_ndarray()).as_np_ndarray()
        else:
          sum1 = fc_out + data2
      return sum1

  attrs = {'fc': {'with_sum': 'true'}}
  if quantize_mode is not None:
    attrs['fc']['quantized'] = 'true'
    if quantize_mode == 'smart':
      attrs['fc']['enabled_float_output'] = mx.nd.get_dtype_name(mx.np.float32)
  num_hidden=10
  net = FCWithSumExample(num_hidden, add_op, fc_out_add)
  if flatten:
    data_shapes = [data_shape, (data_shape[0], num_hidden)]
  else:
    data_shapes = [data_shape, (*data_shape[0:-1], num_hidden)]
  check_fusion(net, data_shapes, attrs,
               out_types=[out_type],
               check_fusion=(quantize_mode is None),
               check_quantization=(quantize_mode is not None) and flatten,
               quantize_mode=quantize_mode)

@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('relu', ['noleaky_re', 'leaky_relu'])
@pytest.mark.parametrize('flatten', ['flat', 'nofl'])
@pytest.mark.parametrize('fc_out_add', ['lhs', 'rhs'])
@pytest.mark.parametrize('add_op', ['elemwise_add'])
def test_fc_add(data_shape, add_op, fc_out_add, flatten, relu):
  function_fc_add(data_shape, add_op, None, fc_out_add, flatten=='flat', relu, None)

@mx.util.use_np
@pytest.mark.seed(1234) # Seed set because the test is not robust enough to operate on random data
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('quantize_mode', ['full', 'smart'])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
@pytest.mark.parametrize('fc_out_add', ['lhs', 'rhs'])
@pytest.mark.parametrize('add_op', ['elemwise_add'])
def test_fc_add_quantized(data_shape, add_op, quantize_mode, fc_out_add, out_type):
  function_fc_add(data_shape, add_op, quantize_mode, fc_out_add, True, 'noleaky_re', out_type)


class NegFCAdd(nn.HybridBlock):
  #
  #  data  --------------------------> 'add_op'  ------------>
  #                                   /                        \
  #  sg_oned_dnn_fully_connected ---->                           npi_add -->
  #                                   \                        /
  #                                    npi_multiply_scalar -->
  def __init__(self, num_hidden, add_op, fc_out_add, scaled_fc_out, flatten, **kwargs):
    super(NegFCAdd, self).__init__(**kwargs)
    self.fca = nn.Dense(units=num_hidden, flatten=flatten)
    self.elemwise_add = (add_op == 'elemwise_add')
    self.fc_out_as_rhs = (fc_out_add == 'rhs')
    self.scaled_fc_out_as_rhs = (scaled_fc_out == 's_rhs')

  def forward(self, data1a, data2):
    fc_out = self.fca(data1a)
    scaled_fc_out = fc_out * 200.0
    if self.fc_out_as_rhs:
      if  self.elemwise_add:
        sum1 = mx.nd.elemwise_add(data2.as_nd_ndarray(), fc_out.as_nd_ndarray()).as_np_ndarray()
      else:
        sum1 = data2 + fc_out
    else:
      if  self.elemwise_add:
        sum1 = mx.nd.elemwise_add(fc_out.as_nd_ndarray(), data2.as_nd_ndarray()).as_np_ndarray()
      else:
        sum1 = fc_out + data2
    if self.scaled_fc_out_as_rhs:
      sum2 = sum1 + scaled_fc_out
    else:
      sum2 = scaled_fc_out + sum1
    return sum2

@mx.util.use_np
@pytest.mark.parametrize('add_op', ['elemwise_add'])
@pytest.mark.parametrize('data_shape', [DATA_SHAPE[0]])
@pytest.mark.parametrize('flatten', ['flat', 'nofl'])
@pytest.mark.parametrize('fc_out_add', ['lhs', 'rhs'])
@pytest.mark.parametrize('scaled_fc_out', ['s_lhs', 's_rhs'])
def test_neg_fc_add(data_shape, add_op, flatten, fc_out_add, scaled_fc_out):
  '''
  Test if FullyConnected operator which output is not used for only one 'add_op' input is not fused.
  See NegFCAdd for used graph example
  '''
  flatten = (flatten == 'flat')
  num_hidden = 10
  net = NegFCAdd(num_hidden, add_op, fc_out_add, scaled_fc_out, flatten)
  if flatten:
    data_shapes = [data_shape, (data_shape[0], num_hidden)]
  else:
    data_shapes = [data_shape, (*data_shape[0:-1], num_hidden)]
  attrs = []
  excluded_attrs = ['with_sum']
  check_neg_fusion(net, attrs, excluded_attrs, data_shapes, name='fc')

@mx.util.use_np
@pytest.mark.parametrize('add_op', ['elemwise_add'])
@pytest.mark.parametrize('data_shape', [DATA_SHAPE[1]])
@pytest.mark.parametrize('fc_out_add', ['lhs', 'rhs'])
@pytest.mark.parametrize('scaled_fc_out', ['s_lhs', 's_rhs'])
def test_neg_fc_add_quantized(data_shape, add_op, fc_out_add, scaled_fc_out):
  '''
  Test if FullyConnected operator which output is not used for only one 'add_op' input
  is not fused for quantized model.
  See NegFCAdd for used graph example.
  '''
  num_hidden = 10
  net = NegFCAdd(num_hidden, add_op, fc_out_add, scaled_fc_out, True)
  data_shapes = [data_shape, (data_shape[0], num_hidden)]
  attrs = []
  excluded_attrs = ['with_sum']
  check_neg_fusion_quantized(net, attrs, excluded_attrs, data_shapes, name='fc')


def function_add_quantized(data_shape, add_op, quantize_mode, relu, out_type, broadcast, calib_mode):
  class SumExample(nn.HybridBlock):
    def __init__(self,  add_op, **kwargs):
      super(SumExample, self).__init__(**kwargs)
      self.elemwise_add = (add_op == 'ele_add')
      self.relu = (relu == 'relu')

    def forward(self, data1a, data2):
      fc_out = data1a
      if self.relu:
        fc_out = mx.npx.relu(fc_out)
      if  self.elemwise_add:
        sum1 = mx.nd.elemwise_add(data2.as_nd_ndarray(), fc_out.as_nd_ndarray()).as_np_ndarray()
      else:
        sum1 = data2 + fc_out
      return sum1

  attrs = {add_op: {}}
  net = SumExample(add_op)
  if broadcast:
    broadcasted_shape = (1,) + data_shape[1:-1] + (1,)
    data_shapes = [broadcasted_shape, data_shape]
  else:
    data_shapes = [data_shape, data_shape]

  # check_calibration could be enabled if check_qsym_calibrated will be reimplemented
  # to find operator names instead of node names
  check_quantize(net, data_shapes, out_type, name="contrib_quantized_" + add_op,
                 quantize_mode=quantize_mode, attrs_dict=attrs, calib_mode=calib_mode,
                 check_calibration=(calib_mode != 'none') and False, check_fusion=False)


@mx.util.use_np
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
@pytest.mark.parametrize('calib_mode', ['naive', 'none'])
@pytest.mark.parametrize('quantize_mode', ['full', 'smart'])
@pytest.mark.parametrize('relu', ['nore', 'relu'])
@pytest.mark.parametrize('broadcast', ['broadcast', 'no_broadc'])
@pytest.mark.parametrize('add_op', ['ele_add', 'npi_add'])
def test_add_quantized(add_op, quantize_mode, out_type, relu, broadcast, calib_mode):
  """
  The test check results from quantization of simple graph
  with npi_add or elemwise_add with additional relu which force
  unsigned representation of one inputs to the add operator.
  Due to construction of quantization code unsigned int8 is never choosen
  for scenario without calibration as operators always raports min = -max
  """
  broadcastB = (broadcast ==  'broadcast')
  if broadcastB and add_op == 'ele_add':
    # elemwise_Add doesn't support broadcasting
    pytest.skip()
  data_shape = DATA_SHAPE[0]
  function_add_quantized(data_shape, add_op, quantize_mode, relu, out_type, broadcastB, calib_mode)
