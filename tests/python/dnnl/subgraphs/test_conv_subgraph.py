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
import pytest
from subgraph_common import check_fusion, check_neg_fusion, check_quantize
from subgraph_common import CustomNormalInit, DATA_SHAPE, RELU6, TailNegBlock
from subgraph_common import DATA_SHAPE, SG_PASS_NAME, QUANTIZE_SG_PASS_NAME
from mxnet.contrib import quantization
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal, assert_almost_equal_with_err

mx.npx.reset_np()

@mx.util.use_np
def test_float64_fallback():
  class ConvWithDtype(nn.HybridBlock):
    def __init__(self, dtype='float32', **kwargs):
        super(ConvWithDtype, self).__init__(**kwargs)
        self.weight = mx.gluon.Parameter('weight', dtype=dtype, allow_deferred_init=True)
        self.bias = mx.gluon.Parameter('bias', dtype=dtype, allow_deferred_init=True)

    def forward(self, x):
        out = mx.npx.convolution(x, kernel=(1,1), num_filter=3,
                                 weight=self.weight.data(x.device), no_bias=False,
                                 bias=self.bias.data(x.device))
        return out
    
    def infer_shape(self, x):
        self.weight.shape = (3, 3, 1, 1)
        self.bias.shape = (3,)

  dtype = 'float64'
  net = ConvWithDtype(dtype=dtype)
  in_data = mx.np.random.normal(size=[3,3,3,3], dtype=dtype)
  net.initialize()
  out = net(in_data)
  out.wait_to_read()
  assert in_data.dtype == out.dtype


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_single_conv(use_bias, data_shape):
  # single conv fusion case
  class Conv(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Conv, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)

    def forward(self, x):
        out = self.conv0(x)
        return out

  attr = {'conv': []}
  net = Conv()
  check_fusion(net, data_shape, attr)

@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
@pytest.mark.parametrize('module', [mx.np, mx.nd])
def test_conv_transpose_conv(use_bias, data_shape, out_type, module):

  class Conv_Transpose_Conv(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Conv_Transpose_Conv, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=32, kernel_size=(5, 5), strides=1, use_bias=use_bias)

    def forward(self, x):
      out = self.conv0(x)
      if module == mx.nd:
        out = out.as_nd_ndarray()
      out = module.transpose(out, axes = [0,1,3,2])
      out = self.conv1(out.as_np_ndarray())
      return out

  net = Conv_Transpose_Conv()
  check_quantize(net, data_shape, out_type)

@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
@pytest.mark.parametrize('module', [mx.npx, mx.nd])
def test_conv_reshape_conv(use_bias, data_shape, out_type, module):

  class Conv_Reshape_Conv(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Conv_Reshape_Conv, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=32, kernel_size=(5, 5), strides=1, use_bias=use_bias)

    def forward(self, x):
      out = self.conv0(x)
      if module == mx.npx:
        attrs = {"newshape": (-1, int(out.shape[1]/4), out.shape[2]*2, out.shape[3]*2)}
      else:
        attrs = {"shape": (-1, int(out.shape[1]/4), out.shape[2]*2, out.shape[3]*2)}
        out = out.as_nd_ndarray()
      out = getattr(module, "reshape")(out, **attrs)
      out = self.conv1(out.as_np_ndarray())
      return out

  net = Conv_Reshape_Conv()
  check_quantize(net, data_shape, out_type)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_add(use_bias, data_shape):
  # conv + add fusion case
  class ConvAdd(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1)
        self.pool = nn.AvgPool2D(pool_size=(1,1))

    def forward(self, x):
      out = self.conv0(x) + self.pool(self.conv1(x))
      return out
    
  attr = {'conv': {'with_sum': 'true'}}
  net = ConvAdd(use_bias=use_bias)
  check_fusion(net, data_shape, attr, check_quantization=False)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
def test_pos_conv_add2(no_bias, data_shape):
  # conv + add fusion case 2
  class ConvAdd(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1)
        self.pool = nn.AvgPool2D(pool_size=(1,1))

    def forward(self, x):
      out = self.pool(self.conv1(x)) + self.conv0(x)
      return out

  attr = {'conv': {'with_sum': 'true'}}
  net = ConvAdd(use_bias=True)
  check_fusion(net, data_shape, attr, check_quantization=False)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_conv_add3(no_bias, data_shape, out_type):
  # conv + add fusion case 3
  class ConvAdd(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=data_shape[1], kernel_size=(1, 1), strides=1, use_bias=use_bias)

    def forward(self, x):
      out = x + self.conv0(x)
      return out

  net = ConvAdd(use_bias=True)
  check_quantize(net, data_shape, out_type)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_conv_add4(no_bias, data_shape, out_type):
  # conv + add fusion case 4
  class ConvAdd(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=data_shape[1], kernel_size=(1, 1), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)

    def forward(self, x):
      out = self.conv1(x + self.conv0(x))
      return out

  net = ConvAdd(use_bias=True)
  check_quantize(net, data_shape, out_type)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", False), #TODO(bgawrych): investigate
    ("sigmoid", False),
    ("log_sigmoid", False),
    ("mish", False),
    ("tanh", False), #TODO(bgawrych): investigate
    #("softrelu", True), #TODO(bgawrych): bug in oneDNN with AVX
    ("relu6", False), #TODO(bgawrych): investigate
    ("leakyrelu", True),
    ("gelu", True),
    ("gelu_tanh", True)
])
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_act_add(data_shape, alg, quantize, use_bias):
# conv + act + add fusion case
  class ConvActAdd(nn.HybridBlock):
    def __init__(self, use_bias, alg, **kwargs):
        super(ConvActAdd, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias,
                               weight_initializer=mx.init.Xavier(magnitude=2.24))
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        elif alg == "gelu_tanh":
          self.act = nn.GELU(approximation='tanh')
        else:
          self.act = nn.Activation(activation = alg)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)

    def forward(self, x):
        out = self.act(self.conv0(x)) + self.conv1(x)
        return out

  attrs = {'sg_onednn_conv_act_0': {'with_act': 'true'},
           'sg_onednn_conv_add_1': {'with_sum': 'true'}}

  net = ConvActAdd(use_bias, alg)
  check_fusion(net, data_shape, attrs, check_quantization=quantize)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", True),
    ("sigmoid", False),
    ("log_sigmoid", False),
    ("mish", False),
    ("tanh", False),
    ("softrelu", False),
    ("relu6", True),
    ("leakyrelu", True),
    ("gelu", True),
    ("gelu_tanh", True)
])
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_bn_act(use_bias, data_shape, alg, quantize):
# conv + bn + act fusion case
  class ConvBNAct(nn.HybridBlock):
    def __init__(self, alg, use_bias, **kwargs):
        super(ConvBNAct, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.bn = nn.BatchNorm()
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        elif alg == "gelu_tanh":
          self.act = nn.GELU(approximation='tanh')
        else:
          self.act = nn.Activation(activation = alg)

    def forward(self, x):
      out = self.act(self.bn(self.conv0(x)))
      return out

  attr = {'conv': {'with_bn': 'true', 'with_act': 'true'}}
  net = ConvBNAct(alg, use_bias)
  check_fusion(net, data_shape, attr, check_quantization=quantize)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", False),
    ("sigmoid", False),
    ("log_sigmoid", False),
    ("mish", False),
    ("tanh", False),
    #("softrelu", True), #TODO(bgawrych): failing fusion check - difference in random single element
    ("relu6", False),
    ("leakyrelu", False),
    ("gelu", False),
    ("gelu_tanh", False)
])
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_bn_sum_act(use_bias, data_shape, alg, quantize):
  # conv + bn + add + act fusion case
  class ConvBNSumAct(nn.HybridBlock):
    def __init__(self, alg, use_bias, **kwargs):
        super(ConvBNSumAct, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1)
        self.bn = nn.BatchNorm()
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        elif alg == "gelu_tanh":
          self.act = nn.GELU(approximation='tanh')
        else:
          self.act = nn.Activation(activation = alg)

    def forward(self, x):
        out = self.bn(self.conv0(x)) + self.conv1(x)
        out = self.act(out)
        return out

  attr = {'sg_onednn_conv_bn_add_act': {'with_sum': 'true', 'with_postsum_act': 'true', 'with_bn': 'true'}}
  net = ConvBNSumAct(alg, use_bias)
  check_fusion(net, data_shape, attr, check_quantization=quantize)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('input_num,dim', [
    (2, -1),
    (2, 1),
    (4, 2),
    (4, 3)
])
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_single_concat(data_shape, input_num, dim, out_type):
  # single concat case
  class SingleConcat(nn.HybridBlock):
    def __init__(self, input_num, dim, **kwargs):
        super(SingleConcat, self).__init__(**kwargs)
        self.concat = nn.HybridConcatenate(axis=dim)
        for _ in range(input_num):
            self.concat.add(nn.Identity())

    def forward(self, x):
        out = self.concat(x)
        return out

  concat = SingleConcat(input_num, dim)
  check_quantize(concat, data_shape, out_type, name='conv',
                  check_calibration=False)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_single_concat_pos_neg(data_shape, out_type):
  class ConvDataConcat(nn.HybridBlock):
    def __init__(self, dim, **kwargs):
        super(ConvDataConcat, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=4, kernel_size=(1, 1), strides=1, use_bias=False)
        self.act = nn.Activation(activation = 'relu')
        self.concat_dim = dim

    def forward(self, x):
        relu_out = self.act(self.conv0(x))
        out = mx.np.concatenate([x, relu_out], axis=self.concat_dim)
        return out

  concat = ConvDataConcat(dim=1)
  check_quantize(concat, data_shape, out_type, name='', check_calibration=False)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('out_type', ['int8', 'auto'])
def test_pos_concat_scale_align(data_shape, out_type):
  # concat scale alignment case
  class ConcatScaleAlign(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ConcatScaleAlign, self).__init__(**kwargs)
        self.shared_weight = mx.gluon.Parameter('shared_weight', shape=(64, data_shape[1], 3, 3),
                                                init=mx.init.Xavier(magnitude=2.24),
                                                dtype='float32', allow_deferred_init=True)

    def forward(self, x):
        conv1 = mx.npx.convolution(x, kernel=(3,3), num_filter=64,
                                   weight=self.shared_weight.data(x.device), no_bias=True)
        conv2 = mx.npx.convolution(x, kernel=(3,3), num_filter=64,
                                   weight=self.shared_weight.data(x.device)*2, no_bias=True)
        conv3 = mx.npx.convolution(x, kernel=(3,3), num_filter=64,
                                   weight=self.shared_weight.data(x.device)*3, no_bias=True)
        conv4 = mx.npx.convolution(x, kernel=(3,3), num_filter=64,
                                   weight=self.shared_weight.data(x.device)*4, no_bias=True)
        return mx.np.concatenate([conv1, conv2, conv3, conv4], axis=1)

    def infer_shape(self, x, *args):
        self.shared_weight.weight = (64, data_shape[1], 3, 3)

  concat = ConcatScaleAlign()
  check_quantize(concat, data_shape, out_type, check_calibration=True,
                  check_scale_align=True)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('alg,quantize', [
    ("relu", True),
    ("sigmoid", False),
    ("log_sigmoid", False),
    ("mish", False),
    ("tanh", False),
    ("softrelu", False),
    ("relu6", True),
    ("leakyrelu", True),
    ("gelu", True),
    ("gelu_tanh", True)
])
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_act(use_bias, data_shape, alg, quantize):
  # conv + act fusion case
  class ConvAct(nn.HybridBlock):
    def __init__(self, use_bias, alg, **kwargs):
        super(ConvAct, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        if alg == "relu6":
          self.act = RELU6()
        elif alg == "leakyrelu":
          self.act = nn.LeakyReLU(0.25)
        elif alg == "gelu":
          self.act = nn.GELU()
        elif alg == "gelu_tanh":
          self.act = nn.GELU(approximation='tanh')
        else:
          self.act = nn.Activation(activation = alg)

    def forward(self, x):
        out = self.act(self.conv0(x))
        return out

  attrs = {'conv': {'with_act': 'true'}}
  net = ConvAct(False, alg)
  check_fusion(net, data_shape, attrs, check_quantization=quantize)
  net = ConvAct(True, alg)
  check_fusion(net, data_shape, attrs, check_quantization=quantize)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
def test_pos_conv_bn(use_bias, data_shape):
  # conv + bn fusion case
  class ConvBN(nn.HybridBlock):
    def __init__(self, use_bias, **kwargs):
        super(ConvBN, self).__init__(**kwargs)
        self.conv0 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=1, use_bias=use_bias)
        self.bn = nn.BatchNorm()

    def forward(self, x):
        out = self.bn(self.conv0(x))
        return out

  attr = {'conv': {'with_bn': 'true'}}
  net = ConvBN(use_bias)
  check_fusion(net, data_shape, attr)


# used in multiple tests
class ConvBNSum(nn.HybridBlock):
  def __init__(self, channels, reverse_sum_order, **kwargs):
      super(ConvBNSum, self).__init__(**kwargs)
      self.conv0 = nn.Conv2D(channels=channels, kernel_size=(1, 1), strides=1, use_bias=False)
      self.bn = nn.BatchNorm()
      self.reverse = reverse_sum_order

  def forward(self, x):
      if self.reverse:
        return self.bn(self.conv0(x)) + x
      else:
        return x + self.bn(self.conv0(x))


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('reverse_sum_order', [True, False])
@pytest.mark.parametrize('dedup_subgraph', [True, False])
def test_conv_bn_sum(data_shape, reverse_sum_order, dedup_subgraph):
  attr = {'sg_onednn_conv_bn_add_0' : {'with_bn': 'true'}}
  # channels after conv+bn should be same as input channels
  net = ConvBNSum(channels=data_shape[1] ,reverse_sum_order=reverse_sum_order)
  check_fusion(net, data_shape, attr, out_types=['int8', 'auto'], dedup_subgraph=dedup_subgraph)


# used in multiple tests
class MobileNetV2Struct(nn.HybridBlock):
  def __init__(self, reverse_sum_order, **kwargs):
      super(MobileNetV2Struct, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(1, 1), strides=(1,1), use_bias=False)
      self.conv2 = nn.Conv2D(channels=64, kernel_size=(1, 1), strides=(1,1), use_bias=False)
      self.bn1 = nn.BatchNorm()
      self.bn2 = nn.BatchNorm()
      self.reverse = reverse_sum_order

  def forward(self, x):
      out = self.bn1(self.conv1(x))
      if self.reverse:
        return self.bn2(self.conv2(out)) + out
      else:
        return out + self.bn2(self.conv2(out))


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('reverse_sum_order', [True, False])
@pytest.mark.parametrize('dedup_subgraph', [True, False])
def test_mobilenetv2_struct(data_shape, reverse_sum_order, dedup_subgraph):
  attr = {'sg_onednn_conv_bn_0' : {'with_bn': 'true'}}
  net = MobileNetV2Struct(reverse_sum_order=reverse_sum_order)
  check_fusion(net, data_shape, attr, out_types=['int8', 'auto'], dedup_subgraph=dedup_subgraph)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('reverse_sum_order', [False, True])
@pytest.mark.parametrize('model_name', ['conv_bn_sum', 'mobilenetv2_struct'])
def test_deduplication(data_shape, reverse_sum_order, model_name):
  data_nd = mx.np.random.uniform(-1, 1, size=data_shape, device=mx.cpu())
  if (model_name == 'mobilenetv2_struct'):
    model_dedup = MobileNetV2Struct(reverse_sum_order=reverse_sum_order)
  else:
    # channels after conv+bn should be same as input channels
    model_dedup = ConvBNSum(channels=data_shape[1], reverse_sum_order=reverse_sum_order)

  model_dedup.initialize()
  model_no_dedup = copy.copy(model_dedup)

  model_dedup.optimize_for(data_nd, backend='ONEDNN', dedup_subgraph = True, skip_infer = True)
  out = model_dedup(data_nd)

  model_dedup.optimize_for(data_nd, backend='ONEDNN', dedup_subgraph = False, skip_infer = True)
  out_dedup = model_no_dedup(data_nd)

  assert_almost_equal(out.asnumpy(), out_dedup.asnumpy(), rtol=1e-3, atol=1e-1)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn(data_shape):
  # conv + bn can't be fusion case
  # eg.1
  # conv --------- > bn
  #  |
  #  |
  #  -------------> [custom op]
  class NegConvBN(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(NegConvBN, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.bn1 = nn.BatchNorm()
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()

    def forward(self, x):
      conv = self.conv1(x)
      bn = self.bn1(conv)
      pool = self.pool(conv)

      return self.tailneg(bn, pool)

  attrs = []
  excluded_attrs = []
  net = NegConvBN()
  check_neg_fusion(net, attrs, excluded_attrs, data_shape)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_relu(data_shape):
  # conv + relu can't be fusion case
  # eg.1
  # conv -----------> relu
  #  |
  #  |
  #  ---------------> [custom op]
  class NegConvReLU(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(NegConvReLU, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()

    def forward(self, x):
      conv = self.conv1(x)
      bn = self.act(conv)
      pool = self.pool(conv)
      return self.tailneg(bn, pool)

  attrs = []
  excluded_attrs = []
  net = NegConvReLU()
  check_neg_fusion(net, attrs, excluded_attrs, data_shape)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_add(data_shape):
  # conv + add can't be fusion case
  # eg.1
  #  ---------------> [custom op]
  #  |
  #  |
  # conv -----------> add
  #                   |
  #                   |
  # added ------------>
  class NegConvAdd(nn.HybridBlock):
    def __init__(self, **kwargs):
      super(NegConvAdd, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()
      self.add_value = mx.gluon.Parameter('add_value', init=mx.init.Xavier(magnitude=2.24),
                                          dtype='float32', allow_deferred_init=True)

    def forward(self, x):
      conv = self.conv1(x)
      print(conv.shape)
      sum1 = conv + self.add_value.data(x.device)
      pool = self.pool(conv)
      return self.tailneg(sum1, pool)
    
    def infer_shape(self, x):
      self.add_value.shape = (data_shape[0], 64, data_shape[2]-2, data_shape[3]-2)

  attrs = []
  excluded_attrs = ['with_sum']
  net = NegConvAdd()
  check_neg_fusion(net, attrs, excluded_attrs, data_shape)

@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_relu(data_shape):
  # conv + bn + relu can't be fusion case
  # eg.1
  #   --------------> [custom op]
  #   |
  # conv -----------> bn -----------> relu
  #
  # eg.2
  #                   --------------> [custom op]
  #                   |
  # conv -----------> bn -----------> relu
  class NegConvBNRelu(nn.HybridBlock):
    def __init__(self, batchnorm_pool = False, **kwargs):
      super(NegConvBNRelu, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.bn = nn.BatchNorm()
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()
      self.batchnorm_pool = batchnorm_pool

    def forward(self, x):
      conv = self.conv1(x)
      bn = self.bn(conv)
      relu = self.act(bn)
      pool = self.pool(bn) if self.batchnorm_pool else self.pool(conv)
      return self.tailneg(relu, pool)

  # eg.1 ([custom op] = pool11)
  net1 = NegConvBNRelu()
  attrs1 = []
  excluded_attrs1 = []
  check_neg_fusion(net1, attrs1, excluded_attrs1, data_shape)

  # eg.2 ([custom op] = pool)
  net2 = NegConvBNRelu(batchnorm_pool=True)
  attrs2 = ['with_bn']
  excluded_attrs2 = ['with_act']
  check_neg_fusion(net2, attrs2, excluded_attrs2, data_shape)


@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
def test_neg_conv_bn_add_relu(data_shape):
  # conv + bn + add + relu can't be fusion case
  # eg.1
  #   --------------> [custom op]
  #   |
  # conv -----------> bn -----------> add -----------> relu
  #
  # eg.2
  #                    -------------> [custom op]
  #                    |
  # conv -----------> bn -----------> add -----------> relu
  #
  # eg.3
  #                                    --------------> [custom op]
  #                                    |
  # conv -----------> bn -----------> add -----------> relu

  class NegConvBNAddRelu(nn.HybridBlock):
    def __init__(self, connect_mode = "conv_customop", **kwargs):
      super(NegConvBNAddRelu, self).__init__(**kwargs)
      self.conv1 = nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1,1), use_bias=False)
      self.bn = nn.BatchNorm()
      self.act = nn.Activation('relu')
      self.pool = nn.AvgPool2D(pool_size=(4,4))
      self.tailneg = TailNegBlock()
      self.connect_mode = connect_mode
      self.add_value = mx.gluon.Parameter('add_value', init=mx.init.Xavier(magnitude=2.24),
                                          dtype='float32', allow_deferred_init=True)

    def forward(self, x):
      conv = self.conv1(x)
      bn = self.bn(conv)
      print(bn.shape)
      sum1 = bn + self.add_value.data(x.device)
      relu = self.act(sum1)
      if self.connect_mode == "conv_customop":
        pool = self.pool(conv)
      elif self.connect_mode == "bn_customop":
        pool = self.pool(bn)
      else:
        pool = self.pool(sum1)
      return self.tailneg(relu, pool)

    def infer_shape(self, x):
      self.add_value.shape = (data_shape[0], 64, data_shape[2]-2, data_shape[3]-2)

  # eg.1
  net1 = NegConvBNAddRelu(connect_mode = "conv_customop")
  attrs1 = []
  excluded_attrs1 = ['with_sum', 'with_postsum_act', 'with_bn']
  check_neg_fusion(net1, attrs1, excluded_attrs1, data_shape)

  # eg.2
  net2 = NegConvBNAddRelu(connect_mode = "bn_customop")
  attrs2 = ['with_bn']
  excluded_attrs2 = ['with_sum', 'with_postsum_act']
  check_neg_fusion(net2, attrs2, excluded_attrs2, data_shape)

  # eg.3
  net3 = NegConvBNAddRelu(connect_mode = "add_customop")
  attrs3 = ['with_bn', 'with_sum']
  excluded_attrs3 = ['with_postsum_act']
  check_neg_fusion(net3, attrs3, excluded_attrs3, data_shape)



@mx.util.use_np
@pytest.mark.parametrize('data_min,data_max,weight_min,weight_max', [
    (-1, 1, 0, 0),
    (-1, 1, -1e-6, +1e-6),
    (0, 0, 1, 1),
    (-1e-6, +1e-6, -1, 1),
    (-1e-6, +1e-6, -1e-6, +1e-6),
    (0, 0, 0, 0)
])
def test_quantized_conv_bias_overflow(data_min, data_max, weight_min, weight_max):
  data_shape = (1, 32, 2, 2)
  data_nd = mx.np.random.uniform(data_min, data_max, size=data_shape, device=mx.cpu())
  weight_nd = mx.np.random.uniform(weight_min, weight_max, size=[64, 32, 1, 1], device=mx.cpu())
  bias_nd = mx.np.random.uniform(-1, +1, size=[64], device=mx.cpu())

  class ConvBiasOverflow(nn.HybridBlock):
        def __init__(self, dtype='float32', **kwargs):
            super(ConvBiasOverflow, self).__init__(**kwargs)
            self.weight = mx.gluon.Parameter('weight', dtype=dtype, allow_deferred_init=True)
            self.bias = mx.gluon.Parameter('bias', dtype=dtype, allow_deferred_init=True)

        def forward(self, x):
            conv1 = mx.npx.convolution(x, num_filter=64, kernel=(1,1),
                                       weight=self.weight.data(x.device),
                                       no_bias=False, bias=self.bias.data(x.device))
            return conv1
        
        def infer_shape(self, x):
            self.weight.shape = (64, x.shape[1], 1, 1)
            self.bias.shape = (64,)

  net = ConvBiasOverflow()
  net.initialize()
  net(data_nd) # dummy run

  net.weight.data()[:] = weight_nd
  net.bias.data()[:] = bias_nd
  out = net(data_nd)
  
  calib_data = mx.gluon.data.DataLoader(data_nd, batch_size=data_shape[0])
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
  data = mx.symbol.Variable('data', shape=data_shape, dtype='float32')
  weight = mx.symbol.Variable('weight', dtype='float32')
  bias = mx.symbol.Variable('bias', dtype='float32')
  sym = mx.symbol.FullyConnected(data=data, weight=weight, bias=bias, name='fc', num_hidden=64)
  data_nd = mx.random.uniform(data_min, data_max, shape=data_shape, ctx=mx.cpu())
  weight_nd = mx.random.uniform(weight_min, weight_max, shape=[64, 32], ctx=mx.cpu())
  bias_nd = mx.random.uniform(-1, +1, shape=[64], ctx=mx.cpu())
  arg_params = {
      'weight': weight_nd,
      'bias': bias_nd
  }

  ex = sym._bind(mx.cpu(), arg_params, args_grad=None)
  ex.forward(data = data_nd)
  ex.outputs[0].wait_to_read()
  sym_sg = sym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
  
  calib_data = mx.gluon.data.DataLoader(data_nd, batch_size=1)
  qsym, qarg_params, qaux_params = quantization.quantize_model(sym=sym_sg,
                                                               arg_params=arg_params,
                                                               aux_params={},
                                                               device=mx.cpu(),
                                                               excluded_sym_names=None,
                                                               excluded_op_names=None,
                                                               quantized_dtype='int8',
                                                               calib_mode='naive',
                                                               calib_data=calib_data,
                                                               num_calib_batches=1,
                                                               quantize_mode='full')
  qarg_params['data'] = data_nd
  qsym = qsym.optimize_for(QUANTIZE_SG_PASS_NAME, dedup_subgraph=True, skip_infer=True)
  qex = qsym._bind(mx.cpu(), qarg_params, args_grad=None)
  qex.forward()
  qex.outputs[0].wait_to_read()
  assert_almost_equal_with_err(ex.outputs[0].asnumpy(), qex.outputs[0].asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)

@mx.util.use_np
@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_bn_relu_fusion(axis):
    dummy_data = mx.np.random.uniform(-1.0, 1.0, size=(32, 3, 224, 224))

    net = mx.gluon.nn.HybridSequential()
    net.add(mx.gluon.nn.BatchNorm(axis=axis))
    net.add(mx.gluon.nn.Activation('relu'))
    net.initialize()

    out1 = net(dummy_data)
    out1.wait_to_read()
    net.optimize_for(dummy_data, backend='ONEDNN')
    out2 = net(dummy_data)

    assert_almost_equal(out1, out2)
