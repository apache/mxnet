/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file layer_norm.cc
 * \brief Implements Ba et. al, Layer Normalization (https://arxiv.org/abs/1607.06450).
*/

#include "layer_norm-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LayerNormParam);

static bool LayerNormShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_shape,
                           std::vector<TShape> *out_shape) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
  const TShape &dshape = in_shape->at(layernorm::kData);
  int axis = param.axis;
  if (axis < 0) {
    axis += static_cast<int>(dshape.ndim());
  }
  CHECK(axis >= 0 && axis < static_cast<int>(dshape.ndim()))
    << "Channel axis out of range: axis=" << param.axis;

  const int channelCount = dshape[axis];

  if (dshape.ndim() == 0) {
    return false;
  }

  in_shape->at(layernorm::kGamma) = TShape(Shape1(channelCount));
  in_shape->at(layernorm::kBeta) = TShape(Shape1(channelCount));

  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  TShape moments_shape(dshape.begin(), dshape.end());
  moments_shape[axis] = 1;
  out_shape->push_back(moments_shape);  // kMean
  out_shape->push_back(moments_shape);  // kInvstd
  return true;
}


NNVM_REGISTER_OP(LayerNorm)
.describe(R"code(Layer normalization.

Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis and then 
compute the normalized output, which has the same shape as input, as following:

.. math::

  out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta

Both ``gamma`` and ``beta`` are learnable parameters.

Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
``data_std``. Note that no gradient will be passed through these two outputs.

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is -1, which sets the channel
axis to be the last item in the input shape.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<LayerNormParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "std"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  return param.output_mean_var ? 3 : 1;
})
.set_attr<nnvm::FInferShape>("FInferShape", LayerNormShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 3>)
.set_attr<FCompute>("FCompute<cpu>", LayerNormCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", [](const nnvm::NodePtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> heads;
  heads.push_back(ograds[0]);  // ograd
  heads.push_back(n->inputs[0]);  // data
  heads.push_back(n->inputs[1]);  // gamma
  heads.emplace_back(nnvm::NodeEntry{n, 1, 0});  // mean
  heads.emplace_back(nnvm::NodeEntry{ n, 2, 0 });  // std
  return MakeGradNode("_backward_LayerNorm", n, heads, n->attrs.dict);
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input data to layer normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_arguments(LayerNormParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_LayerNorm)
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<LayerNormParam>)
.set_attr<FCompute>("FCompute<cpu>", LayerNormGradCompute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

}  // namespace op
}  // namespace mxnet
