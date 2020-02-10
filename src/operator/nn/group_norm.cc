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
 * Copyright (c) 2019 by Contributors
 * \file group_norm.cc
 * \brief Implements Group Normalization (https://arxiv.org/abs/1803.08494).
*/

#include "group_norm-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(GroupNormParam);

static bool GroupNormShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_shape,
                           mxnet::ShapeVector *out_shape) {
  const GroupNormParam& param = nnvm::get<GroupNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
  const mxnet::TShape &dshape = in_shape->at(groupnorm::kData);
  CHECK_GE(dshape.ndim(), 3U);
  const int num_groups = param.num_groups;
  CHECK_EQ(dshape[1] % num_groups, 0) << "# of channels must be divisible by # of groups";

  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  in_shape->at(groupnorm::kGamma) = mxnet::TShape(Shape1(num_groups));
  in_shape->at(groupnorm::kBeta) = mxnet::TShape(Shape1(num_groups));

  out_shape->clear();
  out_shape->push_back(dshape);

  mxnet::TShape moments_shape(2, 1);
  moments_shape[0] = dshape[0];
  moments_shape[1] = num_groups;
  out_shape->push_back(moments_shape);
  out_shape->push_back(moments_shape);
  return true;
}

NNVM_REGISTER_OP(GroupNorm)
.describe(R"code(Group normalization.

The input channels are separated into ``num_groups`` groups, each containing ``num_channels / num_groups`` channels.
The mean and standard-deviation are calculated separately over the each group.

.. math::

  data = data.reshape((N, num_groups, C // num_groups, ...))
  out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta

Both ``gamma`` and ``beta`` are learnable parameters.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<GroupNormParam>)
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
  const GroupNormParam& param = nnvm::get<GroupNormParam>(attrs.parsed);
  return param.output_mean_var ? 3 : 1;
})
.set_attr<mxnet::FInferShape>("FInferShape", GroupNormShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 3>)
.set_attr<FCompute>("FCompute<cpu>", GroupNormCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", [](const nnvm::ObjectPtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> heads;
  heads.push_back(ograds[0]);  // ograd
  heads.push_back(n->inputs[0]);  // data
  heads.push_back(n->inputs[1]);  // gamma
  heads.emplace_back(nnvm::NodeEntry{n, 1, 0});  // mean
  heads.emplace_back(nnvm::NodeEntry{ n, 2, 0 });  // std
  return MakeGradNode("_backward_GroupNorm", n, heads, n->attrs.dict);
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.add_argument("data", "NDArray-or-Symbol", "Input data")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_arguments(GroupNormParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_GroupNorm)
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<GroupNormParam>)
.set_attr<FCompute>("FCompute<cpu>", GroupNormGradCompute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

}  // namespace op
}  // namespace mxnet
