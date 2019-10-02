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
 * \file attrs.cc
 * \author Junru Shao
 */
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include <nnvm/node.h>

#include "../../../../operator/nn/activation-inl.h"
#include "../../../../operator/nn/batch_norm-inl.h"
#include "../../../../operator/nn/convolution-inl.h"
#undef Assign

#include "../../../include/bridge/legacy_nnvm.h"
#include "../../../include/op/attrs/nn.h"

namespace mxnet {
namespace v3 {
namespace bridge {
namespace legacy_nnvm {

using ir::Array;
using ir::Attrs;
using ir::Call;
using ir::CallNode;
using ir::Integer;
using ir::Op;

static Array<Integer> AsArray(const mxnet::TShape &from) {
  Array<Integer> result;
  for (const auto &item : from) {
    result.push_back(Integer(item));
  }
  return result;
}

static Attrs ConvertAttrs(const mxnet::op::ConvolutionParam &attrs,
                          const nnvm::NodeAttrs node_attrs) {
  static std::unordered_map<int, std::string> layout_map = {
      {mshadow::kNCW, "NCW"},      // 1-d conv
      {mshadow::kNCHW, "NCHW"},    // 2-d conv
      {mshadow::kNHWC, "NHWC"},    // 2-d conv
      {mshadow::kNCDHW, "NCDHW"},  // 3-d conv
      {mshadow::kNDHWC, "NDHWC"},  // 3-d conv
  };
  auto relay_attrs = ir::make_node<v3::op::attrs::ConvAttrs>();
  relay_attrs->stride = AsArray(attrs.stride);
  relay_attrs->dilation = AsArray(attrs.dilate);
  relay_attrs->padding = AsArray(attrs.pad);
  relay_attrs->groups = attrs.num_group;
  relay_attrs->layout = layout_map[attrs.layout.value()];
  relay_attrs->capsule = NNVMCapsule::make(node_attrs);
  return ir::Attrs(relay_attrs);
}

static Attrs ConvertAttrs(const mxnet::op::BatchNormParam &attrs,
                          const nnvm::NodeAttrs &node_attrs) {
  auto relay_attrs = ir::make_node<v3::op::attrs::BatchNormAttrs>();
  relay_attrs->eps = attrs.eps;
  relay_attrs->momentum = attrs.momentum;
  relay_attrs->affine = !attrs.fix_gamma;
  relay_attrs->capsule = NNVMCapsule::make(node_attrs);
  return ir::Attrs(relay_attrs);
}

Call ConvertCall(const nnvm::Op *op, const nnvm::NodeAttrs &attrs,
                 const ir::Array<ir::Expr> &args) {
  CHECK(op != nullptr) << "InternalError: operator undefined.";
  if (op->name == "Convolution") {
    static const Op &op = Op::Get("nn.conv2d");
    const auto &nnvm_attrs =
        nnvm::get<mxnet::op::ConvolutionParam>(attrs.parsed);
    return CallNode::make(op, args, ConvertAttrs(nnvm_attrs, attrs));
  } else if (op->name == "BatchNorm") {
    static const Op &op = Op::Get("nn.batch_norm");
    const auto &nnvm_attrs = nnvm::get<mxnet::op::BatchNormParam>(attrs.parsed);
    return CallNode::make(op, args, ConvertAttrs(nnvm_attrs, attrs));
  } else if (op->name == "elemwise_add") {
    static const Op &op = Op::Get("add");
    return CallNode::make(op, args, {});
  } else if (op->name == "Activation") {
    static std::unordered_map<int, Op> op_map = {
        {mxnet::op::activation::kReLU, Op::Get("nn.relu")},
        {mxnet::op::activation::kSigmoid, Op::Get("sigmoid")},
        {mxnet::op::activation::kTanh, Op::Get("tanh")},
    };
    const auto &nnvm_attrs =
        nnvm::get<mxnet::op::ActivationParam>(attrs.parsed);
    if (op_map.count(nnvm_attrs.act_type)) {
      return CallNode::make(op_map[nnvm_attrs.act_type], args, {});
    }
  }
  LOG(INFO) << "Warning: cannot recognize NNVM operator " << op->name
            << ", fallback to add";
  return CallNode::make(Op::Get("add"), args, {}, {});
}

}  // namespace legacy_nnvm
}  // namespace bridge
}  // namespace v3
}  // namespace mxnet
#endif
