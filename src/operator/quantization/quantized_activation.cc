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
 * \file quantized_activation.cc
*/
#include <mxnet/op_attr_types.h>
#include "../nn/activation-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

bool QuantizedActivationShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_shape,
                              std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 3U);
  if (shape_is_none(in_shape->at(0))) return false;
  SHAPE_ASSIGN_CHECK(*in_shape, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*in_shape, 2, TShape{1});
  out_shape->clear();
  out_shape->push_back((*in_shape)[0]);
  out_shape->push_back(TShape{1});
  out_shape->push_back(TShape{1});
  return true;
}

bool QuantizedActivationType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_type,
                             std::vector<int> *out_type) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), 3U);
  CHECK_EQ(out_type->size(), 3U);
  if (param.act_type == activation::kReLU) {
    TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt8);
  } else {
    LOG(FATAL) << "_contrib_quantized_act only supports act_type=relu for now";
  }
  TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

inline static bool QuantizedActivationStorageType(const nnvm::NodeAttrs &attrs,
                                                  const int dev_mask,
                                                  DispatchMode *dispatch_mode,
                                                  std::vector<int> *in_attrs,
                                                  std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);

  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_MKLDNN == 1
  const ActivationParam &param = nnvm::get<ActivationParam>(attrs.parsed);
  if (dev_mask == mshadow::cpu::kDevMask && param.act_type == activation::kReLU) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#else
  CHECK_EQ(out_attrs->size(), 3);
#endif
  for (int& out_attr : *out_attrs)
    out_attr = kDefaultStorage;
  return true;
}

NNVM_REGISTER_OP(_contrib_quantized_act)
.describe(R"code(Activation operator for input and output data type of int8.
The input and output data comes with min and max thresholds for quantizing
the float32 data into int8.

.. Note::
     This operator only supports forward propogation. DO NOT use it in training.
     This operator only supports `relu`)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "min_data", "max_data"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInferType>("FInferType", QuantizedActivationType)
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedActivationShape)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedActivationStorageType)
.set_attr<FNeedRequantize>("FNeedRequantize",
  [](const NodeAttrs& attrs) {
    const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
    CHECK(param.act_type == activation::kReLU)
      << "_contrib_quantized_act only supports act_type=relu for now";
    return false;
  })
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_arguments(ActivationParam::__FIELDS__());


NNVM_REGISTER_OP(Activation)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  nnvm::NodePtr node = nnvm::Node::Create();
  if (param.act_type == activation::kReLU) {
    node->attrs.op = Op::Get("_contrib_quantized_act");
    node->attrs.name = "quantized_" + attrs.name;
  } else {
    LOG(INFO) << "Currently, quantized activation only supports relu, exclude "
              << attrs.name << " which act_type is " << param.act_type;
    node->attrs.op = nullptr;
    node->attrs.name = attrs.name;
  }
  node->attrs.dict = attrs.dict;
  if (node->op() != nullptr && node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
});

}  // namespace op
}  // namespace mxnet
