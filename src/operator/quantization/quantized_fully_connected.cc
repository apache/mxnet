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
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected.cc
 * \brief
 * \author Ziheng Jiang, Jun Wu
*/
#include "../nn/fully_connected-inl.h"

namespace mxnet {
namespace op {

bool QuantizedFullyConnectedShape(const nnvm::NodeAttrs& attrs,
                                  std::vector<TShape> *in_shape,
                                  std::vector<TShape> *out_shape) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  CHECK(param.flatten) << "QuantizedFullyConnectedOp only supports flatten=true for now";
  using namespace mshadow;
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_shape->size(), num_inputs * 3);
  CHECK_EQ(out_shape->size(), 3U);

  CHECK(!shape_is_none(in_shape->at(0)))
    << "QuantizedFullyConnectedOp input data shape must be given";
  const TShape& dshape = in_shape->at(0);
  TShape wshape = Shape2(param.num_hidden, dshape.ProdShape(1, dshape.ndim()));
  SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
  if (!param.no_bias) {
    TShape bshape = Shape1(param.num_hidden);
    SHAPE_ASSIGN_CHECK(*in_shape, 2, bshape);
  }

  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
  }

  SHAPE_ASSIGN_CHECK(*out_shape, 0, TShape({dshape[0], wshape[0]}));
  SHAPE_ASSIGN_CHECK(*out_shape, 1, TShape({1}));
  SHAPE_ASSIGN_CHECK(*out_shape, 2, TShape({1}));
  return true;
}

bool QuantizedFullyConnectedType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_type,
                                 std::vector<int> *out_type) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_type->size(), num_inputs * 3);
  CHECK_EQ(out_type->size(), 3U);

  for (size_t i = 0; i < num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
  }
  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt32);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.describe(R"code(Fully Connected operator for input, weight and bias data type of int8,
and accumulates in type int32 for the output. For each argument, two more arguments of type
float32 must be provided representing the thresholds of quantizing argument from data
type float32 to int8. The final outputs contain the convolution result in int32, and min
and max thresholds representing the threholds for quantizing the float32 output into int32.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
    return param.no_bias? 6 : 9;
  })
.set_num_outputs(3)
.set_attr_parser(ParamParser<FullyConnectedParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
    if (param.no_bias) {
      return std::vector<std::string>{"data", "weight", "min_data", "max_data",
                                      "min_weight", "max_weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias", "min_data", "max_data",
                                      "min_weight", "max_weight", "min_bias", "max_bias"};
    }
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuantizedFullyConnectedShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedFullyConnectedType)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "weight.")
.add_argument("bias", "NDArray-or-Symbol", "bias.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of weight.")
.add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of weight.")
.add_argument("min_bias", "NDArray-or-Symbol", "Minimum value of bias.")
.add_argument("max_bias", "NDArray-or-Symbol", "Maximum value of bias.")
.add_arguments(FullyConnectedParam::__FIELDS__());

NNVM_REGISTER_OP(FullyConnected)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("_contrib_quantized_fully_connected");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
