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
 * \file quantized_sum.cc
 * \brief
*/
#include "../tensor/elemwise_unary_op.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_quantized_sum-inl.h"
#endif

namespace mxnet {
namespace op {

static bool SumShape(const nnvm::NodeAttrs& attrs,  mxnet::ShapeVector* in_shape,
                         mxnet::ShapeVector* out_shape) {
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_shape->size(), 6U);
  // C, C_min, C_max
  CHECK_EQ(out_shape->size(), 3U);
  CHECK_EQ((*in_shape)[0], (*in_shape)[1]);


  SHAPE_ASSIGN_CHECK(*in_shape, 2, TShape{1});
  SHAPE_ASSIGN_CHECK(*in_shape, 3, TShape{1});
  SHAPE_ASSIGN_CHECK(*in_shape, 4, TShape{1});
  SHAPE_ASSIGN_CHECK(*in_shape, 5, TShape{1});

  SHAPE_ASSIGN_CHECK(*out_shape, 0, (*in_shape)[0]);
  SHAPE_ASSIGN_CHECK(*out_shape, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*out_shape, 2, TShape{1});
  return true;
}

static bool SumType(const nnvm::NodeAttrs& attrs, std::vector<int>* in_type,
                       std::vector<int>* out_type) {
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_type->size(), 6U);
  // C, C_min, C_max
  CHECK_EQ(out_type->size(), 3U);

  // A, B
  const int elem_add_num = 2;
  for (int i = 0; i < elem_add_num; ++i) {
    if (in_type->at(i) == mshadow::kInt8) {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
    } else {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kUint8);
    }
  }
  // C
  int dtype = mshadow::kInt32;
#if MXNET_USE_MKLDNN == 1
  const RequantizeSumParam& params = nnvm::get<RequantizeSumParam>(attrs.parsed);
  if (params.max_calib_range.has_value() && params.min_calib_range.has_value()) {
    dtype = (in_type->at(0) == in_type->at(1)) ? in_type->at(0) : mshadow::kInt8;
  }
#endif
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  // C_min
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  // C_max
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);

  return true;
}

NNVM_REGISTER_OP(_contrib_quantized_sum)
.describe(R"code(elem_add operator for input dataA and input dataB data type of int8,
and accumulates in type int32 for the output. For each argument, two more arguments of type
float32 must be provided representing the thresholds of quantizing argument from data
type float32 to int8. The final outputs contain result in int32, and min
and max thresholds representing the threholds for quantizing the float32 output into int32.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.

)code")
.set_num_inputs([](const NodeAttrs& attrs) {
// A, B, A_min, A_max, B_min, B_max
  return 6;
})
// C, C_min, C_max
.set_num_outputs(3)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"lhs", "rhs", "lhs_min", "lhs_max", "rhs_min", "rhs_max"}; \
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "min_output", "max_output"};
})
.set_attr<nnvm::FInferType>("FInferType", SumType)
.set_attr<mxnet::FInferShape>("FInferShape", SumShape)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.add_argument("lhs", "NDArray-or-Symbol", "first input")
.add_argument("rhs", "NDArray-or-Symbol", "4th input")
.add_argument("lhs_min", "NDArray-or-Symbol", "second input")
.add_argument("lhs_max", "NDArray-or-Symbol", "third input")
.add_argument("rhs_min", "NDArray-or-Symbol", "5th input")
.add_argument("rhs_max", "NDArray-or-Symbol", "6th input");


NNVM_REGISTER_OP(elemwise_add)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_contrib_quantized_sum");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  if (node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
});

}  // namespace op
}  // namespace mxnet
