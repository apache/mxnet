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
 * \file quantized_transpose.cc
 * \author: Rafal Litka, rafal.litka@intel.com
 */
#include <mxnet/op_attr_types.h>
#include "../tensor/matrix_op-inl.h"
#include "../numpy/np_matrix_op-inl.h"

namespace mxnet {
namespace op {

inline bool QuantizedTransposeType(const nnvm::NodeAttrs& attrs,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

typedef bool (*TransposeShapeFunAny)(const nnvm::NodeAttrs&,
                                     mxnet::ShapeVector*,
                                     mxnet::ShapeVector*);

template <TransposeShapeFunAny TransposeShapeFun>
inline bool QuantizedTransposeShape(const nnvm::NodeAttrs& attrs,
                                    mxnet::ShapeVector* in_attrs,
                                    mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  mxnet::ShapeVector qin_attrs(1);
  mxnet::ShapeVector qout_attrs(1);
  SHAPE_ASSIGN_CHECK(qin_attrs, 0, (*in_attrs)[0]);
  SHAPE_ASSIGN_CHECK(qout_attrs, 0, (*out_attrs)[0]);
  bool ret = TransposeShapeFun(attrs, &qin_attrs, &qout_attrs);
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, qin_attrs[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, qout_attrs[0]);
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*in_attrs, 2, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, mxnet::TShape{1});
  return ret;
}

#define MXNET_OPERATOR_REGISTER_QUANTIZED_TRANSPOSE(name)                                    \
  NNVM_REGISTER_OP(name)                                                                     \
      .set_num_inputs(3)                                                                     \
      .set_num_outputs(3)                                                                    \
      .set_attr<nnvm::FInferType>("FInferType", QuantizedTransposeType)                      \
      .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                             \
      .set_attr<nnvm::FListInputNames>(                                                      \
          "FListInputNames",                                                                 \
          [](const NodeAttrs& attrs) {                                                       \
            return std::vector<std::string>{"data", "min_data", "max_data"};                 \
          })                                                                                 \
      .set_attr<nnvm::FListOutputNames>(                                                     \
          "FListOutputNames",                                                                \
          [](const NodeAttrs& attrs) {                                                       \
            return std::vector<std::string>{"output", "min_output", "max_output"};           \
          })                                                                                 \
      .set_attr<FQuantizable>("FQuantizable",                                                \
                              [](const NodeAttrs& attrs) { return QuantizeType::kSupport; }) \
      .add_argument("data", "NDArray-or-Symbol", "Array to be transposed.")                  \
      .add_argument("min_data",                                                              \
                    "NDArray-or-Symbol",                                                     \
                    "The minimum scalar value "                                              \
                    "possibly produced for the data")                                        \
      .add_argument("max_data",                                                              \
                    "NDArray-or-Symbol",                                                     \
                    "The maximum scalar value "                                              \
                    "possibly produced for the data")

MXNET_OPERATOR_REGISTER_QUANTIZED_TRANSPOSE(_npx_quantized_transpose)
    .set_attr_parser(ParamParser<NumpyTransposeParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedTransposeShape<NumpyTransposeShape>)
    .add_arguments(NumpyTransposeParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_QUANTIZED_TRANSPOSE(_contrib_quantized_transpose)
    .add_alias("quantized_transpose")
    .set_attr_parser(ParamParser<TransposeParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", QuantizedTransposeShape<TransposeShape>)
    .add_arguments(TransposeParam::__FIELDS__());

NNVM_REGISTER_OP(transpose).set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op       = Op::Get("_contrib_quantized_transpose");
  node->attrs.name     = "quantized_" + attrs.name;
  node->attrs.dict     = attrs.dict;
  if (node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
});

NNVM_REGISTER_OP(_npi_transpose).set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op       = Op::Get("_npx_quantized_transpose");
  node->attrs.name     = "quantized_" + attrs.name;
  node->attrs.dict     = attrs.dict;
  if (node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
});

}  // namespace op
}  // namespace mxnet
