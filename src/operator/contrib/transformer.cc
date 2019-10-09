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
 *  Copyright (c) 2018 by Contributors
 * \file transformer.cc
 * \brief CPU implementation of the operators used in Transformer
 */
#include <mxnet/base.h>
#include "./transformer-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(InterleavedMatMulParam);

static bool InterleavedMatMulSelfAttQKShape(const NodeAttrs& attrs,
                                            mxnet::ShapeVector* in_shape,
                                            mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1);
  auto qkv_shape = in_shape->at(0);
  CHECK_EQ(qkv_shape.ndim(), 3);
  out_shape->resize(1);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({params.heads * qkv_shape[1], qkv_shape[0], qkv_shape[0]}));
  return true;
}

static bool InterleavedMatMulSelfAttValAttShape(const NodeAttrs& attrs,
                                                mxnet::ShapeVector* in_shape,
                                                mxnet::ShapeVector* out_shape) {
  CHECK_EQ(in_shape->size(), 2);
  auto qkv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);
  CHECK_EQ(qkv_shape.ndim(), 3);
  CHECK_EQ(att_shape.ndim(), 3);
  CHECK_EQ(qkv_shape[0], att_shape[1]);
  CHECK_EQ(qkv_shape[0], att_shape[2]);
  CHECK_EQ(qkv_shape[2] % 3, 0);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
    mxnet::TShape({qkv_shape[0], qkv_shape[1], qkv_shape[2] / 3}));
  return true;
}

static bool InterleavedMatMulEncDecQKShape(const NodeAttrs& attrs,
                                           mxnet::ShapeVector* in_shape,
                                           mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2);
  auto q_shape = in_shape->at(0);
  auto kv_shape = in_shape->at(1);
  CHECK_EQ(q_shape.ndim(), 3);
  CHECK_EQ(kv_shape.ndim(), 3);
  CHECK_EQ(q_shape[2] * 2, kv_shape[2]);
  CHECK_EQ(q_shape[1], kv_shape[1]);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({q_shape[1] * params.heads, q_shape[0], kv_shape[0]}));
  return true;
}

static bool InterleavedMatMulEncDecValAttShape(const NodeAttrs& attrs,
                                               mxnet::ShapeVector* in_shape,
                                               mxnet::ShapeVector* out_shape) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2);
  auto kv_shape = in_shape->at(0);
  auto att_shape = in_shape->at(1);
  CHECK_EQ(kv_shape[0], att_shape[2]);
  CHECK_EQ(kv_shape[1] * params.heads, att_shape[0]);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,
      mxnet::TShape({att_shape[1], kv_shape[1], kv_shape[2] / 2}));
  return true;
}

NNVM_REGISTER_OP(interleaved_matmul_selfatt_qk)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulSelfAttQKShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseIn{"_backward_interleaved_matmul_selfatt_qk"})
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Interleaved queries, keys and values")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_qk)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);

NNVM_REGISTER_OP(interleaved_matmul_selfatt_valatt)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries_keys_values", "attention"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulSelfAttValAttShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  ElemwiseGradUseIn{"_backward_interleaved_matmul_selfatt_valatt"})
.add_argument("queries_keys_values", "NDArray-or-Symbol", "Queries, keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_valatt)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);

NNVM_REGISTER_OP(interleaved_matmul_encdec_qk)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"queries", "keys_values"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulEncDecQKShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient",
    ElemwiseGradUseIn{"_backward_interleaved_matmul_encdec_qk"})
.add_argument("queries", "NDArray-or-Symbol", "Queries")
.add_argument("keys_values", "NDArray-or-Symbol", "Keys and values interleaved")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_encdec_qk)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);

NNVM_REGISTER_OP(interleaved_matmul_encdec_valatt)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InterleavedMatMulParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"keys_values", "attention"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", InterleavedMatMulEncDecValAttShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient",
    ElemwiseGradUseIn{"_backward_interleaved_matmul_encdec_valatt"})
.add_argument("keys_values", "NDArray-or-Symbol", "Keys and values interleaved")
.add_argument("attention", "NDArray-or-Symbol", "Attention maps")
.add_arguments(InterleavedMatMulParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_interleaved_matmul_encdec_valatt)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<InterleavedMatMulParam>);


// relu
MXNET_OPERATOR_REGISTER_UNARY(_contrib_div_sqrt_dim)
.describe(R"code(Rescale the input by the square root of the channel dimension.

   out = data / sqrt(data.shape[-1])

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", DivSqrtDimForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_contrib_div_sqrt_dim"});

}  // namespace op
}  // namespace mxnet
