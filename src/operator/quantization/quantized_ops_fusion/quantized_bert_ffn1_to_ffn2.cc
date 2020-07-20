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
 * Copyright (c) 2020 by Contributors
 * \file quantized_bert_ffn1_to_ffn2.cc
 * \brief
*/
#include <vector>
#include "../quantization_utils.h"
#include "../../nn/fully_connected-inl.h"
#include "./quantized_bert_ffn1_to_ffn2-inl.h"

namespace mxnet {
namespace op {


bool QuantizedBERTFFN1TOFFN2Shape(const nnvm::NodeAttrs& attrs,
                                  mxnet::ShapeVector *in_shape,
                                  mxnet::ShapeVector *out_shape) {
  const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
  using namespace mshadow;
  uint32_t num_inputs = 3;
  CHECK_EQ(in_shape->size(), num_inputs * 3);
  CHECK_EQ(out_shape->size(), 3U);

  mxnet::TShape dshape = (*in_shape)[0];
  // require data ndim to be known
  if (!mxnet::ndim_is_known(dshape)) return false;

  index_t num_input;
  if (!param.flatten) {
    num_input = dshape[dshape.ndim() - 1];
  } else {
    num_input = dshape.ProdShape(1, dshape.ndim());
  }

  mxnet::TShape wshape = Shape2(param.num_hidden, num_input);
  SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
    
  mxnet::TShape bshape = Shape1(param.num_hidden);
  SHAPE_ASSIGN_CHECK(*in_shape, 2, bshape);

  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape(1, 1));
  }

  if (!param.flatten) {
    mxnet::TShape result_shape(dshape);
    result_shape[dshape.ndim() - 1] = param.num_hidden;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param.num_hidden));
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape(1, 1));

  if ((*out_shape)[0].ndim() > 0) {
    dshape[0] = ((*out_shape)[0])[0];
    SHAPE_ASSIGN_CHECK(*in_shape, 0, dshape);
  }
  return true;
}

bool QuantizedBERTFFN1TOFFN2Type(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_type,
                                 std::vector<int> *out_type) {
  const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
  uint32_t num_inputs = 3;
  CHECK_EQ(in_type->size(), num_inputs * 3);
  CHECK_EQ(out_type->size(), 3U);

  TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
  for (size_t i = 1; i < num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
  }
  for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);
  return true;
}

bool QuantizedBERTFFN1TOFFN2StorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
  uint32_t num_inputs = 3;
  CHECK_EQ(in_attrs->size(), num_inputs * 3);
  CHECK_EQ(out_attrs->size(), 3U);

  *dispatch_mode = DispatchMode::kFCompute;

  for (auto &v : *out_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }

  for (auto &v : *in_attrs) {
    v = kDefaultStorage;
    if (common::stype_string(v).compare("unknown") == 0) {
      return false;
    }
  }
  return true;
}

void QuantizedBERTFFN1TOFFN2ForwardCPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &in_data,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &out_data) {
    
    LOG(FATAL) << "Quantized Bert FFN1 to FFN2 fusion operator currently only supports GPU";
}

DMLC_REGISTER_PARAMETER(QuantizedBERTFFN1TOFFN2Param);

NNVM_REGISTER_OP(_contrib_quantized_bert_ffn1_to_ffn2_fusion)
.describe(R"code(Quantized bert_ffn1_to_ffn2_fusion operator for input, weight and bias data type of int8,
and calculates the fullyconnected outputs, apply GELU operator and then quantize the output back to int8. 
For each argument, two more arguments of type float32 must be provided representing the thresholds of 
quantizing argument from data type float32 to int8. The final outputs contain the result in int8, and min
and max thresholds representing the threholds for quantizing the float output into int8.

.. Note::
    This operator only supports forward propogation. DO NOT use it in training.)code" ADD_FILELINE)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
    return 9;
  })
.set_num_outputs(3)
.set_attr_parser(ParamParser<QuantizedBERTFFN1TOFFN2Param>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const QuantizedBERTFFN1TOFFN2Param& param = nnvm::get<QuantizedBERTFFN1TOFFN2Param>(attrs.parsed);
      return std::vector<std::string>{"data", "weight", "bias", "min_data", "max_data",
                                      "min_weight", "max_weight", "min_bias", "max_bias"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedBERTFFN1TOFFN2Shape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedBERTFFN1TOFFN2Type)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedBERTFFN1TOFFN2StorageType)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FCompute>("FCompute<cpu>", QuantizedBERTFFN1TOFFN2ForwardCPU)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "weight.")
.add_argument("bias", "NDArray-or-Symbol", "bias.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of weight.")
.add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of weight.")
.add_argument("min_bias", "NDArray-or-Symbol", "Minimum value of bias.")
.add_argument("max_bias", "NDArray-or-Symbol", "Maximum value of bias.")
.add_arguments(QuantizedBERTFFN1TOFFN2Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
