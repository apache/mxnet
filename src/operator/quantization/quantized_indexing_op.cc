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
 * \file quantized_indexing_op.cc
*/
#include <mxnet/op_attr_types.h>
#include "../tensor/indexing_op.h"

namespace mxnet {
namespace op {


inline bool QuantizedEmbeddingOpShape(const nnvm::NodeAttrs& attrs,
                                      mxnet::ShapeVector *in_attrs,
                                      mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const mxnet::TShape &dshape = (*in_attrs)[quantized_embedding::kData];
  if (!ndim_is_known(dshape)) return false;
  const EmbeddingParam& param = nnvm::get<EmbeddingParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*in_attrs, quantized_embedding::kWeight, Shape2(param.input_dim,
                                                                     param.output_dim));
  SHAPE_ASSIGN_CHECK(*in_attrs, quantized_embedding::kWeightMin, mxnet::TShape(1, 1));
  SHAPE_ASSIGN_CHECK(*in_attrs, quantized_embedding::kWeightMax, mxnet::TShape(1, 1));
  out_attrs->clear();

  mxnet::TShape oshape(dshape.ndim()+1, -1);
  for (int i = 0; i < dshape.ndim(); ++i) {
    oshape[i] = dshape[i];
  }
  oshape[dshape.ndim()] = param.output_dim;
  out_attrs->push_back(oshape);
  out_attrs->push_back(mxnet::TShape(1, 1));
  out_attrs->push_back(mxnet::TShape(1, 1));
  return shape_is_known(oshape);
}

inline bool QuantizedEmbeddingOpType(const nnvm::NodeAttrs& attrs,
                                     std::vector<int> *in_type,
                                     std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 4U);
  CHECK_GE(out_type->size(), 3U);
  int itype = (*in_type)[0];
  CHECK_NE(itype, -1) << "First input must have specified type";
  TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_type, 3, mshadow::kFloat32);
  out_type->clear();
  out_type->push_back(mshadow::kInt8);
  int dtype_out_min = 0;
  int dtype_out_max = 0;
  out_type->push_back(dtype_out_min);
  out_type->push_back(dtype_out_max);
  return true;
}

// storage type inference function for Embedding
inline bool QuantizedEmbeddingOpForwardStorageType(const nnvm::NodeAttrs& attrs,
                                                   const int dev_mask,
                                                   DispatchMode* dispatch_mode,
                                                   std::vector<int>* in_attrs,
                                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 4U);
  CHECK_EQ(out_attrs->size(), 3U);
  const int& data_stype = in_attrs->at(quantized_embedding::kData);
  const int& weight_stype = in_attrs->at(quantized_embedding::kWeight);
  const int& weight_min_stype = in_attrs->at(quantized_embedding::kWeightMin);
  const int& weight_max_stype = in_attrs->at(quantized_embedding::kWeightMax);
  int& out_stype = out_attrs->at(quantized_embedding::kOut);
  int& out_stype_min = out_attrs->at(quantized_embedding::kOutMin);
  int& out_stype_max = out_attrs->at(quantized_embedding::kOutMax);
  bool dispatched = false;
  CHECK_EQ(weight_min_stype, kDefaultStorage);
  CHECK_EQ(weight_max_stype, kDefaultStorage);
  if (!dispatched && data_stype == kDefaultStorage && weight_stype == kDefaultStorage) {
    // dns, dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
    dispatched = storage_type_assign(&out_stype_min, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
    dispatched = storage_type_assign(&out_stype_max, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && data_stype == kDefaultStorage && weight_stype == kRowSparseStorage) {
    // dns, rsp -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  return dispatched;
}

void QuantizedEmbeddingOpForward(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  CHECK_EQ(req[quantized_embedding::kOut], kWriteTo);
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 3U);
  CHECK_EQ(inputs[quantized_embedding::kWeight].ndim(), 2U)
          << "Embedding layer expects its weight to be two-dimensional. "
          << inputs[quantized_embedding::kWeight].ndim()
          << " dimensional input is given instead";
  mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
  EmbeddingOpForwardDnsImpl<cpu>(s, inputs[quantized_embedding::kData],
                                 inputs[quantized_embedding::kWeight],
                                 req[quantized_embedding::kOut],
                                 outputs[quantized_embedding::kOut]);
  float min_weight = inputs[quantized_embedding::kWeightMin].dptr<float>()[0];
  float max_weight = inputs[quantized_embedding::kWeightMax].dptr<float>()[0];
  outputs[quantized_embedding::kOutMin].dptr<float>()[0] = min_weight;
  outputs[quantized_embedding::kOutMax].dptr<float>()[0] = max_weight;
}

NNVM_REGISTER_OP(_contrib_quantized_embedding)
.describe(R"code(Maps integer indices to int8 vector representations (embeddings).
)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(3)
.set_attr_parser(ParamParser<EmbeddingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight", "min_weight", "max_weight"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedEmbeddingOpShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedEmbeddingOpType)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedEmbeddingOpForwardStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", QuantizedEmbeddingOpForward)
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input array to the embedding operator.")
.add_argument("weight", "NDArray-or-Symbol", "The embedding weight matrix.")
.add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of data.")
.add_arguments(EmbeddingParam::__FIELDS__());

NNVM_REGISTER_OP(Embedding)
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kSupport;
})
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    EmbeddingParam param;
    param.Init(attrs.dict);
    nnvm::ObjectPtr node = nnvm::Node::Create();
    if (param.dtype == mshadow::kFloat32) {
      node->attrs.op = Op::Get("_contrib_quantized_embedding");
      node->attrs.name = "quantized_" + attrs.name;
    } else {
      node->attrs.op = Op::Get("Embedding");
      node->attrs.name = attrs.name;
    }
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  })
.set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", [](
  const NodeAttrs &attrs, const size_t index, const std::string quantize_granularity) {
  return (index == 0);
});
}  // namespace op
}  // namespace mxnet

