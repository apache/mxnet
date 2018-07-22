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
 * Copyright (c) 2018 by Contributors
 * \file trt.cc
 * \brief TensorRT operation registration
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include "./tensorrt-inl.h"

#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/serialization.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(TRTParam);

OpStatePtr GetPtrMapping(nvinfer1::ICudaEngine* trt_engine,
                         tensorrt::NameToIdx_t input_map,
                         tensorrt::NameToIdx_t output_map) {
  TRTEngineParam param;
  for (int b = 0; b < trt_engine->getNbBindings(); ++b) {
    const std::string& binding_name = trt_engine->getBindingName(b);
    if (trt_engine->bindingIsInput(b)) {
      param.binding_map.emplace_back(input_map[binding_name],
                                     tensorrt::TypeIO::Inputs);
    } else {
      param.binding_map.emplace_back(output_map[binding_name],
                                     tensorrt::TypeIO::Outputs);
    }
  }
  param.trt_executor = trt_engine->createExecutionContext();
  return OpStatePtr::Create<TRTEngineParam>(param);
}

OpStatePtr TRTCreateState(const nnvm::NodeAttrs& attrs, Context /*ctx*/,
                          const std::vector<TShape>& /*ishape*/,
                          const std::vector<int>& /*itype*/) {
  const auto& node_param = nnvm::get<TRTParam>(attrs.parsed);

  ::onnx::ModelProto model_proto;
  bool success = model_proto.ParseFromString(node_param.serialized_onnx_graph);
  if (!success) {
    LOG(FATAL) << "Problems parsing serialized ONNX model.";
  }
  auto graph = model_proto.graph();
  auto first_input_type = graph.input(0).type().tensor_type();
  auto dim_value = first_input_type.shape().dim(0).dim_value();
  auto batch_size = static_cast<int32_t >(dim_value);
  // Need to set up max workspace size based on device properties
  nvinfer1::ICudaEngine* const trt_engine = ::onnx_to_tensorrt::onnxToTrtCtx(
      node_param.serialized_onnx_graph, batch_size, 1 << 30);

  tensorrt::NameToIdx_t output_map;
  for (auto& el : node_param.output_map) {
    output_map[el.first] = std::get<0>(el.second);
  }
  return GetPtrMapping(trt_engine, node_param.input_map, output_map);
}

void TRTParamParser(nnvm::NodeAttrs* attrs) {
  TRTParam param_;

  try {
    param_.Init(attrs->dict);
    common::Deserialize(&param_.input_map, param_.serialized_input_map);
    common::Deserialize(&param_.output_map, param_.serialized_output_map);
    param_.onnx_pb_graph.ParseFromString(param_.serialized_onnx_graph);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }

  attrs->parsed = std::move(param_);
}

inline bool TRTInferShape(const NodeAttrs& attrs, std::vector<TShape>* /*in_shape*/,
                          std::vector<TShape>* out_shape) {
  const auto &node_param = nnvm::get<TRTParam>(attrs.parsed);
  for (auto& el : node_param.output_map) {
    (*out_shape)[std::get<0>(el.second)] = std::get<1>(el.second);
  }
  return true;
}

inline bool TRTInferStorageType(const NodeAttrs& /*attrs*/, const int /*dev_mask*/,
                                DispatchMode* dispatch_mode,
                                std::vector<int>* /*in_storage_type*/,
                                std::vector<int>* out_storage_type) {
  return storage_type_assign(out_storage_type, mxnet::kDefaultStorage,
                             dispatch_mode, DispatchMode::kFCompute);
}

inline bool TRTInferType(const NodeAttrs& attrs, std::vector<int>* /*in_dtype*/,
                         std::vector<int>* out_dtype) {
  const auto& node_param = nnvm::get<TRTParam>(attrs.parsed);
  for (auto& el : node_param.output_map) {
    (*out_dtype)[std::get<0>(el.second)] = std::get<3>(el.second);
  }
  return true;
}

inline std::vector<std::string> TRTListInputNames(const NodeAttrs& attrs) {
  std::vector<std::string> output;
  const auto& node_param = nnvm::get<TRTParam>(attrs.parsed);
  output.resize(node_param.input_map.size());
  for (auto& el : node_param.input_map) {
    output[el.second] = el.first;
  }
  return output;
}

inline std::vector<std::string> TRTListOutputNames(const NodeAttrs& attrs) {
  std::vector<std::string> output;
  const auto& node_param = nnvm::get<TRTParam>(attrs.parsed);
  output.resize(node_param.output_map.size());
  for (auto& el : node_param.output_map) {
    output[std::get<0>(el.second)] = el.first;
  }
  return output;
}

NNVM_REGISTER_OP(_trt_op)
    .describe(R"code(TRT operation (one engine)
)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const auto& node_param = nnvm::get<TRTParam>(attrs.parsed);
      return node_param.input_map.size();
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      const auto& node_param = nnvm::get<TRTParam>(attrs.parsed);
      return node_param.output_map.size();
    })
    .set_attr_parser(TRTParamParser)
    .set_attr<nnvm::FInferShape>("FInferShape", TRTInferShape)
    .set_attr<nnvm::FInferType>("FInferType", TRTInferType)
    .set_attr<nnvm::FListInputNames>("FListInputNames", TRTListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames", TRTListOutputNames)
    .set_attr<FCreateOpState>("FCreateOpState", TRTCreateState)
    .set_attr<FInferStorageType>("FInferStorageType", TRTInferStorageType);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
