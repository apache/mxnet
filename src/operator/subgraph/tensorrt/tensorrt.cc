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
 * \file tensorrt.cc
 * \brief TensorRT operation registration
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include "./tensorrt-inl.h"

namespace mxnet {
namespace op {

inline uint32_t TRTNumInputs(const nnvm::NodeAttrs& attrs) {
  const TRTParam& param = nnvm::get<TRTParam>(attrs.parsed);
  const auto inputs_to_idx = param.inputs_to_idx;
  return inputs_to_idx.size();
}

inline std::vector<std::string> TRTListInputNames(const nnvm::NodeAttrs& attrs) {
  std::vector<std::string> outputs;
  const TRTParam& param = nnvm::get<TRTParam>(attrs.parsed);
  const auto inputs_to_idx = param.inputs_to_idx;
  for (auto& p : inputs_to_idx) {
    outputs[p.second] = p.first;
  }
  return outputs;
}

inline bool TRTInferShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_shapes,
                          std::vector<TShape> *out_shapes) {
  using namespace exec;
  const nnvm::Symbol subgraph_sym = *(attrs.subgraphs[0]);
  const TRTParam& param = nnvm::get<TRTParam>(attrs.parsed);
  auto params_map = param.params_map;
  auto inputs_to_idx = param.inputs_to_idx;
  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_shapes->size() + params_map.size());
  CHECK_EQ(idx_g.outputs().size(), out_shapes->size());

  // Put the input and output shapes to the shape vector.
  mxnet::ShapeVector shapes(idx_g.num_node_entries());
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_shapes->size() + params_map.size());
  for (size_t i = 0; i < input_nids.size(); i++) {
    auto node = idx_g[input_nids[i]].source;
    auto eid = idx_g.entry_id(input_nids[i], 0);
    auto it_params = params_map.find(node->attrs.name);
    auto it_inputs = inputs_to_idx.find(node->attrs.name);
    if (it_params != params_map.end()) {
      shapes[eid] = it_params->second.shape();
    } else if (it_inputs != inputs_to_idx.end()) {
      shapes[eid] = in_shapes->at(it_inputs->second);
    } else {
      LOG(FATAL) << node->attrs.name << " shape information is missing for attributes inference";
    }
  }
  CHECK_EQ(g.outputs.size(), out_shapes->size());
  for (size_t i = 0; i < out_shapes->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    shapes[eid] = out_shapes->at(i);
  }

  // Infer shape of the graph.
  g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
  g = exec::InferShape(std::move(g));
  // Copy the inferred shape back to the input shapes and the output shapes.
  shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  // assign to in_shapes
  for (size_t i = 0; i < input_nids.size(); ++i) {
    const auto node = idx_g[input_nids[i]].source;
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    auto it = inputs_to_idx.find(node->attrs.name);
    if (it != inputs_to_idx.end()) {
      SHAPE_ASSIGN_CHECK(*in_shapes, it->second, shapes[eid]);
    }
  }
  // assign to out_shapes
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    SHAPE_ASSIGN_CHECK(*out_shapes, i, shapes[eid]);
  }
  // Check if we have inferred the shapes correctly.
  return g.GetAttr<size_t>("shape_num_unknown_nodes") == 0;
}

inline bool TRTInferType(const nnvm::NodeAttrs& attrs,
                    std::vector<int> *in_types,
                    std::vector<int> *out_types) {
  const nnvm::Symbol subgraph_sym = *(attrs.subgraphs[0]);
  const TRTParam& param = nnvm::get<TRTParam>(attrs.parsed);
  auto params_map = param.params_map;
  auto inputs_to_idx = param.inputs_to_idx;

  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_types->size() + params_map.size());
  CHECK_EQ(idx_g.outputs().size(), out_types->size());

  // Put the input and output data types to the dtype vector.
  nnvm::DTypeVector types(idx_g.num_node_entries(), -1);
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_types->size() + params_map.size());
  for (size_t i = 0; i < input_nids.size(); i++) {
    auto node = idx_g[input_nids[i]].source;
    auto eid = idx_g.entry_id(input_nids[i], 0);
    auto it_params = params_map.find(node->attrs.name);
    auto it_inputs = inputs_to_idx.find(node->attrs.name);
    if (it_params != params_map.end()) {
      types[eid] = -1;
    } else if (it_inputs != inputs_to_idx.end()) {
      types[eid] = in_types->at(it_inputs->second);
    } else {
      LOG(FATAL) << node->attrs.name
                 << " dtype information is missing for attributes inference";
    }
  }
  CHECK_EQ(g.outputs.size(), out_types->size());
  for (size_t i = 0; i < out_types->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    types[eid] = out_types->at(i);
  }

  // Infer data type of the graph.
  g.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(types));
  g = exec::InferType(std::move(g));

  types = g.GetAttr<nnvm::DTypeVector>("dtype");
  // assign to in_types
  for (size_t i = 0; i < input_nids.size(); ++i) {
    const auto node = idx_g[input_nids[i]].source;
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    auto it = inputs_to_idx.find(node->attrs.name);
    if (it != inputs_to_idx.end()) {
      TYPE_ASSIGN_CHECK(*in_types, it->second, types[eid]);
    }
  }
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    TYPE_ASSIGN_CHECK(*out_types, i, types[eid]);
  }

  // Check if we have inferred the dtypes correctly.
  return g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0;
}

inline bool TRTInferStorageType(const nnvm::NodeAttrs& attrs,
                           const int dev_mask,
                           DispatchMode* dispatch_mode,
                           std::vector<int>* in_stypes,
                           std::vector<int>* out_stypes) {
  const nnvm::Symbol subgraph_sym = *(attrs.subgraphs[0]);
  const TRTParam& param = nnvm::get<TRTParam>(attrs.parsed);
  auto params_map = param.params_map;
  auto inputs_to_idx = param.inputs_to_idx;
  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_stypes->size() + params_map.size());
  CHECK_EQ(idx_g.outputs().size(), out_stypes->size());
  exec::DevMaskVector dev_masks(idx_g.num_node_entries(), dev_mask);

  // Put the input and output storages to the storage vector.
  StorageTypeVector stypes(idx_g.num_node_entries(), kUndefinedStorage);
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_stypes->size() + params_map.size());
  for (size_t i = 0; i < input_nids.size(); i++) {
    auto node = idx_g[input_nids[i]].source;
    auto eid = idx_g.entry_id(input_nids[i], 0);
    auto it_params = params_map.find(node->attrs.name);
    auto it_inputs = inputs_to_idx.find(node->attrs.name);
    if (it_params != params_map.end()) {
      stypes[eid] = it_params->second.storage_type();
    } else if (it_inputs != inputs_to_idx.end()) {
      stypes[eid] = in_stypes->at(it_inputs->second);
    } else {
      LOG(FATAL) << node->attrs.name
                 << " storage type information is missing for attributes inference";
    }
  }
  CHECK_EQ(g.outputs.size(), out_stypes->size());
  for (size_t i = 0; i < out_stypes->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    stypes[eid] = out_stypes->at(i);
  }

  // Infer storage type of the graph.
  bool dev_match = g.attrs.count("dev_mask") &&
                   g.GetAttr<exec::DevMaskVector>("dev_mask") == dev_masks;
  if (!dev_match) {
    g.attrs["dev_mask"] = std::make_shared<dmlc::any>(std::move(dev_masks));
  }
  g.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(stypes));
  g = exec::InferStorageType(std::move(g));

  stypes = g.GetAttr<StorageTypeVector>("storage_type");
  // assign to in_types
  for (size_t i = 0; i < input_nids.size(); ++i) {
    const auto node = idx_g[input_nids[i]].source;
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    auto it = inputs_to_idx.find(node->attrs.name);
    if (it != inputs_to_idx.end()) {
      STORAGE_TYPE_ASSIGN_CHECK(*in_stypes, it->second, stypes[eid]);
    }
  }

  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    STORAGE_TYPE_ASSIGN_CHECK(*out_stypes, i, stypes[eid]);
  }
  // Check if we have inferred the storages correctly.
  return g.GetAttr<size_t>("storage_type_num_unknown_nodes") == 0;
}

void TRTParamParser(nnvm::NodeAttrs* attrs) {
  TRTParam& _param = nnvm::get<TRTParam>(attrs->parsed);
  std::string prefix = "subgraph_param_";
  std::string str_dtype, str_shape, str_pointer, _tmp;
  for (auto it = attrs->dict.begin(); it != attrs->dict.end();) {
    std::string attrs_name = it->first;
    if (std::equal(prefix.begin(), prefix.end(), attrs_name.begin())) {
      std::string param_name = attrs_name.substr(prefix.size(),
                                                 attrs_name.size() - prefix.size());
      // TODO(cfujitsang): find a less dirty way to give weights
      NDArray *cache = reinterpret_cast<NDArray*>(stol(it->second));
      _param.params_map.emplace(param_name, cache->Copy(Context()));
      _param.params_map[param_name].WaitToRead();
      it = attrs->dict.erase(it);
    } else {
      ++it;
    }
  }
  attrs->parsed = std::move(_param);
}

OpStatePtr TRTCreateState(const nnvm::NodeAttrs& attrs, Context ctx,
                          const std::vector<TShape>& in_shape,
                          const std::vector<int>& in_type) {
  const auto& node_param = nnvm::get<TRTParam>(attrs.parsed);
  nnvm::Graph graph;
  graph.outputs = attrs.subgraphs[0]->outputs;
  uint32_t max_batch_size = dmlc::GetEnv("MXNET_TENSORRT_MAX_BATCH_SIZE", in_shape[0][0]);
  if (max_batch_size < in_shape[0][0]) {
    LOG(INFO) << "Warning: max batch size changed to be is: " << in_shape[0][0]
              << " instead of: " << max_batch_size;
    max_batch_size = in_shape[0][0];
  }
  std::unordered_map<std::string, NDArray> params_map = node_param.params_map;
  const auto& inputs_to_idx = node_param.inputs_to_idx;
  const auto& outputs_to_idx = node_param.outputs_to_idx;
  const auto& idx_g = graph.indexed_graph();
  const auto& input_nids = idx_g.input_nodes();
  mxnet::ShapeVector shape_inputs(input_nids.size());
  nnvm::DTypeVector dtype_inputs(input_nids.size());
  for (int i = 0; i < input_nids.size(); ++i) {
    auto node = idx_g[input_nids[i]].source;
    auto it_params = params_map.find(node->attrs.name);
    auto it_inputs = inputs_to_idx.find(node->attrs.name);
    if (it_params != params_map.end()) {
      shape_inputs[i] = it_params->second.shape();
      dtype_inputs[i] = it_params->second.dtype();
    } else if (it_inputs != inputs_to_idx.end()) {
      shape_inputs[i] = in_shape[it_inputs->second];
      dtype_inputs[i] = in_type[it_inputs->second];
    } else {
      LOG(FATAL) << node->attrs.name << " attribute is missing for attributes inference";
    }
  }
  mxnet::ShapeVector out_shape(graph.outputs.size());
  nnvm::DTypeVector out_type(graph.outputs.size(), -1);
  mxnet::ShapeVector _in_shape(in_shape.begin(), in_shape.end());
  nnvm::DTypeVector _in_type(in_type.begin(), in_type.end());
  TRTInferShape(attrs, &_in_shape, &out_shape);
  TRTInferType(attrs, &_in_type, &out_type);
  nnvm::DTypeVector dtypes(idx_g.num_node_entries());
  mxnet::ShapeVector shapes(idx_g.num_node_entries());
  for (int i = 0; i < graph.outputs.size(); ++i) {
    auto eid = idx_g.entry_id(graph.outputs[i]);
    dtypes[eid] = out_type[i];
    shapes[eid] = out_shape[i];
  }
  graph.attrs["dtype_inputs"] = std::make_shared<nnvm::any>(std::move(dtype_inputs));
  graph.attrs["shape_inputs"] = std::make_shared<nnvm::any>(std::move(shape_inputs));
  graph.attrs["dtype"]        = std::make_shared<nnvm::any>(std::move(dtypes));
  graph.attrs["shape"]        = std::make_shared<nnvm::any>(std::move(shapes));
  auto onnx_graph = op::nnvm_to_onnx::ConvertNnvmGraphToOnnx(graph, &params_map);
  auto trt_tuple = ::onnx_to_tensorrt::onnxToTrtCtx(onnx_graph, max_batch_size, 1 << 30);
  return OpStatePtr::Create<TRTEngineParam>(std::move(std::get<0>(trt_tuple)),
                                            std::move(std::get<1>(trt_tuple)),
                                            std::move(std::get<2>(trt_tuple)),
                                            inputs_to_idx, outputs_to_idx);
}

NNVM_REGISTER_OP(_TensorRT)
    .describe(R"code(TRT operation (one engine)
)code" ADD_FILELINE)
    .set_num_inputs(TRTNumInputs)
    .set_num_outputs(DefaultSubgraphOpNumOutputs)
    .set_attr_parser(TRTParamParser)
    .set_attr<mxnet::FInferShape>("FInferShape", TRTInferShape)
    .set_attr<nnvm::FInferType>("FInferType", TRTInferType)
    .set_attr<nnvm::FListInputNames>("FListInputNames", TRTListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames", DefaultSubgraphOpListOutputs)
    .set_attr<FCreateOpState>("FCreateOpState", TRTCreateState)
    .set_attr<FInferStorageType>("FInferStorageType", TRTInferStorageType);

MXNET_REGISTER_SUBGRAPH_BACKEND(TensorRT);

MXNET_REGISTER_SUBGRAPH_PROPERTY(TensorRT, TensorrtProperty);
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
