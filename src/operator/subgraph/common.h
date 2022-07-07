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

#ifndef MXNET_OPERATOR_SUBGRAPH_COMMON_H_
#define MXNET_OPERATOR_SUBGRAPH_COMMON_H_

#include <string>
#include <set>
#include <vector>
#include "../elemwise_op_common.h"
#include "../../imperative/exec_pass.h"

namespace mxnet {
namespace op {

enum SelectStatus { kFail = 0, kStart, kSuccess };

inline uint32_t DefaultSubgraphOpNumInputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListInputNames(nnvm::Symbol::kAll).size();
}

inline uint32_t DefaultSubgraphOpNumOutputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListOutputNames().size();
}

inline std::vector<std::string> DefaultSubgraphOpListInputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListInputNames(nnvm::Symbol::kAll);
}

inline std::vector<std::string> DefaultSubgraphOpListOutputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListOutputNames();
}

inline bool DefaultSubgraphOpShapeHelper(const nnvm::Symbol& subgraph_sym,
                                         mxnet::ShapeVector* in_shapes,
                                         mxnet::ShapeVector* out_shapes) {
  using namespace exec;
  nnvm::Graph g;
  g.outputs         = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_shapes->size());
  CHECK_EQ(idx_g.outputs().size(), out_shapes->size());

  // Put the input and output shapes to the shape vector.
  mxnet::ShapeVector shapes(idx_g.num_node_entries());
  const auto& input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_shapes->size());
  for (size_t i = 0; i < in_shapes->size(); i++) {
    auto eid    = idx_g.entry_id(input_nids[i], 0);
    shapes[eid] = in_shapes->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_shapes->size());
  for (size_t i = 0; i < out_shapes->size(); i++) {
    auto eid    = idx_g.entry_id(g.outputs[i]);
    shapes[eid] = out_shapes->at(i);
  }

  // Infer shape of the graph.
  g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
  g                = exec::InferShape(std::move(g));

  // Copy the inferred shape back to the input shapes and the output shapes.
  shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  // assign to in_shapes
  for (size_t i = 0; i < in_shapes->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    SHAPE_ASSIGN_CHECK(*in_shapes, i, shapes[eid]);
  }
  // assign to out_shapes
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    SHAPE_ASSIGN_CHECK(*out_shapes, i, shapes[eid]);
  }
  // Check if we have inferred the shapes correctly.
  return g.GetAttr<size_t>("shape_num_unknown_nodes") == 0;
}

inline bool DefaultSubgraphOpShape(const nnvm::NodeAttrs& attrs,
                                   mxnet::ShapeVector* in_shapes,
                                   mxnet::ShapeVector* out_shapes) {
  return DefaultSubgraphOpShapeHelper(*attrs.subgraphs[0], in_shapes, out_shapes);
}

inline bool DefaultSubgraphOpTypeHelper(const nnvm::Symbol& subgraph_sym,
                                        std::vector<int>* in_types,
                                        std::vector<int>* out_types) {
  nnvm::Graph g;
  g.outputs         = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_types->size());
  CHECK_EQ(idx_g.outputs().size(), out_types->size());

  // Put the input and output data types to the dtype vector.
  nnvm::DTypeVector types(idx_g.num_node_entries(), -1);
  const auto& input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_types->size());
  for (size_t i = 0; i < in_types->size(); i++) {
    auto eid   = idx_g.entry_id(input_nids[i], 0);
    types[eid] = in_types->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_types->size());
  for (size_t i = 0; i < out_types->size(); i++) {
    auto eid   = idx_g.entry_id(g.outputs[i]);
    types[eid] = out_types->at(i);
  }

  // Infer data type of the graph.
  g.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(types));
  g                = exec::InferType(std::move(g));

  types = g.GetAttr<nnvm::DTypeVector>("dtype");
  // assign to in_types
  for (size_t i = 0; i < in_types->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    TYPE_ASSIGN_CHECK(*in_types, i, types[eid]);
  }
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    TYPE_ASSIGN_CHECK(*out_types, i, types[eid]);
  }
  // Check if we have inferred the dtypes correctly.
  return g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0;
}

inline bool DefaultSubgraphOpType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int>* in_types,
                                  std::vector<int>* out_types) {
  return DefaultSubgraphOpTypeHelper(*attrs.subgraphs[0], in_types, out_types);
}

inline bool DefaultSubgraphOpStorageTypeHelper(const nnvm::Symbol& subgraph_sym,
                                               const int dev_mask,
                                               DispatchMode* dispatch_mode,
                                               std::vector<int>* in_stypes,
                                               std::vector<int>* out_stypes) {
  nnvm::Graph g;
  g.outputs         = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_stypes->size());
  CHECK_EQ(idx_g.outputs().size(), out_stypes->size());
  exec::DevMaskVector dev_masks(idx_g.num_node_entries(), dev_mask);

  // Put the input and output storages to the storage vector.
  StorageTypeVector stypes(idx_g.num_node_entries(), kUndefinedStorage);
  const auto& input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_stypes->size());
  for (size_t i = 0; i < in_stypes->size(); i++) {
    auto eid    = idx_g.entry_id(input_nids[i], 0);
    stypes[eid] = in_stypes->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_stypes->size());
  for (size_t i = 0; i < out_stypes->size(); i++) {
    auto eid    = idx_g.entry_id(g.outputs[i]);
    stypes[eid] = out_stypes->at(i);
  }

  // Infer storage type of the graph.
  bool dev_match =
      g.attrs.count("dev_mask") && g.GetAttr<exec::DevMaskVector>("dev_mask") == dev_masks;
  if (!dev_match) {
    g.attrs["dev_mask"] = std::make_shared<dmlc::any>(std::move(dev_masks));
  }
  g.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(stypes));
  g                       = exec::InferStorageType(std::move(g));

  stypes = g.GetAttr<StorageTypeVector>("storage_type");
  // assign to in_types
  for (size_t i = 0; i < in_stypes->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    STORAGE_TYPE_ASSIGN_CHECK(*in_stypes, i, stypes[eid]);
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

inline bool DefaultSubgraphOpStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int>* in_stypes,
                                         std::vector<int>* out_stypes) {
  return DefaultSubgraphOpStorageTypeHelper(
      *attrs.subgraphs[0], dev_mask, dispatch_mode, in_stypes, out_stypes);
}

inline ExecType DefaultSubgraphOpExecType(const nnvm::NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
}

inline std::vector<uint32_t> DefaultSubgraphOpMutableInputsHelper(
    const nnvm::Symbol& subgraph_sym) {
  const std::vector<std::string> input_names = subgraph_sym.ListInputNames(nnvm::Symbol::kAll);
  const std::vector<std::string> immutable_input_names =
      subgraph_sym.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
  const std::vector<std::string> mutable_input_names =
      subgraph_sym.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
  CHECK_EQ(immutable_input_names.size() + mutable_input_names.size(), input_names.size());
  std::vector<uint32_t> ret;
  size_t i1 = 0, i2 = 0;
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (i1 < immutable_input_names.size() && input_names[i] == immutable_input_names[i1]) {
      ++i1;
    } else {
      CHECK(i2 < mutable_input_names.size());
      CHECK_EQ(input_names[i], mutable_input_names[i2]);
      ++i2;
      ret.push_back(i);
    }
  }
  return ret;
}

inline std::vector<uint32_t> DefaultSubgraphOpMutableInputs(const nnvm::NodeAttrs& attrs) {
  return DefaultSubgraphOpMutableInputsHelper(*attrs.subgraphs[0]);
}

inline std::vector<ResourceRequest> DefaultSubgraphOpResourceRequestHelper(
    const nnvm::Symbol& subgraph_sym) {
  static auto& fresource = Op::GetAttr<FResourceRequest>("FResourceRequest");
  std::set<ResourceRequest::Type> resource_types;
  DFSVisit(subgraph_sym.outputs, [&](const nnvm::ObjectPtr& node) {
    if (!node->is_variable() && fresource.count(node->op())) {
      for (ResourceRequest& r : fresource[node->op()](node->attrs)) {
        resource_types.insert(r.type);
      }
    }
  });
  return std::vector<ResourceRequest>(resource_types.begin(), resource_types.end());
}

inline std::vector<ResourceRequest> DefaultSubgraphOpResourceRequest(const nnvm::NodeAttrs& attrs) {
  return DefaultSubgraphOpResourceRequestHelper(*attrs.subgraphs[0]);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_COMMON_H_
