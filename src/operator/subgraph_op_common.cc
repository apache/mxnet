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

#include "./subgraph_op_common.h"
#include "./operator_common.h"
#include "../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

bool InferSubgraphDataType(const nnvm::Symbol &subgraph,
                           std::vector<int> *in_type,
                           std::vector<int> *out_type) {
  nnvm::DTypeVector dtype_inputs = *in_type;
  nnvm::Graph g;
  g.outputs = subgraph.outputs;
  const auto& idx = g.indexed_graph();
  CHECK_EQ(idx.input_nodes().size(), in_type->size());
  CHECK_EQ(idx.outputs().size(), out_type->size());
  imperative::CheckAndInferType(&g, std::move(dtype_inputs), true);

  const auto &dtypes = g.GetAttr<nnvm::DTypeVector>("dtype");

  // Inferring the data type in the subgraph may infer the data type of the inputs.
  // We need to copy the inferred input data types back.
  const auto &input_nids = idx.input_nodes();
  CHECK_EQ(input_nids.size(), in_type->size());
  for (size_t i = 0; i < in_type->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    TYPE_ASSIGN_CHECK(*in_type, i, dtypes[eid]);
  }

  for (size_t i = 0; i < g.outputs.size(); i++)
    TYPE_ASSIGN_CHECK(*out_type, i, dtypes[idx.entry_id(g.outputs[i])]);
  return true;
}

bool InferSubgraphStorage(const nnvm::Symbol &subgraph,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  nnvm::Graph g;
  g.outputs = subgraph.outputs;
  const auto& idx = g.indexed_graph();
  CHECK_EQ(idx.input_nodes().size(), in_attrs->size());
  CHECK_EQ(idx.outputs().size(), out_attrs->size());
  exec::DevMaskVector dev_masks(idx.num_nodes(), dev_mask);
  StorageTypeVector storage_type_inputs = *in_attrs;
  imperative::CheckAndInferStorageType(&g, std::move(dev_masks),
                                       std::move(storage_type_inputs), true);

  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");

  // Inferring the storage in the subgraph may infer the storage of the inputs.
  // We need to copy the inferred input storage back.
  const auto &input_nids = idx.input_nodes();
  CHECK_EQ(input_nids.size(), in_attrs->size());
  for (size_t i = 0; i < in_attrs->size(); i++) {
    auto eid = idx.entry_id(input_nids[i], 0);
    STORAGE_TYPE_ASSIGN_CHECK(*in_attrs, i, stypes[eid]);
  }

  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  auto &outputs = idx.outputs();
  CHECK(outputs.size() == out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); i++)
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, stypes[idx.entry_id(outputs[i])]);
  return true;
}

bool InferSubgraphBackwardStorage(const nnvm::Symbol &subgraph,
                                  const int dev_mask,
                                  DispatchMode* dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  using namespace nnvm;
  // construct backward graph
  nnvm::Graph grad_graph;
  nnvm::Graph fwd_graph;
  std::vector<Node *> potential_nodes;
  {
    fwd_graph.outputs = subgraph.outputs;
    std::vector<nnvm::NodeEntry> ograd_entries;
    ograd_entries.reserve(fwd_graph.outputs.size());
    for (size_t i = 0; i < fwd_graph.outputs.size(); ++i) {
      ograd_entries.emplace_back(NodeEntry{Node::Create(), 0, 0});
    }

    std::vector<NodeEntry> xs;
    std::vector<NodePtr> args = subgraph.ListInputs(nnvm::Symbol::kReadOnlyArgs);
    xs.reserve(args.size());
    for (const auto& i : args)
      xs.emplace_back(NodeEntry{i, 0, 0});
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";

    static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
    grad_graph = pass::Gradient(
        fwd_graph, fwd_graph.outputs, xs, ograd_entries,
        exec::AggregateGradient, nullptr, nullptr,
        zero_ops, "_copy");
    potential_nodes.reserve(fwd_graph.outputs.size() + xs.size() + ograd_entries.size());
    for (auto e : ograd_entries)
      potential_nodes.push_back(e.node.get());
    for (auto e : xs)
      potential_nodes.push_back(e.node.get());
    for (auto e : fwd_graph.outputs)
      potential_nodes.push_back(e.node.get());
  }

  const auto& idx = grad_graph.indexed_graph();
  auto input_nodes = idx.input_nodes();
  StorageTypeVector storage_type_inputs(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); i++) {
    auto node_id = input_nodes[i];
    const nnvm::IndexedGraph::Node &n = idx[node_id];
    auto it = std::find(potential_nodes.begin(), potential_nodes.end(), n.source);
    CHECK(it != potential_nodes.end());
    size_t idx = it - potential_nodes.begin();
    CHECK_LT(idx, in_attrs->size());
    storage_type_inputs[i] = in_attrs->at(idx);
  }
  CHECK_EQ(idx.outputs().size(), out_attrs->size());
  exec::DevMaskVector dev_masks(idx.num_nodes(), dev_mask);
  imperative::CheckAndInferStorageType(&grad_graph, std::move(dev_masks),
                                       std::move(storage_type_inputs), true);

  const auto& stypes = grad_graph.GetAttr<StorageTypeVector>("storage_type");
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  auto &outputs = idx.outputs();
  CHECK(outputs.size() == out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); i++)
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, stypes[idx.entry_id(outputs[i])]);
  return true;
}

}  // namespace op
}  // namespace mxnet
