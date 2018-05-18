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
 *  Copyright (c) 2016 by Contributors
 * \file quantization.cc
 * \brief
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <mxnet/op_attr_types.h>
#include <unordered_set>

namespace mxnet {
namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::Graph;

// use the function in quantize_graph.cc
NodePtr CreateNode(std::string, std::string);
NodePtr CloneVariableNode(const nnvm::Node& src) {
  CHECK(src.is_variable());
  CHECK_EQ(src.inputs.size(), 0U);
  CHECK_EQ(src.control_deps.size(), 0U);
  NodePtr node = nnvm::Node::Create();
  *node = src;
  return node;
}


Graph PartitionGraph(Graph&& g) {
#if 0
  DFSVisit(g.outputs, [&](const NodePtr& node) {
    LOG(INFO) << node->attrs.name;
    for (const auto& e : node->inputs) {
      LOG(INFO) << "NodeEntry: " << e.node->attrs.name << ", " << e.index << ", " << e.version;
    }
  });
#endif

  Symbol whole_graph_sym;
  whole_graph_sym.outputs = g.outputs;
  // DO NOT define node name for subgraph op because it would serve
  // as the prefix of the output names. We want to use the original
  // output names of the subgraph.
  NodePtr subgraph_node_ptr = CreateNode("_subgraph_op", "");
  subgraph_node_ptr->attrs.parsed = whole_graph_sym;
  const auto& idx = g.indexed_graph();
  const auto& input_node_ids = idx.input_nodes();
  for (size_t i = 0; i < input_node_ids.size(); ++i) {
    const auto& input_node = idx[input_node_ids[i]];
    // also need to clone the attrs of the source variable node
    NodePtr new_input_node = CloneVariableNode(*input_node.source);
    CHECK_EQ(new_input_node->inputs.size(), 0U);
    CHECK_EQ(subgraph_node_ptr->inputs.size(), i);
    nnvm::NodeEntry new_node_entry{new_input_node, 0, 0};
    subgraph_node_ptr->inputs.emplace_back(new_node_entry);
  }
  Graph ret;
  static const auto& flist_outputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  auto list_output_names_func = flist_outputs.get(subgraph_node_ptr->op(), nullptr);
  CHECK(list_output_names_func != nullptr);
  const size_t num_outputs = list_output_names_func(subgraph_node_ptr->attrs).size();
  for (uint32_t i = 0; i < num_outputs; ++i) {
    ret.outputs.emplace_back(nnvm::NodeEntry{subgraph_node_ptr, i, 0});
  }
  return ret;
}

NNVM_REGISTER_PASS(PartitionGraph)
.describe("")
.set_body(PartitionGraph)
.set_change_graph(true);

}  // namespace op
}  // namespace mxnet
