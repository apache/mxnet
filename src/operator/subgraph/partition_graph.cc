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
#include <queue>
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

namespace sg {  // sg stands for subgraph

struct SimpleNode;
using SimpleNodePtr = std::shared_ptr<SimpleNode>;

struct SimpleNode {
  static SimpleNodePtr Create() {
    return std::make_shared<SimpleNode>();
  }
  SimpleNode() : label(-1), node(nullptr) {}
  int label;
  nnvm::Node* node;
  std::unordered_map<Node*, int> outputs;
  //std::unordered_map<SimpleNodePtr, int> inputs;
  //std::unordered_map<SimpleNodePtr, int> outputs;
};

void CreateSimpleGraph(const Graph& g,
                       std::vector<SimpleNodePtr>* simple_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  simple_nodes->reserve(indexed_graph.num_nodes());
  for (size_t nid = 0; nid < indexed_graph.num_nodes(); ++nid) {
    SimpleNodePtr sn = SimpleNode::Create();
    sn->node = const_cast<nnvm::Node*>(indexed_graph[nid].source);
    for (auto& e : sn->node->inputs) {
      const auto input_nid = indexed_graph.node_id(e.node.get());
      CHECK_LT(input_nid, simple_nodes->size());
      std::unordered_map<Node*, int>& input_node_outputs = (*simple_nodes)[input_nid]->outputs;
      auto it = input_node_outputs.find(sn->node);
      if (it == input_node_outputs.end()) {
        input_node_outputs.emplace(sn->node, 1);
      } else {
        ++(it->second);
      }
    }
    simple_nodes->emplace_back(std::move(sn));
  }
#if 0
  DFSVisit(g.outputs, [&](const NodePtr& node) {
    LOG(INFO) << node->attrs.name;
    auto it = node_map2->find(node.get());
    if (it == node_map2.end()) {
    }

    for (const auto& e : node->inputs) {
      LOG(INFO) << "NodeEntry: " << e.node->attrs.name << ", " << e.index << ", " << e.version;
    }
  });
#endif
}

void LabelSubgraph(const Graph&g,
                   const std::unordered_set<std::string>& op_names,
                   const int label,
                   const size_t snid,  // simple node id
                   const std::vector<SimpleNodePtr>& simple_nodes,
                   std::vector<SimpleNode*>* subgraph_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  std::queue<SimpleNode*> node_queue;
  node_queue.push(simple_nodes[snid].get());
  while (!node_queue.empty()) {
    SimpleNode* cur_node = node_queue.front();
    node_queue.pop();
    cur_node->label = label;
    subgraph_nodes->push_back(cur_node);
    // get qualified adjacent input nodes
    for (auto& e : cur_node->node->inputs) {
      if (!e.node->is_variable() && op_names.count(e.node->op()->name)) {
        const auto nid = indexed_graph.node_id(e.node.get());
        CHECK_LT(nid, simple_nodes.size());
        if (simple_nodes[nid]->label == -1) {  // this node has not been visited yet
          node_queue.push(simple_nodes[nid].get());
        } else {
          CHECK_EQ(simple_nodes[nid]->label, label);
        }
      }
    }
    // get qualified output nodes
    for (auto it = cur_node->outputs.begin(); it != cur_node->outputs.end(); ++it) {
      CHECK(!it->first->is_variable());
      if (op_names.count(it->first->op()->name)) {
        const auto nid = indexed_graph.node_id(it->first);
        CHECK_LT(nid, simple_nodes.size());
        if (simple_nodes[nid]->label == -1) {  // this node has not been visited yet
          node_queue.push(simple_nodes[nid].get());
        } else {
          CHECK_EQ(simple_nodes[nid]->label, label);
        }
      }
    }
  }
}

// number of subgraphs found
void FindSubgraphs(const Graph& g,
                   const std::unordered_set<std::string>& op_names,
                   const std::vector<SimpleNodePtr>& simple_nodes,
                   std::vector<std::vector<SimpleNode*>>* subgraph_nodes) {
  //CHECK(simple_nodes != nullptr);
  const auto& indexed_graph = g.indexed_graph();
  CHECK_EQ(indexed_graph.num_nodes(), simple_nodes.size());
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    nnvm::Node* node = simple_nodes[i]->node;
    if (!node->is_variable() && simple_nodes[i]->label == -1 && op_names.count(node->op()->name)) {
      subgraph_nodes->emplace_back();
      LabelSubgraph(g, op_names, subgraph_nodes->size() - 1, i, simple_nodes, &subgraph_nodes->back());
    }
  }
}

}  // namespace sg

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
