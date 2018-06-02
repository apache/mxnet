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

#include "./subgraph_op.h"

namespace nnvm {
NodePtr CreateVariableNode(const std::string& name);
}

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

void CreateSimpleGraph(const Graph& g,
                       std::vector<SimpleNodePtr>* simple_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  simple_nodes->reserve(indexed_graph.num_nodes());
  for (size_t nid = 0; nid < indexed_graph.num_nodes(); ++nid) {
    SimpleNodePtr sn = SimpleNode::Create();
    sn->node = const_cast<nnvm::Node*>(indexed_graph[nid].source);
    for (size_t i = 0; i < sn->node->inputs.size(); ++i) {
      const auto& e = sn->node->inputs[i];
      const auto input_nid = indexed_graph.node_id(e.node.get());
      CHECK_LT(input_nid, simple_nodes->size());
      auto& input_node_outputs = (*simple_nodes)[input_nid]->outputs;
      auto it = input_node_outputs.find(sn->node);
      if (it == input_node_outputs.end()) {
        input_node_outputs.emplace(sn->node, std::vector<int>{static_cast<int>(i)});
      } else {
        it->second.push_back(i);
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

void PrintSubgraph(const std::vector<SimpleNode*>& simple_nodes) {
  std::string op_names = "";
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    op_names += simple_nodes[i]->node->attrs.name + ' ';
  }
  LOG(INFO) << "Subgraph node names: " << op_names;
}

/*
 * This function traverses the nodes in a computation graph from a starting
 * node following the input links and output links, and marks all nodes that
 * can be accessed from the starting node.
 */
void LabelSubgraph(const Graph&g,
                   SubgraphSelectorPtr select_func,
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
      if (select_func->SelectInput(*cur_node->node, *e.node)) {
        const auto nid = indexed_graph.node_id(e.node.get());
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1)
          node_queue.push(simple_nodes[nid].get());
      }
    }
    // get qualified output nodes
    for (auto it = cur_node->outputs.begin(); it != cur_node->outputs.end(); ++it) {
      if (select_func->SelectOutput(*cur_node->node, *it->first)) {
        const auto nid = indexed_graph.node_id(it->first);
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1)
          node_queue.push(simple_nodes[nid].get());
      }
    }
  }
}

/*
 * This function finds subgraphs with all nodes that meet certain criteria.
 * All nodes in a subgraph are marked with the same label.
 * All nodes in a subgraph have to be connected with each other. If a node
 * doesn't meet the given criteria, it will be marked with a separate label.
 */
void FindSubgraphs(const Graph& g,
                   const SubgraphProperty &subg_prop,
                   const std::vector<SimpleNodePtr>& simple_nodes,
                   std::vector<std::vector<SimpleNode*>>* subgraph_nodes) {
  //CHECK(simple_nodes != nullptr);
  const auto& indexed_graph = g.indexed_graph();
  CHECK_EQ(indexed_graph.num_nodes(), simple_nodes.size());
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    nnvm::Node* node = simple_nodes[i]->node;
    auto select_func = subg_prop.CreateSubgraphSelector();
    if (select_func->Select(*node) && simple_nodes[i]->label == -1) {
      subgraph_nodes->emplace_back();
      LabelSubgraph(g, select_func, subgraph_nodes->size() - 1, i, simple_nodes,
                    &subgraph_nodes->back());
    }
  }
}

// Reorder entries according to their source nodes' topological order
void ReorderEntries(const Graph& g, std::vector<nnvm::NodeEntry*>* entries) {
  const auto& indexed_graph = g.indexed_graph();
  auto top_cmp = [&](const nnvm::NodeEntry* e1, const nnvm::NodeEntry* e2) {
    const auto nid1 = indexed_graph.node_id(e1->node.get());
    const auto nid2 = indexed_graph.node_id(e2->node.get());
    if (nid1 == nid2) {
      return e1->index < e2->index;
    }
    return nid1 < nid2;
  };
  std::sort(entries->begin(), entries->end(), top_cmp);
}

// find the input entries of a subgraph
void FindInputEntries(const Graph& g,
                      const std::vector<SimpleNodePtr>& simple_nodes,
                      const std::vector<SimpleNode*>& subgraph_nodes,
                      std::vector<nnvm::NodeEntry*>* input_entries) {
  const auto& indexed_graph = g.indexed_graph();
  int label = -1;
  for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
    if (label == -1) {
      label = subgraph_nodes[i]->label;
    } else {
      CHECK_EQ(subgraph_nodes[i]->label, label);
    }
    for (auto& e : subgraph_nodes[i]->node->inputs) {
      const auto nid = indexed_graph.node_id(e.node.get());
      // this is a node not belonging to the subgraph
      if (simple_nodes[nid]->label != label)
        input_entries->push_back(&e);
    }
  }
  ReorderEntries(g, input_entries);
}

// find the output entries of a subgraph
void FindOutputEntries(Graph* g,
                       const std::vector<SimpleNodePtr>& simple_nodes,
                       const std::vector<SimpleNode*>& subgraph_nodes,
                       std::vector<nnvm::NodeEntry*>* output_entries) {
  if (subgraph_nodes.empty()) return;
  const auto& indexed_graph = g->indexed_graph();
  int label = -1;
  for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
    if (label == -1) {
      label = subgraph_nodes[i]->label;
    } else {
      CHECK_EQ(subgraph_nodes[i]->label, label);
    }
    for (auto it = subgraph_nodes[i]->outputs.begin();
         it != subgraph_nodes[i]->outputs.end(); ++it) {
      const auto nid = indexed_graph.node_id(it->first);
      // this is a node not belonging to the subgraph
      if (simple_nodes[nid]->label != label) {
        // TODO(zhengda) I need to test this.
        for (int idx : it->second) {
          output_entries->push_back(&simple_nodes[nid]->node->inputs[idx]);
        }
      }
    }
  }
  // Check if current subgraph contains a node which is the last node
  // of the whole graph. If so, save its corresponding entry as well.
  for (auto& entry : g->outputs) {
    const auto nid = indexed_graph.node_id(entry.node.get());
    if (simple_nodes[nid]->label == label) {
      output_entries->push_back(&entry);
    }
  }
  ReorderEntries(*g, output_entries);
}

void PrintNodeEntry(const nnvm::NodeEntry& entry) {
  std::string ret = "NodeEntry: node_name=" + entry.node->attrs.name
    + ", index=" + std::to_string(entry.index) + ", version=" + std::to_string(entry.version);
  LOG(INFO) << ret;
}

void PrintNodeEntries(const std::vector<nnvm::NodeEntry*>& entries) {
  for (size_t i = 0; i < entries.size(); ++i) {
    PrintNodeEntry(*entries[i]);
  }
}

/*
 * Given a computation graph and a set of input node entries, this function cuts
 * the node entries and creates new variable nodes as the input nodes of the
 * subgraph. It returns the nodes that connect to the subgraph directly and
 * the names of the new variable nodes.
 */
void CutGraphInputs(const std::vector<nnvm::NodeEntry *> &input_entries,
                    bool skip_var, std::vector<nnvm::NodeEntry> *orig_entries) {
  orig_entries->reserve(input_entries.size());
  for (size_t i = 0; i < input_entries.size(); i++) {
    nnvm::NodeEntry *e = input_entries[i];
    // If the node is a variable itself, we may want to skip the node.
    if (e->node->is_variable() && skip_var)
      continue;

    orig_entries->push_back(*e);
    nnvm::Symbol sym;
    sym.outputs.push_back(*e);
    const auto output_names = sym.ListOutputNames();
    CHECK_EQ(output_names.size(), 1U);
    nnvm::NodePtr n = nnvm::CreateVariableNode(output_names[0]);
    *e = nnvm::NodeEntry{n, 0, 0};
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
  if (!g.HasAttr("subgraph_property")) {  // treat the whole graph as a subgraph
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
  } else {
    using namespace sg;
    const SubgraphPropertyPtr& subg_prop = g.GetAttr<SubgraphPropertyPtr>("subgraph_property");
    std::vector<SimpleNodePtr> simple_nodes;
    CreateSimpleGraph(g, &simple_nodes);
    std::vector<std::vector<SimpleNode*>> subgraph_nodes;
    FindSubgraphs(g, *subg_prop, simple_nodes, &subgraph_nodes);
    std::vector<nnvm::NodeEntry*> entries;
    // TODO(junwu): take care of the situation when the op is the last op
    for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
      PrintSubgraph(subgraph_nodes[i]);

      // Break the input links.
      LOG(INFO) << "Searching for input entries...";
      entries.clear();
      FindInputEntries(g, simple_nodes, subgraph_nodes[i], &entries);
      std::vector<nnvm::NodeEntry> orig_input_entries;
      sg::CutGraphInputs(entries, false, &orig_input_entries);
      PrintNodeEntries(entries);

      LOG(INFO) << "Searching for output entries...";
      entries.clear();
      FindOutputEntries(&g, simple_nodes, subgraph_nodes[i], &entries);

      // Create a subgraph.
      nnvm::Symbol sym;
      sym.outputs.resize(entries.size());
      for (size_t i = 0; i < entries.size(); i++)
        sym.outputs[i] = *entries[i];
      nnvm::NodePtr n = subg_prop->CreateSubgraphNode(sym);

      // Connect the external nodes to the subgraph node.
      for (uint32_t i = 0; i < entries.size(); i++)
        *entries[i] = nnvm::NodeEntry{n, i, 0};
      // TODO(zhengda) this may not be the right order for input entries of a subgraph?
      n->inputs = orig_input_entries;
      PrintNodeEntries(entries);
    }
    return g;
  }
}

NNVM_REGISTER_PASS(PartitionGraph)
.describe("")
.set_body(PartitionGraph)
.set_change_graph(true);

}  // namespace op
}  // namespace mxnet
