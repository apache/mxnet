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
 * \file partition_graph.cc
 * \brief
 */
#include <queue>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <mxnet/op_attr_types.h>
#include <unordered_set>
#include <stack>

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
        input_node_outputs.emplace(sn->node, std::vector<size_t>{i});
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

void ResetSubgraphNodes(std::vector<SimpleNode*>* subgraph_nodes) {
  for (auto sn : *subgraph_nodes) {
    sn->label = -1;
  }
  subgraph_nodes->clear();
}

/*
 * This function traverses the nodes in a computation graph from a starting
 * node following the input links and output links, and marks all nodes that
 * can be accessed from the starting node. Before the function returns,
 * it will conduct checking whether there is a loop between the potential subgraph
 * and the outside nodes. If so, add the node that should break the loop
 * in excluded_nodes and return false. Otherwise, return true.
 */
bool LabelSubgraph(const Graph& g,
                   SubgraphSelectorPtr select_func,
                   const int label,
                   const size_t snid,  // simple node id, this is a seed
                   const std::vector<SimpleNodePtr>& simple_nodes,
                   std::vector<SimpleNode*>* subgraph_nodes,
                   std::unordered_set<const nnvm::Node*>* excluded_nodes = nullptr) {
  const auto& indexed_graph = g.indexed_graph();
  std::queue<SimpleNode*> node_queue;
  if (!excluded_nodes || !excluded_nodes->count(simple_nodes[snid]->node)) {
    node_queue.push(simple_nodes[snid].get());
  }
  // key: nodes that serve as input/output nodes to the subgraph
  // value: pair of vectors of nodes in the subgraph. The first vector contains the output nodes of the key in the subgraph,
  // and the second vector contains the input ndoes of the key in the subgraph.
  // If both vectors are non-empty, it means there is a loop between the subgraph and the key node.
  // When breaking the loop, we want to start removing the node with the largest node id.
  std::unordered_map<const nnvm::Node*,
    std::pair<std::vector<const nnvm::Node*>, std::vector<const nnvm::Node*>>> non_subgraph_node_map;
  while (!node_queue.empty()) {
    SimpleNode* cur_node = node_queue.front();
    node_queue.pop();
    cur_node->label = label;
    subgraph_nodes->push_back(cur_node);
    // get qualified adjacent input nodes
    for (auto& e : cur_node->node->inputs) {
      const bool select_input = (!excluded_nodes || !excluded_nodes->count(e.node.get()))
        && select_func->SelectInput(*cur_node->node, *e.node);
      if (select_input) {
        // e.node is a subgraph node
        const auto nid = indexed_graph.node_id(e.node.get());
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1) {
          node_queue.push(simple_nodes[nid].get());
        }
      } else {
        // e.node is an input node of the subgraph
        non_subgraph_node_map[e.node.get()].first.push_back(cur_node->node);
      }
    }
    // get qualified output nodes
    for (auto it = cur_node->outputs.begin(); it != cur_node->outputs.end(); ++it) {
      const bool select_output = (!excluded_nodes || !excluded_nodes->count(it->first))
          && select_func->SelectOutput(*cur_node->node, *it->first);
      if (select_output) {
        // it->first is a subgraph node
        const auto nid = indexed_graph.node_id(it->first);
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1) {
          node_queue.push(simple_nodes[nid].get());
        }
      } else {
        // it->first is an output node of the subgraph
        non_subgraph_node_map[it->first].second.push_back(cur_node->node);
      }
    }
  }
  auto node_cmp = [&] (const nnvm::Node* node1, const nnvm::Node* node2) {
    return indexed_graph.node_id(node1) < indexed_graph.node_id(node2);
  };
  // check whether there is a loop between the subgraph and its input/output nodes
  int excluded_node_id = -1;
  for (auto& kv : non_subgraph_node_map) {
    auto& output_nodes = kv.second.first;
    auto& input_nodes = kv.second.second;
    if (!output_nodes.empty() && !input_nodes.empty()) {
      // there is a loop between kv->first and the subgraph
      std::sort(output_nodes.begin(), output_nodes.end(), node_cmp);
      std::sort(input_nodes.begin(), input_nodes.end(), node_cmp);
      const auto node_id = std::max(indexed_graph.node_id(output_nodes.back()),
                                    indexed_graph.node_id(input_nodes.back()));
      excluded_node_id = std::max(excluded_node_id, static_cast<int>(node_id));
    }
  }
  if (excluded_node_id != -1) {
    CHECK_LT(excluded_node_id, static_cast<int>(simple_nodes.size()));
    CHECK_NE(excluded_node_id, static_cast<int>(snid))
      << "A cycle is found in the computational graph between nodes "
      << simple_nodes[excluded_node_id]->node->attrs.name << " and "
      << simple_nodes[snid]->node->attrs.name;
    excluded_nodes->insert(simple_nodes[excluded_node_id]->node);
    ResetSubgraphNodes(subgraph_nodes);
    return false;
  }
  return true;
}

void FindSubgraph(const Graph& g,
                  SubgraphSelectorPtr select_func,
                  const int label,
                  const size_t snid,  // simple node id, this is a seed
                  const std::vector<SimpleNodePtr>& simple_nodes,
                  std::vector<SimpleNode*>* subgraph_nodes) {
  std::unordered_set<const nnvm::Node*> excluded_nodes;
  const size_t max_num_retry = simple_nodes.size() * simple_nodes.size();
  size_t count = 0;
  bool success = false;
  while (!success && count < max_num_retry) {
    success = LabelSubgraph(g, select_func, label, snid, simple_nodes, subgraph_nodes, &excluded_nodes);
    if (!success) {
      CHECK(!excluded_nodes.empty());
      std::string excluded_node_names;
      for (auto node : excluded_nodes) {
        excluded_node_names += node->attrs.name + ", ";
      }
      LOG(INFO) << "Found a cycle when BFS from node " << simple_nodes[snid]->node->attrs.name
                << ". Excluding nodes " << excluded_node_names << "and retrying";
    }
    ++count;
  }
  if (!success) {
    LOG(INFO) << "Tried " << count << " times of finding subgraphs starting from node "
               << simple_nodes[snid]->node->attrs.name << " without success because a loop "
                  "is always found between the subgraph and some other nodes. Will treat "
                  "seed node " << simple_nodes[snid]->node->attrs.name << "as a subgraph with one node";
    CHECK(subgraph_nodes->empty());
    simple_nodes[snid]->label = label;
    subgraph_nodes->push_back(simple_nodes[snid].get());
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
      FindSubgraph(g, select_func, subgraph_nodes->size() - 1, i, simple_nodes,
                   &subgraph_nodes->back());
    }
  }
}

/*!
 * \brief Sort entries according to their topological order.
 * Note that entry ids cannot be used to sort entries.
 * \param entry_top_order_map mapping from entry pointer to its topological position in the graph
 * \param entries Node entries to be sorted
 */
void SortEntries(const std::unordered_map<const nnvm::NodeEntry*, size_t>& entry_top_order_map,
                 std::vector<nnvm::NodeEntry*>* entries) {
  auto entry_cmp = [&](const nnvm::NodeEntry* e1, const nnvm::NodeEntry* e2) {
    const auto it1 = entry_top_order_map.find(e1);
    CHECK(it1 != entry_top_order_map.end());
    const auto it2 = entry_top_order_map.find(e2);
    CHECK(it2 != entry_top_order_map.end());
    return it1->second < it2->second;
  };
  std::sort(entries->begin(), entries->end(), entry_cmp);
}

/*
 * \brief find the input entries of a subgraph
 * \param input_entry_map mapping from node entry pointer to the
 * pair of its dest node and index in the dest node's inputs
 */
void FindInputEntries(const Graph& g,
                      const std::vector<SimpleNodePtr>& simple_nodes,
                      const std::vector<SimpleNode*>& subgraph_nodes,
                      const std::unordered_map<const nnvm::NodeEntry*, size_t>& entry_top_order_map,
                      std::vector<nnvm::NodeEntry*>* input_entries) {
  const auto& indexed_graph = g.indexed_graph();
  int label = -1;
  for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
    if (label == -1) {
      label = subgraph_nodes[i]->label;
    } else {
      CHECK_EQ(subgraph_nodes[i]->label, label);
    }
    auto& inputs = subgraph_nodes[i]->node->inputs;
    for (size_t j = 0; j < inputs.size(); ++j) {
      auto& e = inputs[j];
      if (indexed_graph.exist(e.node.get())) {
        // e's source node is not a subgraph node
        const auto nid = indexed_graph.node_id(e.node.get());
        // this is a node not belonging to the subgraph
        if (simple_nodes[nid]->label != label) {
          input_entries->push_back(&e);
        }
      } else {
        // e's source node is a subgraph node.
        // In this case, two subgraphs are adjacent.
        input_entries->push_back(&e);
      }
    }
  }
  SortEntries(entry_top_order_map, input_entries);
}

// find the output entries of a subgraph
void FindOutputEntries(Graph* g,
                       const std::vector<SimpleNodePtr>& simple_nodes,
                       const std::vector<SimpleNode*>& subgraph_nodes,
                       const std::unordered_map<const nnvm::NodeEntry*, size_t>& entry_top_order_map,
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
      if (indexed_graph.exist(it->first)) {
        // if the output node is a normal graph node (not a subgraph node)
        const auto nid = indexed_graph.node_id(it->first);
        // this is a node not belonging to the current subgraph
        if (simple_nodes[nid]->label != label) {
          // TODO(zhengda) I need to test this.
          for (auto idx : it->second) {
            auto& e = simple_nodes[nid]->node->inputs[idx];
            output_entries->push_back(&e);
          }
        }
      } else {
        // if the output node is a subgraph node
        // two graphs are adjacent
        for (auto idx : it->second) {
          output_entries->push_back(&(it->first->inputs[idx]));
        }
      }
    }
  }
  // Check if current subgraph contains a node which is the last node
  // of the whole graph. If so, save its corresponding entry as well.
  for (size_t i = 0; i < g->outputs.size(); ++i) {
    auto& entry = g->outputs[i];
    // The entry might has been updated as an output of
    // a subgraph node. In this case, no need
    // to check its source for the current subgraph. Otherwise,
    // do the following.
    if (indexed_graph.exist(entry.node.get())) {
      const auto nid = indexed_graph.node_id(entry.node.get());
      if (simple_nodes[nid]->label == label) {
        output_entries->push_back(&entry);
      }
    }
  }
  SortEntries(entry_top_order_map, output_entries);
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
 * \brief Given a computation graph and a set of input node entries, this function cuts
 * the node entries and creates new variable nodes as the input nodes of the
 * subgraph. It returns the nodes that connect to the subgraph directly and
 * the names of the new variable nodes.
 */
void CutGraphInputs(const std::vector<nnvm::NodeEntry*> &input_entries,
                    std::vector<nnvm::NodeEntry> *orig_entries,
                    const bool skip_var = false) {
  orig_entries->resize(input_entries.size());
  for (size_t i = 0; i < input_entries.size(); ++i) {
    nnvm::NodeEntry *e = input_entries[i];
    // If the node is a variable itself, we may want to skip the node.
    if (e->node->is_variable() && skip_var) {
      continue;
    }

    orig_entries->at(i) = *e;
    nnvm::Symbol sym;
    sym.outputs.push_back(*e);
    const auto output_names = sym.ListOutputNames();
    CHECK_EQ(output_names.size(), 1U);
    nnvm::NodePtr n = nnvm::CreateVariableNode(output_names[0]);
    *e = nnvm::NodeEntry{n, 0, 0};
  }
}

// Replace a set of nodes belonging to the same subgraph with a subgrpah node
// and keep the subgraph in the subgraph node. The input entries and output entries
// of the subgraph node are kept in the same order as the subgraph's.
void CreateSubgraphNode(Graph* g,
                        const std::vector<SimpleNodePtr>& simple_nodes,
                        const std::vector<SimpleNode*>& subgraph_nodes,
                        const size_t subgraph_id,
                        std::unordered_map<const nnvm::NodeEntry*, size_t>* entry_top_order_map) {
  LOG(INFO) << "Searching for input entries...";
  std::vector<nnvm::NodeEntry*> input_entries;
  FindInputEntries(*g, simple_nodes, subgraph_nodes, *entry_top_order_map, &input_entries);
  std::vector<nnvm::NodeEntry> orig_input_entries;
  // TODO(junwu): Confirm what value to pass to skip_var
  CutGraphInputs(input_entries, &orig_input_entries, false);
  PrintNodeEntries(input_entries);

  LOG(INFO) << "Searching for output entries...";
  std::vector<nnvm::NodeEntry*> output_entries;
  FindOutputEntries(g, simple_nodes, subgraph_nodes, *entry_top_order_map, &output_entries);

  // Create a subgraph for the subgraph node
  nnvm::Symbol sym;
  sym.outputs.resize(output_entries.size());
  for (size_t i = 0; i < output_entries.size(); ++i) {
    sym.outputs[i] = *output_entries[i];
  }
  const SubgraphPropertyPtr& subg_prop = g->GetAttr<SubgraphPropertyPtr>("subgraph_property");
  nnvm::NodePtr n = subg_prop->CreateSubgraphNode(sym, subgraph_id);

  // Connect the external nodes to the subgraph node.
  for (size_t i = 0; i < output_entries.size(); ++i) {
    *output_entries[i] = nnvm::NodeEntry{n, static_cast<uint32_t>(i), 0};
  }
  n->inputs = orig_input_entries;
  const auto& indexed_graph = g->indexed_graph();
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    auto& e = n->inputs[i];
    // update entry_top_order_map with newly created orig_input_entries
    auto it = entry_top_order_map->find(input_entries[i]);
    CHECK(it != entry_top_order_map->end());
    CHECK_EQ(entry_top_order_map->count(&e), 0U);
    entry_top_order_map->emplace(&e, it->second);
    // update input entries' source simple nodes' outputs map
    nnvm::Node* node = e.node.get();
    if (indexed_graph.exist(node)) {
      const auto nid = indexed_graph.node_id(node);
      SimpleNode* sn = simple_nodes[nid].get();
      for (SimpleNode* dest_node : subgraph_nodes) {
        sn->outputs.erase(dest_node->node);
      }
      sn->outputs[n.get()].push_back(i);
    }
  }
  PrintNodeEntries(output_entries);
}

}  // namespace sg

/*!
 * \brief Sort entries of all the nodes' inputs vectors in the topological order.
 * This is going to be used to sort input/output entries of subgraphs to keep
 * the topological order unchanged.
 */
void TopSortEntries(const Graph& g, std::unordered_map<const nnvm::NodeEntry*, size_t>* entry_top_order_map) {
  CHECK(entry_top_order_map != nullptr);
  std::unordered_set<const nnvm::Node*> visited;
  // meaning of tuple: (graph node, index of node's inputs, node entry as the output of the graph node)
  std::stack<std::tuple<nnvm::Node*, size_t, const nnvm::NodeEntry*>> s;
  auto in_degree = [] (const nnvm::Node* node)->size_t {
    if (!node) {
      return 0;
    }
    CHECK_EQ(node->control_deps.size(), 0U);
    return node->inputs.size();
  };
  for (auto& e : g.outputs) {
    nnvm::Node* node = e.node.get();
    if (visited.count(node) == 0U) {
      s.emplace(node, 0U, &e);
      visited.insert(node);
    }
    while (!s.empty()) {
      auto& top = s.top();
      if (std::get<1>(top) == in_degree(std::get<0>(top))) {
        // The node's inputs has been exhausted.
        entry_top_order_map->emplace(std::get<2>(top), entry_top_order_map->size());
        s.pop();
      } else {
        // The node still has input entries not visited.
        CHECK_LT(std::get<1>(top), std::get<0>(top)->inputs.size());
        auto& entry = std::get<0>(top)->inputs[std::get<1>(top)++];
        nnvm::Node* input_node = entry.node.get();
        if (visited.count(input_node) == 0U) {
          // The entry's source node has not been visited.
          // Push the entry to the stack for marking order later.
          s.emplace(input_node, 0U, &entry);
          visited.insert(input_node);
        } else {
          // The entry's source node has been visited before.
          // Marking order for it.
          entry_top_order_map->emplace(&entry, entry_top_order_map->size());
        }
      }
    }
  }
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
    // top sort NodeEntry of all the nodes' inputs
    std::unordered_map<const nnvm::NodeEntry*, size_t> entry_top_order_map;
    TopSortEntries(g, &entry_top_order_map);

    // Create undirected graph for ease of finding subgraphs
    std::vector<SimpleNodePtr> simple_nodes;
    CreateSimpleGraph(g, &simple_nodes);
#if 0
    const auto& indexed_graph = g.indexed_graph();
    for (size_t i = 0; i < indexed_graph.num_nodes(); ++i) {
      const nnvm::Node* node = indexed_graph[i].source;
      LOG(INFO) << node->attrs.name;
    }
#endif
    std::vector<std::vector<SimpleNode*>> subgraph_nodes;
    FindSubgraphs(g, *subg_prop, simple_nodes, &subgraph_nodes);
    for (size_t i = 0; i < subgraph_nodes.size(); ++i) {
#if 0
      // TODO(junwu): for debugging purpose, delete the following two lines
      std::set<SimpleNode*> simple_node_set(subgraph_nodes[i].begin(), subgraph_nodes[i].end());
      CHECK_EQ(simple_node_set.size(), subgraph_nodes[i].size());
#endif
      PrintSubgraph(subgraph_nodes[i]);
      CreateSubgraphNode(&g, simple_nodes, subgraph_nodes[i], i, &entry_top_order_map);
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
