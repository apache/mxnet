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
 * \file simple_partition_pass.h
 * \brief Simple pass for partitioning a graph.
 * \author Clement Fuji Tsang
 */
#ifndef MXNET_EXECUTOR_SIMPLE_PARTITION_PASS_H_
#define MXNET_EXECUTOR_SIMPLE_PARTITION_PASS_H_

#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator.h>
#include <nnvm/graph_attr_types.h>
#include <utility>
#include <deque>
#include <algorithm>
#include <vector>

#include "exec_pass.h"

namespace mxnet {
namespace exec {


/*!
 * \brief Custom graph class, which contains bi-directional nodes
 * required for traversing in both directions (from outputs to inputs
 * and vice versa). It is a non-owning layer on top of NNVM graph, since
 * NNVM graph enables traversing only in 1 direction (from outputs to inputs).
 */
class BidirectionalGraph {
 public:
  struct Node {
    nnvm::Node* nnvmptr;
    std::vector<Node*> inputs;
    std::vector<Node*> outputs;
  };

  explicit BidirectionalGraph(const Graph &g) {
    auto& idx = g.indexed_graph();
    auto num_nodes = idx.num_nodes();
    nodes.reserve(num_nodes);
    nnvm2nid.reserve(num_nodes);
    outputs.reserve(idx.outputs().size());
    // Create all the nodes in a new graph from
    // nodes in the NNVM graph and store them
    // in nodes array
    DFSVisit(g.outputs, [this](const nnvm::ObjectPtr& n) {
      Node new_node;
      new_node.nnvmptr = n.get();
      nnvm2nid[n.get()] = static_cast<uint32_t>(nodes.size());
      nodes.emplace_back(std::move(new_node));
    });
    // Create all connections between nodes in
    // the graph (both directions)
    for (const auto& it : nnvm2nid) {
      nnvm::Node* nnvmnode = it.first;
      uint32_t nid = it.second;
      for (auto& n : nnvmnode->inputs) {
        uint32_t input_nid = nnvm2nid[n.node.get()];
        nodes[input_nid].outputs.emplace_back(&nodes[nid]);
        nodes[nid].inputs.emplace_back(&nodes[input_nid]);
      }
    }
    // Create output connections from the graph
    for (auto& e : g.outputs) {
      uint32_t nid = nnvm2nid[e.node.get()];
      outputs.emplace_back(&nodes[nid]);
    }
  }

  /* \brief Get all subsets of nodes, where:
   *  - graph constructed from nodes in each subset is a connected graph
   *  - every node fulfills a predicate is_compatible
   *  - if nodes u and v are part of a subset, then for each path between
   *    u and v in the original directed graph, all nodes on those paths
   *    are also part of the subset
   * \param is_compatible A function taking nnvm::Node* and returning bool
   *                      which identifies which nodes should be included in
   *                      subsets.
   */
  template<typename FCompatible>
  std::vector<std::unordered_set<Node*>> get_subsets(FCompatible is_compatible) {
    std::vector<std::unordered_set<Node*>> subgraphs;
    std::unordered_set<Node*> incomp_set;
    std::vector<std::pair<bool, PairSet>> separation_sets;
    // Check each node for compatibility
    // and, if it is incompatible, mark nodes
    // on each side of it as not possible to be
    // in the same subset
    for (Node& node : nodes) {
      if (!is_compatible(node.nnvmptr)) {
        incomp_set.insert(&node);
      }
    }
    for (Node& node : nodes) {
      if (incomp_set.count(&node) != 0) {
        // Check if all your inputs are incompatible too.
        // If so, then your separation set does not matter,
        // because it will covered by the sets of your inputs
        bool inside_node = true;
        for (Node* input : node.inputs) {
          if (incomp_set.count(input) == 0) {
            inside_node = false;
          }
        }
        if (!inside_node) {
          std::unordered_set<Node*> in_graph;
          std::unordered_set<Node*> out_graph;
          std::vector<Node*> dummy_head;
          dummy_head.emplace_back(&node);
          DFS(dummy_head, false, [&out_graph](Node* node) {
              out_graph.insert(node);
          });
          DFS(dummy_head, true, [&in_graph](Node* node) {
              in_graph.insert(node);
          });
            separation_sets.push_back(std::make_pair(true,
                                                     std::make_pair(in_graph, out_graph)));
        } else {
          separation_sets.push_back(std::make_pair(false, PairSet()));
        }
      } else {
        separation_sets.push_back(std::make_pair(false, PairSet()));
      }
    }
    IncompMap incomp_map;
    // For each node construct the map of nodes that cannot be in
    // the same subset
    index_t num_nodes = nodes.size();
    for (index_t i = 0; i < num_nodes; ++i) {
      const auto n = &(nodes[i]);
      if (incomp_set.count(n) == 0) {
        for (index_t j = i + 1; j < num_nodes; ++j) {
          const auto& sep_set_pair = separation_sets[j];
          if (sep_set_pair.first && incomp_map[n].count(&nodes[j]) == 0) {
            const auto& p = sep_set_pair.second;
            if (p.first.count(n)) {
              incomp_map[n].insert(p.second.begin(), p.second.end());
            } else if (p.second.count(n)) {
              incomp_map[n].insert(p.first.begin(), p.first.end());
            }
          }
        }
        for (index_t j = i - 1; j >= 0; --j) {
          const auto& sep_set_pair = separation_sets[j];
          if (sep_set_pair.first && incomp_map[n].count(&nodes[j]) == 0) {
            const auto& p = sep_set_pair.second;
            if (p.first.count(n)) {
              incomp_map[n].insert(p.second.begin(), p.second.end());
            } else if (p.second.count(n)) {
              incomp_map[n].insert(p.first.begin(), p.first.end());
            }
          }
        }
        for (Node* incomp_n : incomp_set) {
          incomp_map[n].erase(incomp_n);
        }
      }
    }
    std::unordered_set<Node*> unused_set;

    for (auto& n : nodes) {
      if (incomp_set.count(&n) == 0) {
        unused_set.insert(&n);
      }
    }
    std::unordered_set<Node*> visited;
    std::deque<Node*> stack(outputs.begin(), outputs.end());
    // Create subsets
    while (!stack.empty()) {
      Node* vertex = stack.front();
      stack.pop_front();
      if (!visited.count(vertex)) {
        visited.insert(vertex);
        if (unused_set.count(vertex)) {
          subgraphs.emplace_back(naive_grow_subgraph(vertex, &unused_set, &incomp_map));
        }
        for (Node* input : vertex->inputs) {
          stack.emplace_back(input);
        }
      }
    }
    return subgraphs;
  }

 private:
  using PairSet = std::pair<std::unordered_set<Node*>, std::unordered_set<Node*>>;
  using PairVec = std::pair<std::vector<Node*>, std::vector<Node*>>;
  using IncompMap = std::unordered_map<Node*, std::unordered_set<Node*>>;

  /* \brief Traverse the graph using DFS in either direction.
   * \param heads Starting nodes for the DFS algorithm.
   * \param reverse If true, DFS will traverse the graph from
   *                outputs to inputs. Otherwise, it will
   *                traverse the graph from inputs to outputs.
   * \param fvisit Function to call on each visisted node.
   */
  template <typename FVisit>
  void DFS(const std::vector<Node*>& heads, bool reverse, FVisit fvisit) {
    std::unordered_set<Node*> visited;
    std::vector<Node*> vec(heads.begin(), heads.end());
    visited.reserve(heads.size());
    while (!vec.empty()) {
      Node* vertex = vec.back();
      vec.pop_back();
      if (visited.count(vertex) == 0) {
        visited.insert(vertex);
        fvisit(vertex);
        std::vector<Node*> nexts = reverse ? vertex->inputs : vertex->outputs;
        for (Node* node : nexts) {
          if (visited.count(node) == 0) {
            vec.emplace_back(node);
          }
        }
      }
    }
  }

  /* \brief Get the connected subgraph that contains the head node,
   *        only previously unused nodes, according to the rules
   *        from incompatibility map.
   * \param head Node which needs to be part of the returned subgraph.
   * \param unused_set Only nodes from this set will be considered when
   *                   adding to the growing subgraph.
   * \param incomp_map Map containing data on which nodes are incompatible
   *                   to be in the same subgraph.
   */
  std::unordered_set<Node*> naive_grow_subgraph(Node* head,
                                                std::unordered_set<Node*>* unused_set,
                                                IncompMap* incomp_map) {
    std::unordered_set<Node*> subgraph;
    std::unordered_set<Node*> incomp_set;
    std::deque<Node*> stack;
    stack.emplace_back(head);
    while (!stack.empty()) {
      Node* vertex = stack.back();
      stack.pop_back();
      if (unused_set->count(vertex) && !incomp_set.count(vertex)) {
        unused_set->erase(vertex);
        subgraph.insert(vertex);
        incomp_set.insert((*incomp_map)[vertex].begin(), (*incomp_map)[vertex].end());
        // Traverse the grpah in both directions
        for (Node* input : vertex->inputs) {
          if (unused_set->count(input) && !incomp_set.count(input)) {
            stack.emplace_back(input);
          }
        }
        for (Node* output : vertex->outputs) {
          if (unused_set->count(output) && !incomp_set.count(output)) {
            stack.emplace_back(output);
          }
        }
      }
    }
    return subgraph;
  }

  friend class Graph;

  std::vector<Node> nodes;
  std::unordered_map<nnvm::Node*, uint32_t> nnvm2nid;
  std::vector<Node*> outputs;
};  // class BidirectionalGraph

using NodeEntrySet = std::unordered_set<nnvm::NodeEntry, nnvm::NodeEntryHash,
                                        nnvm::NodeEntryEqual>;
using NodeRawPtrSet = std::unordered_set<nnvm::Node*>;

/*!
 * \brief Get the output nodes of the subgraph in the main graph.
 * \return a map between the node in the main graph and the output index of the subgraph node
*/
nnvm::NodeEntryMap<uint32_t> GetSubgraphOutputs(Graph g, NodeRawPtrSet subgraph_set) {
  nnvm::NodeEntryMap<uint32_t> outputs;
  uint32_t count = 0;
  for (auto& e : g.outputs) {
    if (subgraph_set.count(e.node.get()) && !outputs.count(e)) {
      outputs.insert({e, count++});
    }
  }
  DFSVisit(g.outputs, [&subgraph_set, &outputs, &count](const nnvm::ObjectPtr &node){
    if (!subgraph_set.count(node.get())) {
      for (auto& e : node->inputs) {
        if (subgraph_set.count(e.node.get()) && !outputs.count(e)) {
          outputs.insert({e, count++});
        }
      }
    }
  });
  return outputs;
}

/*!
 * \brief Create new input nodes of the subgraph and plug them.
 * \return the inputs of the subgraph node in the main graph
*/
std::vector<nnvm::NodeEntry> GetSubgraphInputs(Graph g, NodeRawPtrSet subgraph_set) {
  std::vector<nnvm::NodeEntry> inputs;
  nnvm::NodeEntryMap<nnvm::NodeEntry> entry_map;
  DFSVisit(g.outputs, [&subgraph_set, &inputs, &entry_map](const nnvm::ObjectPtr &node){
    if (subgraph_set.count(node.get())) {
      for (auto &e : node->inputs) {
        if (!subgraph_set.count(e.node.get())) {
          if (entry_map.count(e)) {
            e = entry_map[e];
          } else {
            auto new_node = nnvm::Node::Create();
            new_node->attrs.name = "input_" + std::to_string(inputs.size());
            entry_map.insert({e, nnvm::NodeEntry{new_node, 0, 0}});
            inputs.push_back(e);
            e.node = new_node;
            e.index = 0;
          }
        }
      }
    }
  });
  // Fix ordering of w.r.t to topology
  Graph _g;
  _g.outputs = g.outputs;
  const auto &idx = _g.indexed_graph();
  std::sort(inputs.begin(), inputs.end(),
      [&idx, &entry_map](const nnvm::NodeEntry lhs, const nnvm::NodeEntry rhs) {
        return idx.entry_id(entry_map.at(lhs)) < idx.entry_id(entry_map.at(rhs));
      });
  return inputs;
}

std::unordered_map<uint32_t, uint32_t> GetGraphInputsMap(const Graph& g) {
  std::unordered_map<uint32_t, uint32_t> outputs;
  auto& idx = g.indexed_graph();
  outputs.reserve(idx.num_nodes());
  std::vector<uint32_t> input_nodes = idx.input_nodes();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    outputs[input_nodes[i]] = static_cast<uint32_t>(i);
  }
  return outputs;
}

/*!
 * \brief Helper function to display what nodes are in a specific subset.
 */
void dispNodesSet(Graph g, NodeRawPtrSet s) {
  DFSVisit(g.outputs, [&s](const nnvm::ObjectPtr n){
    if (s.count(n.get())) {
      std::cout << "  Y " << n->attrs.name << std::endl;
    } else {
      std::cout << "  N " << n->attrs.name << std::endl;
    }
  });
}

/*!
 * \brief Replace a set of nodes by a subgraph node.
 */
template<typename FCreateNode>
Graph ReplaceSubgraphs(Graph&& g, const std::vector<NodeRawPtrSet>& subgraph_sets,
                       FCreateNode create_subgraph_node) {
  for (auto subgraph_set : subgraph_sets) {
    // Create MXNet subgraph
    Graph subgraph;
    const auto sub_outputs_in_main = GetSubgraphOutputs(g, subgraph_set);
    subgraph.outputs.resize(sub_outputs_in_main.size());
    for (auto p : sub_outputs_in_main) {
      subgraph.outputs[p.second] = p.first;
    }
    // To generate a subgraph an input has to be replaced by data node (no op)
    // and it has to be agnostic to the node from which it's an output
    // (For example, even if two inputs are two different outputs from the same node,
    // they need to be replaced by two completely separate data nodes)
    auto inputs = GetSubgraphInputs(subgraph, subgraph_set);
    auto subgraph_node = create_subgraph_node(subgraph);
    subgraph_node->inputs = inputs;
    // replug inputs of node out of subgraph to be output of the subgraph node
    // if it was a node in the subgraph
    DFSVisit(g.outputs,
        [&subgraph_node, &subgraph_set, &sub_outputs_in_main](const nnvm::ObjectPtr node) {
      if (!subgraph_set.count(node.get())) {
        for (auto &e : node->inputs) {
          auto it = sub_outputs_in_main.find(e);
          if (it != sub_outputs_in_main.end()) {
            e.node = subgraph_node;
            e.index = it->second;
          }
        }
      }
    });
    // replug outputs of the graph to be output of the subgraph node
    // if it was a node in the subgraph
    for (auto &e : g.outputs) {
      auto it = sub_outputs_in_main.find(e);
      if (it != sub_outputs_in_main.end()) {
        e.node = subgraph_node;
        e.index = it->second;
      }
    }
    // move control dependencies between nodes of the subgraph and out of the subgraph
    // to a dependencies between the subgraph node and the nodes out of the subgraph
    DFSVisit(g.outputs, [&subgraph_node, &subgraph_set](const nnvm::ObjectPtr& node) {
      for (auto &e : node->control_deps) {
        if (subgraph_set.count(e.get()))
          e = subgraph_node;
      }
    });
    DFSVisit(subgraph.outputs, [&subgraph_node, &subgraph_set](const nnvm::ObjectPtr& node) {
      auto it = node->control_deps.begin();
      while (it != node->control_deps.end()) {
        if (subgraph_set.count(it->get())) {
          ++it;
        } else {
          subgraph_node->control_deps.push_back(*it);
          it = node->control_deps.erase(it);
        }
      }
    });
  }
  Graph new_graph;
  new_graph.outputs = g.outputs;
  return new_graph;
}

/* \brief Get all subsets of nodes, where:
 *  - graph constructed from nodes in each subset is a connected graph
 *  - every node fulfills a predicate is_compatible
 *  - if nodes u and v are part of a subset, then for each path between
 *    u and v in the original directed graph, all nodes on those paths
 *    are also part of the subset
 * \param g NNVM graph
 * \param is_compatible A function taking nnvm::Node* and returning bool
 *                      which identifies which nodes should be included in
 *                      subsets.
 */
template<typename FCompatible>
std::vector<NodeRawPtrSet> GetCompatibleSubsets(const Graph& g, FCompatible is_compatible) {
  BidirectionalGraph biG = BidirectionalGraph(g);
  std::vector<std::unordered_set<BidirectionalGraph::Node*>> subsets =
    biG.get_subsets(is_compatible);
  std::vector<NodeRawPtrSet> nnvm_subsets;
  nnvm_subsets.reserve(subsets.size());
  for (auto& subset : subsets) {
    if (subset.size() > 1) {
      NodeRawPtrSet node_set;
      node_set.reserve(subset.size());
      for (auto& n : subset) {
        node_set.insert(n->nnvmptr);
      }
      nnvm_subsets.push_back(node_set);
    }
  }
  return nnvm_subsets;
}

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_EXECUTOR_SIMPLE_PARTITION_PASS_H_
