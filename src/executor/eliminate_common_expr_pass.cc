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
 * \file eliminate_common_expr.cc
 * \brief Eliminate common expressions in the graph
 * \author Przemyslaw Tredak
 */

#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>

#include <vector>
#include <map>
#include <utility>
#include <sstream>

namespace mxnet {
namespace exec {

namespace {

using nnvm::Node;
using nnvm::ObjectPtr;
using nnvm::Graph;
using nnvm::IndexedGraph;

// NodeInput holds the sufficient subset of NodeEntry fields for Node-input equality tests
using NodeInput = std::pair<const Node*, uint32_t>;

/*!
 * \brief Convert a Node's input vector of `NodeEntry` to a vector of the simpler `NodeInput`
 */
std::vector<NodeInput> ConvertInputs(const std::vector<nnvm::NodeEntry>& inputs) {
  std::vector<NodeInput> ret;
  for (const auto& entry : inputs) {
    ret.emplace_back(entry.node.get(), entry.index);
  }
  return ret;
}

/*!
 * \brief Determine if two Nodes have equal function such that one Node can be eliminated.
 */
bool NodeEqual(const Node* n, const Node* m) {
  if (n->is_variable() || m->is_variable()) return false;
  if (n->op() != m->op()) return false;
  // Nodes with different attributes are considered not identical,
  // though this may reject Node pairs that are in fact functionally the same.
  if (n->attrs.dict != m->attrs.dict) return false;

  // Ops that mutate inputs cannot be optimized out
  static auto& fmutate_inputs = Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  if (fmutate_inputs.get(n->op(), nullptr) != nullptr) return false;

  // Stateful ops cannot be be equal to each other
  static auto& fstateful = Op::GetAttr<FCreateOpState>("FCreateOpState");
  if (fstateful.get(n->op(), nullptr) != nullptr)
    return false;

  // Check to see if the user has explicitly set THasDeterministicOutput to override the
  // subsequent determination of Node equality based on resource use.
  static auto& deterministic_output =
      Op::GetAttr<THasDeterministicOutput>("THasDeterministicOutput");
  if (deterministic_output.contains(n->op()))
    return deterministic_output[n->op()];

  // Ops that require resource could ask for
  // random resource, so need to be explicitly marked
  // to be eligible
  static auto& resource_request = Op::GetAttr<FResourceRequest>("FResourceRequest");
  static auto& resource_request_ex = Op::GetAttr<FResourceRequestEx>("FResourceRequestEx");
  if (resource_request.get(n->op(), nullptr) != nullptr) return false;
  if (resource_request_ex.get(n->op(), nullptr) != nullptr) return false;

  return true;
}

// Graph traversal to create a list of pairs of identical-function nodes that can be combined.
std::vector<std::pair<ObjectPtr, ObjectPtr> > GetCommonNodes(const Graph& g) {
  std::vector<std::pair<ObjectPtr, ObjectPtr> > ret;
  // A map between a vector of inputs and those nodes that have those inputs
  std::map<std::vector<NodeInput>, std::vector<const ObjectPtr*> > grouped_nodes;
  // Traverse the graph and group the nodes by their vector of inputs
  nnvm::DFSVisit(g.outputs, [&grouped_nodes](const ObjectPtr& n) {
    if (n->inputs.size() != 0) {
      grouped_nodes[ConvertInputs(n->inputs)].push_back(&n);
    }
  });
  // Now check for identical node ops within the node groups (having identical inputs)
  for (const auto& pair : grouped_nodes) {
    auto &node_group = pair.second;  // Group of nodes that share the same vector of inputs
    if (node_group.size() > 1) {
      std::unordered_set<size_t> visited;
      for (size_t i = 0; i < node_group.size(); ++i) {
        if (visited.count(i)) continue;
        for (size_t j = i + 1; j < node_group.size(); ++j) {
          // If the two Nodes have equal function, then one Node (called the 'replaced') can
          // be eliminated in favor of the other Node (the 'src').
          if (NodeEqual(node_group[i]->get(), node_group[j]->get())) {
            visited.insert(j);
            ObjectPtr src = *node_group[i];
            ObjectPtr replaced = *node_group[j];
            ret.emplace_back(src, replaced);
          }
        }
      }
    }
  }
  return ret;
}

/*!
 * \brief Do a single pass of Node elimination given pairs of identical Nodes.
 */
void EliminateCommonNodes(Graph* g,
                          const std::vector<std::pair<ObjectPtr, ObjectPtr> >& common_nodes) {
  for (const auto &p : common_nodes) {
    std::vector <ObjectPtr> nodes_to_change;
    const ObjectPtr &src = p.first;
    const ObjectPtr &replaced = p.second;
    // Create a `nodes_to_change` list containing the Nodes that refer to the `replaced` Node
    // that is targeted for elimination.
    DFSVisit(g->outputs, [replaced, &nodes_to_change](const ObjectPtr &n) {
      for (const auto &dep : n->control_deps) {
        if (dep == replaced) {
          nodes_to_change.push_back(n);
          return;
        }
      }
      for (const auto &inp : n->inputs) {
        if (inp.node == replaced) {
          nodes_to_change.push_back(n);
          return;
        }
      }
    });

    // Change references to the `replaced` Node within the `nodes_to_change` list to be
    // references to the equivalent `src` Node.
    for (auto &n : nodes_to_change) {
      for (auto &dep : n->control_deps) {
        if (dep == replaced) {
          dep = src;
        }
      }
      for (auto &inp : n->inputs) {
        if (inp.node == replaced) {
          inp.node = src;
        }
      }
    }

    // Add `replaced` Node control dependencies to those of the `src` Node.
    for (const auto &n : replaced->control_deps) {
      src->control_deps.push_back(n);
    }

    // Change graph outputs driven by the `replaced` Node to now point to the `src` Node.
    for (auto& out : g->outputs) {
      if (out.node == replaced) {
        out.node = src;
      }
    }
  }
  // Check for duplicates in outputs and
  // insert Copy nodes as appropriate
  const Op* copy_op = Op::Get("_copy");
  nnvm::NodeEntryMap<size_t> unique_outputs;
  for (size_t i = 0; i < g->outputs.size(); ++i) {
    auto kv = unique_outputs.find(g->outputs[i]);
    if (kv == unique_outputs.end()) {
      unique_outputs.emplace(g->outputs[i], 0);
    } else {
      ObjectPtr copy_node = Node::Create();
      std::ostringstream os;
      os << kv->first.node->attrs.name << "_" << kv->second << "_copy";
      kv->second++;
      copy_node->attrs.op = copy_op;
      copy_node->attrs.name = os.str();
      copy_node->inputs.emplace_back(kv->first);
      g->outputs[i] = nnvm::NodeEntry{copy_node, 0, 0};
    }
  }
}

}  // namespace

/*!
 * \brief Simplify a graph by iteratively eliminating Nodes with identical inputs and function.
 */
nnvm::Graph EliminateCommonExpr(nnvm::Graph&& g) {
  using nnvm::ObjectPtr;
  bool keep_running = true;
  while (keep_running) {
    const auto& common_nodes = GetCommonNodes(g);
    if (common_nodes.empty()) {
      keep_running = false;
    } else {
      EliminateCommonNodes(&g, common_nodes);
    }
  }
  return g;
}

}  // namespace exec
}  // namespace mxnet
