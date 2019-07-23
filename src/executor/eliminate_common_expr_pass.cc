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
using nnvm::NodePtr;
using nnvm::Graph;
using nnvm::IndexedGraph;

std::vector<std::pair<const Node*, uint32_t> >
ConvertInputs(const std::vector<nnvm::NodeEntry>& inputs) {
  std::vector<std::pair<const Node *, uint32_t> > ret;
  for (const auto& entry : inputs) {
    ret.emplace_back(entry.node.get(), entry.index);
  }
  return ret;
}

bool NodeEqual(const Node * n, const Node * m) {
  if (n->is_variable() || m->is_variable()) return false;
  if (n->op() != m->op()) return false;
  if (n->attrs.dict != m->attrs.dict) return false;

  // If an op is marked explicitly as having deterministic output
  static auto& deterministic_output =
    Op::GetAttr<FHasDeterministicOutput>("FHasDeterministicOutput");
  if (deterministic_output.get(n->op(), false)) return true;

  // Stateful ops cannot be be equal to each other
  static auto& fstateful = Op::GetAttr<FCreateOpState>("FCreateOpState");
  if (fstateful.get(n->op(), nullptr) != nullptr) return false;

  // Ops that require resource could ask for
  // random resource, so need to be explicitly marked
  // to be eligible
  static auto& resource_request = Op::GetAttr<FResourceRequest>("FResourceRequest");
  static auto& resource_request_ex = Op::GetAttr<FResourceRequestEx>("FResourceRequestEx");
  if (resource_request.get(n->op(), nullptr) != nullptr) return false;
  if (resource_request_ex.get(n->op(), nullptr) != nullptr) return false;

  // Ops that mutate inputs cannot be optimized out
  static auto& fmutate_inputs = Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  if (fmutate_inputs.get(n->op(), nullptr) != nullptr) return false;

  return true;
}

std::vector<std::pair<NodePtr, NodePtr> > GetCommonNodes(const Graph& g) {
  std::vector<std::pair<NodePtr, NodePtr> > ret;
  std::map<std::vector<std::pair<const Node*, uint32_t> >,
                       std::vector<const NodePtr*> > grouped_nodes;
  nnvm::DFSVisit(g.outputs, [&grouped_nodes](const NodePtr& n) {
    if (n->inputs.size() != 0) {
      grouped_nodes[ConvertInputs(n->inputs)].push_back(&n);
    }
  });
  // Check for common nodes
  for (const auto& pair : grouped_nodes) {
    if (pair.second.size() > 1) {
      std::unordered_set<size_t> visited;
      for (size_t i = 0; i < pair.second.size(); ++i) {
        if (visited.count(i)) continue;
        for (size_t j = i + 1; j < pair.second.size(); ++j) {
          if (NodeEqual(pair.second[i]->get(), pair.second[j]->get())) {
            visited.insert(j);
            NodePtr src = *pair.second[i];
            NodePtr replaced = *pair.second[j];
            ret.emplace_back(src, replaced);
          }
        }
      }
    }
  }
  return ret;
}

void EliminateCommonNodes(Graph * g,
                          const std::vector<std::pair<NodePtr, NodePtr> >& common_nodes) {
  for (const auto& p : common_nodes) {
    std::vector<NodePtr> nodes_to_change;
    const NodePtr& src = p.first;
    const NodePtr& replaced = p.second;
    DFSVisit(g->outputs, [replaced, &nodes_to_change](const NodePtr& n) {
      for (const auto& dep : n->control_deps) {
       if (dep == replaced) {
         nodes_to_change.push_back(n);
         return;
       }
      }
      for (const auto& inp : n->inputs) {
        if (inp.node == replaced) {
          nodes_to_change.push_back(n);
          return;
        }
      }
    });

    for (auto& n : nodes_to_change) {
      for (auto& dep : n->control_deps) {
       if (dep == replaced) {
         dep = src;
       }
      }
      for (auto& inp : n->inputs) {
        if (inp.node == replaced) {
          inp.node = src;
        }
      }
    }

    for (const auto& n : replaced->control_deps) {
      src->control_deps.push_back(n);
    }

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
      NodePtr copy_node = Node::Create();
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

nnvm::Graph EliminateCommonExpr(nnvm::Graph&& g) {
  using nnvm::NodePtr;
  bool keep_running = true;
  while (keep_running) {
    const std::vector<std::pair<NodePtr, NodePtr> >& common_nodes = GetCommonNodes(g);
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
