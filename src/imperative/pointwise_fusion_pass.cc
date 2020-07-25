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
 * \file pointwise_fusion_pass.cc
 * \brief Pass applying pointwise fusion.
 * \author Clement Fuji Tsang
 */

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <queue>
#include "./simple_partition_pass.h"
#include "../operator/fusion/fused_op-inl.h"
#include "../operator/fusion/fused_op.h"
#include "../operator/operator_common.h"

namespace mxnet {
namespace exec {

void WarnFusionNotSupported() {
  static bool issued_warning = false;
  if (!issued_warning) {
    issued_warning = true;
#if defined(_WIN32)
    LOG(WARNING) << "Omitting dynamic fused op creation- not enabled on Windows.  "
                 << "Unset env var MXNET_USE_FUSION=1 to quiet this message.";
#else
    LOG(WARNING) << "Omitting dynamic fused op creation- needs MXNet lib built with "
                   << "USE_CUDA=1 and ENABLE_CUDA_RTC=1.  Unset env var MXNET_USE_FUSION=1 "
                   << "to quiet this message.";
#endif  // defined(_WIN32)
  }
}

#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC

namespace {
  bool IsFusionCompatible(nnvm::Node* n) {
    using namespace mxnet::fusion;
    if (n->op() == nullptr)
      return false;
    std::string op_name = n->op()->name;
    if (ops_desc.count(op_name))
      return true;
    if (slice_ops.count(op_name))
      return false;
    if (std::find(variable_io_ops.begin(),
                  variable_io_ops.end(),
                  op_name) !=
        variable_io_ops.end())
      return true;
    if (op_name == "LeakyReLU") {
        std::string act_type = n->attrs.dict.at("act_type");
        if (LeakyReLU_ops.count(act_type))
          return true;
        else
          return false;
    }
    if (op_name == "_backward_LeakyReLU") {
        std::string act_type = n->attrs.dict.at("act_type");
        if (LeakyReLU_bwd_ops.count(act_type))
          return true;
        else
          return false;
    }
    return false;
  }

  bool IsInputsOnlyCompatible(nnvm::Node* n) {
    using namespace mxnet::fusion;
    if (n->op() == nullptr)
      return false;
    std::string op_name = n->op()->name;
    if (slice_ops.count(op_name)) {
      if (op_name == "slice") {
        // slice with non-default step attribute is not supported
        // currently
        if (n->attrs.dict.count("step") &&
            !(n->attrs.dict.at("step") == "()" ||
              n->attrs.dict.at("step") == "[]")) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const Graph& subgraph, size_t inputs_size) {
    nnvm::Symbol subgraph_sym;
    auto node = nnvm::Node::Create();
    subgraph_sym.outputs = subgraph.outputs;
    node->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(subgraph_sym));
    node->attrs.name = "FusedOp";
    node->attrs.dict["num_inputs"] = std::to_string(inputs_size);
    node->attrs.dict["num_outputs"] = std::to_string(subgraph.outputs.size());
    node->attrs.op = Op::Get("_FusedOp");
    node->op()->attr_parser(&(node->attrs));
    return node;
  }
}  // namespace

/*!
 * \brief Replace a set of nodes by a subgraph node.
 *        This function is used specifically in pointwise fusion.
 */
template<typename FCreateNode>
Graph ReplaceSubgraphsPointwise(Graph&& g, const std::vector<NodeRawPtrSet>& subgraph_sets,
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
    auto subgraph_node = create_subgraph_node(subgraph, inputs.size());
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
    DFSVisit(subgraph.outputs, [&subgraph_node, &subgraph_set](const nnvm::ObjectPtr& node) {
      if (subgraph_set.count(node.get())) {
        auto it = node->control_deps.begin();
        static auto& is_fusion = Op::GetAttr<exec::TIsFusionHelper>("TIsFusionHelper");
        std::vector<nnvm::ObjectPtr> new_control_deps;
        // Use the first control dependency to get the inferattr helper
        if (it != node->control_deps.end()) {
          if (subgraph_set.count(it->get())) {
            new_control_deps.push_back(*it);
          } else {
            if ((*it)->is_variable() || !is_fusion.get((*it)->op(), false)) {
              uint32_t node_id = subgraph_node->control_deps.size();
              subgraph_node->control_deps.push_back(*it);
              auto helper_node = op::MakeNode("_FusedOpOutHelper",
                                              "FusedOp_" + node->attrs.name + "_outhelper",
                                              nullptr,
                                              nullptr,
                                              nullptr);
              helper_node->attrs.parsed =
                FusedOpHelperParamPtr(new FusedOpHelperParam(
                      nnvm::get<FusedOpPtr>(subgraph_node->attrs.parsed),
                      node_id));
              new_control_deps.push_back(helper_node);
            } else {
              new_control_deps.push_back(*it);
            }
          }
          ++it;
        }
        node->control_deps = new_control_deps;
      }
    });

    std::ostringstream name_oss;
    // the name of the new node will be the concatenation of all the node names in the subgraph
    DFSVisit(subgraph.outputs, [&name_oss](const nnvm::ObjectPtr n) {
      if (n->op() != nullptr) {
        name_oss << n->op()->name << "_";
      }
    });
    auto subgraph_name = name_oss.str();
    subgraph_name.pop_back();
    subgraph_node->attrs.name = subgraph_name;

    const auto& index = subgraph.indexed_graph();
    DFSVisit(g.outputs, [&subgraph_node, &subgraph_set, &index](const nnvm::ObjectPtr& node) {
      for (auto &e : node->control_deps) {
        if (subgraph_set.count(e.get())) {
          uint32_t node_id = index.node_id(e.get());
          auto helper_node = op::MakeNode("_FusedOpHelper",
                                          subgraph_node->attrs.name + "_"
                                          + node->attrs.name + "_helper",
                                          nullptr,
                                          nullptr,
                                          nullptr);
          helper_node->attrs.parsed =
            FusedOpHelperParamPtr(new FusedOpHelperParam(
                  nnvm::get<FusedOpPtr>(subgraph_node->attrs.parsed),
                  node_id));
          e = helper_node;
        }
      }
    });
  }
  Graph new_graph;
  new_graph.outputs = g.outputs;
  return new_graph;
}

/* \brief Add nodes as inputs to the subgraph. This is used for operations
 *        which are only compatible when they are the first nodes in the
 *        subgraph.
 */
template <typename IsCompatible>
void AddInputsOnlyCompatible(const Graph &g,
                             std::vector<std::unordered_set<nnvm::Node*> >* subsets,
                             IsCompatible is_compatible) {
  std::unordered_map<nnvm::Node*, uint32_t> node2setidx;
  size_t subgraphs_fullsize = 0;
  for (auto& s : *subsets) {
    subgraphs_fullsize += s.size();
  }
  node2setidx.reserve(subgraphs_fullsize);
  for (size_t i = 0; i < subsets->size(); ++i) {
    for (auto& n : (*subsets)[i]) {
      node2setidx.insert({n, i});
    }
  }
  std::vector<std::vector<nnvm::Node*> > to_add(subsets->size());
  DFSVisit(g.outputs, [&is_compatible, &node2setidx, &to_add](const nnvm::ObjectPtr& n) {
    const auto& it = node2setidx.find(n.get());
    if (it != node2setidx.end()) {
      for (auto& e : n->inputs) {
        if (is_compatible(e.node.get()))
          to_add[it->second].push_back(e.node.get());
      }
    }
  });

  // Avoid duplicating the node that is input of two subsets
  std::unordered_set<nnvm::Node*> added;
  for (size_t i = 0; i < subsets->size(); ++i) {
    std::vector<nnvm::NodeEntry> heads;
    for (auto n : subsets->at(i)) {
      for (auto e : n->inputs) {
        if (!subsets->at(i).count(e.node.get()))
          heads.push_back(e);
      }
    }
    for (size_t j = 0; j < to_add[i].size(); ++j) {
      if (!added.count(to_add[i][j])) {
        bool make_cycle = false;
        const auto& node = to_add[i][j];
        std::vector<nnvm::NodeEntry> _heads;
        std::copy_if(heads.begin(), heads.end(), std::back_inserter(_heads),
                     [&node](const nnvm::NodeEntry& n) {
                       return n.node.get() != node;
                     });
        DFSVisit(_heads, [&make_cycle, &node](const nnvm::ObjectPtr& n) {
          if (n.get() == node)
            make_cycle = true;
        });
        if (!make_cycle) {
          (*subsets)[i].insert(to_add[i][j]);
          added.insert(to_add[i][j]);
        }
      }
    }
  }
}

Graph FusePointwiseForward(Graph &&g) {
  Graph ret;
  g.indexed_graph();
  const auto& num_forward_outputs = g.GetAttr<size_t>("num_forward_outputs");
  Graph fg;
  fg.outputs.insert(fg.outputs.begin(), g.outputs.begin(),
                    g.outputs.begin() + num_forward_outputs);
  auto subsets = GetCompatibleSubsets(fg, IsFusionCompatible);
  AddInputsOnlyCompatible(fg, &subsets, IsInputsOnlyCompatible);
  g = ReplaceSubgraphsPointwise(std::move(g), subsets, CreateSubgraphNode);
  ret.outputs = g.outputs;
  return ret;
}

Graph FusePointwiseBackward(Graph &&g) {
  Graph ret;
  g.indexed_graph();
  const auto& num_forward_outputs = g.GetAttr<size_t>("num_forward_outputs");
  Graph fg;
  fg.outputs.insert(fg.outputs.begin(), g.outputs.begin(),
                    g.outputs.begin() + num_forward_outputs);
  std::unordered_set<nnvm::Node*> exclusion_set;
  DFSVisit(fg.outputs, [&exclusion_set](const nnvm::ObjectPtr& n) {
    exclusion_set.insert(n.get());
  });
  auto subsets = GetCompatibleSubsets(g, [&exclusion_set](nnvm::Node* n) {
    if (exclusion_set.count(n))
      return false;
    return IsFusionCompatible(n);
  });
  g = ReplaceSubgraphsPointwise(std::move(g), subsets, CreateSubgraphNode);
  ret.outputs = g.outputs;
  return ret;
}
#endif  // MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC

}  // namespace exec
}  // namespace mxnet

