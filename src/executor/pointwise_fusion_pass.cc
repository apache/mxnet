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
 * \brief
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/pass_functions.h>
#include "./simple_partition_pass.h"
#include "../operator/fusion/fused_op-inl.h"
#include "../operator/fusion/fused_op.h"
#include "../operator/operator_common.h"

namespace mxnet {
namespace exec {
namespace {
  bool IsFusionCompatible(nnvm::Node* n) {
    using namespace mxnet::detail;
    if (n->op() == nullptr)
      return false;
    std::string op_name = n->op()->name;
    if (fused_op_binary_ops.count(op_name))
      return true;
    if (fused_op_unary_ops.count(op_name))
      return true;
    if (fused_op_special_ops.count(op_name))
      return true;
    if (fused_op_mimo_ops.count(op_name))
      return true;
    return false;
  }
  
  nnvm::NodePtr CreateSubgraphNode(const Graph& subgraph) {
    nnvm::Symbol subgraph_sym;
    auto node = nnvm::Node::Create();
    subgraph_sym.outputs = subgraph.outputs;
    node->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(subgraph_sym));
    std::ostringstream name_oss, params_oss;
    // the name of the new node will be the concatenation of all the node names in the subgraph
    DFSVisit(subgraph.outputs, [&name_oss](const nnvm::NodePtr n) {
      if (n->op() != nullptr)
        name_oss << n->attrs.name << "_";
    });
    auto subgraph_name = name_oss.str();
    subgraph_name.pop_back();
    node->attrs.name = subgraph_name;
    // in case the subgraph contains some of the weights
    for (auto &e : subgraph_sym.ListInputNames(nnvm::Symbol::kAll)) {
      params_oss << e << ";";
    }
    auto params_names = params_oss.str();
    params_names.pop_back();
    //node->attrs.dict["subgraph_params_names"] = params_names;
    node->attrs.dict["symbol_json"] = nnvm::pass::SaveJSON(subgraph);
    node->attrs.dict["num_inputs"] =
        std::to_string(subgraph.indexed_graph().input_nodes().size());
    node->attrs.dict["num_outputs"] = std::to_string(subgraph.outputs.size());
    node->attrs.op = Op::Get("FusedOp");
    node->op()->attr_parser(&(node->attrs));
    return node;
  }
}

/*!
 * \brief Replace a set of nodes by a subgraph node
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
    // To generate a subgraph an input have to be replace by data node (no op)
    // and it have to be agnostic to the node from which it's an output
    // (For exemple even if two inputs are two different outputs from the same node)
    auto inputs = GetSubgraphInputs(subgraph, subgraph_set);
    auto subgraph_node = create_subgraph_node(subgraph);
    subgraph_node->inputs = inputs;
    // replug inputs of node out of subgraph to be output of the subgraph node
    // if it was a node in the subgraph
    DFSVisit(g.outputs,
        [&subgraph_node, &subgraph_set, &sub_outputs_in_main](const nnvm::NodePtr node) {
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
    const auto& index = g.indexed_graph();
    DFSVisit(g.outputs, [&subgraph_node, &subgraph_set, &index](const nnvm::NodePtr& node) {
      for (auto &e : node->control_deps) {
        if (subgraph_set.count(e.get())) {
          uint32_t node_id = index.node_id(e.get());
          auto helper_node = op::MakeNode("_FusedOpHelper",
                                          subgraph_node->attrs.name + "_helper",
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
    DFSVisit(subgraph.outputs, [&subgraph_node, &subgraph_set](const nnvm::NodePtr& node) {
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

Graph FusePointwise(Graph &&g) {
  const auto & num_forward_output = g.GetAttr<size_t>("num_forward_outputs");
  Graph fg;
  fg.outputs.insert(fg.outputs.begin(), g.outputs.begin(),
                    g.outputs.begin() + num_forward_output);
  auto subsets = GetCompatibleSubsets(fg, IsFusionCompatible);
  g = ReplaceSubgraphsPointwise(std::move(g), subsets, CreateSubgraphNode);

  return g;
}

}
}
