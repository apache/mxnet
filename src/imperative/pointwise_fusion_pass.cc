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
#include <chrono>
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
                   << "USE_CUDA=1.  Unset env var MXNET_USE_FUSION=1 "
                   << "to quiet this message.";
#endif  // defined(_WIN32)
  }
}

#if MXNET_USE_CUDA

namespace {

bool IsFusionCompatible(const nnvm::Node* n) {
  using namespace mxnet::fusion;
  if (n->op() == nullptr)
    return false;
  const std::string& op_name = n->op()->name;
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

bool IsInputsOnlyCompatible(const nnvm::Node* n) {
  using namespace mxnet::fusion;
  if (n->op() == nullptr)
    return false;
  const std::string& op_name = n->op()->name;
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

void CreateSubgraphNode(const nnvm::Graph& subgraph,
                        size_t inputs_size,
                        nnvm::Node* subgraph_node) {
  static const Op* fused_op_ptr = Op::Get("_FusedOp");
  subgraph_node->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>());
  subgraph_node->attrs.subgraphs.back()->outputs = subgraph.outputs;
  subgraph_node->attrs.dict["num_inputs"] = std::to_string(inputs_size);
  subgraph_node->attrs.dict["num_outputs"] = std::to_string(subgraph.outputs.size());
  subgraph_node->attrs.op = fused_op_ptr;
  subgraph_node->op()->attr_parser(&(subgraph_node->attrs));
}

struct EntryInfo {
  int source_node;
  int index;
};

inline int SetInsert(const EntryInfo& new_elem,
                     std::vector<EntryInfo>* elements) {
  for (size_t i = 0; i < elements->size(); ++i) {
    if ((new_elem.source_node == elements->at(i).source_node) &&
        (new_elem.index == elements->at(i).index)) {
      return i;
    }
  }
  elements->emplace_back(new_elem);
  return elements->size() - 1;
}

}  // namespace

/* \brief Create (if necessary) copy of the graph, replacing subgraphs with
 *        FusedOps. If there are no subgraphs to be replaced, the
 *        original graph is returned.
 * \param g original graph.
 * \param subgraph_assignment assignment of nodes in g's IndexedGraphs to
 *                            subgraphs. Values from -1 to num_subgraphs - 1
 *                            are allowed, -1 means that the node is not in a
 *                            subgraph.
 * \param num_subgraphs number of subgraphs.
 * \param create_subgraph_node function used to prepare the subgraph node.
 */
template<typename FCreateNode>
Graph CopyAndReplaceSubgraphs(const Graph& g,
                              const std::vector<int>& subgraph_assignment,
                              const int num_subgraphs,
                              FCreateNode create_subgraph_node) {
  if (num_subgraphs == 0) {
    return g;
  }

  Graph ret;

  const auto& idx = g.indexed_graph();

  CHECK_EQ(idx.num_nodes(), subgraph_assignment.size()) <<
    "Every node in the graph needs to be included in subgraph assignment.";

  std::vector<nnvm::ObjectPtr> new_nodes;
  new_nodes.reserve(idx.num_nodes());
  struct SubgraphInfo {
    nnvm::Graph graph;
    nnvm::ObjectPtr subgraph_node;
    std::vector<EntryInfo> outputs;
    std::vector<EntryInfo> inputs;
    std::vector<nnvm::ObjectPtr> input_nodes;
  };

  std::vector<SubgraphInfo> subgraphs(num_subgraphs);

  for (auto& info : subgraphs) {
    info.subgraph_node = nnvm::Node::Create();
  }

  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    // First copy the node, it will be used
    // either in the new graph or inside a
    // subgraph. Variables are not copied.
    if (idx[i].source->op() != nullptr) {
      new_nodes.emplace_back(nnvm::Node::Create());
      auto& node_copy = new_nodes.back();
      node_copy->attrs = idx[i].source->attrs;
      node_copy->info = idx[i].source->info;
    } else {
      new_nodes.emplace_back(idx[i].weak_ref.lock());
      continue;
    }
    auto& node_copy = new_nodes.back();
    const int subgraph_id = subgraph_assignment[i];
    if (subgraph_id != -1) {
      auto& info = subgraphs[subgraph_id];
      for (const auto& input : idx[i].inputs) {
        const int their_subgraph = subgraph_assignment[input.node_id];
        if (their_subgraph == subgraph_id) {
          node_copy->inputs.emplace_back(new_nodes[input.node_id],
                                         input.index,
                                         input.version);
        } else {
          int input_num;
          int output_num;
          if (their_subgraph == -1) {
            input_num = SetInsert({static_cast<int>(input.node_id),
                                   static_cast<int>(input.index)}, &(info.inputs));
          } else {
            auto& their_subgraph_info = subgraphs[their_subgraph];
            output_num = SetInsert({static_cast<int>(input.node_id),
                                    static_cast<int>(input.index)},
                                   &(their_subgraph_info.outputs));
            input_num = SetInsert({static_cast<int>(idx.num_nodes() + their_subgraph),
                                   output_num},
                                  &(info.inputs));
          }
          if (static_cast<size_t>(input_num) == info.input_nodes.size()) {
            info.input_nodes.emplace_back(nnvm::Node::Create());
            info.input_nodes.back()->attrs.name = "input_" + std::to_string(input_num);
            if (their_subgraph == -1) {
              info.subgraph_node->inputs.emplace_back(new_nodes[input.node_id],
                                                      input.index,
                                                      input.version);
            } else {
              info.subgraph_node->inputs.emplace_back(subgraphs[their_subgraph].subgraph_node,
                                                      output_num,
                                                      input.version);
            }
          }
          node_copy->inputs.emplace_back(info.input_nodes[input_num], 0, 0);
        }
      }
    } else {
      for (const auto& input : idx[i].inputs) {
        const int subgraph_id = subgraph_assignment[input.node_id];
        if (subgraph_id == -1) {
          node_copy->inputs.emplace_back(new_nodes[input.node_id],
                                         input.index,
                                         input.version);
        } else {
          auto& info = subgraphs[subgraph_id];
          const int output_num = SetInsert({static_cast<int>(input.node_id),
                                            static_cast<int>(input.index)},
                                           &(info.outputs));
          node_copy->inputs.emplace_back(info.subgraph_node,
                                         output_num,
                                         input.version);
        }
      }
    }

    // Control deps
    for (const auto& dep : idx[i].control_deps) {
      if (subgraph_id == subgraph_assignment[dep]) {
        node_copy->control_deps.emplace_back(new_nodes[dep]);
      }
    }
  }

  ret.outputs.reserve(idx.outputs().size());
  for (const auto& output : idx.outputs()) {
    const int subgraph_id = subgraph_assignment[output.node_id];
    if (subgraph_id == -1) {
      ret.outputs.emplace_back(new_nodes[output.node_id],
                               output.index,
                               output.version);
    } else {
      const int output_num = SetInsert({static_cast<int>(output.node_id),
                                        static_cast<int>(output.index)},
                                       &(subgraphs[subgraph_id].outputs));
      ret.outputs.emplace_back(subgraphs[subgraph_id].subgraph_node,
                               output_num,
                               output.version);
    }
  }

  for (auto& info : subgraphs) {
    info.graph.outputs.reserve(info.outputs.size());
    for (const auto& entry_info : info.outputs) {
      info.graph.outputs.emplace_back(new_nodes[entry_info.source_node],
                                      entry_info.index,
                                      0);
    }
    create_subgraph_node(info.graph, info.inputs.size(), info.subgraph_node.get());
  }

  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    // Add _FusedOpHelper nodes
    const int subgraph_id = subgraph_assignment[i];
    for (size_t dep_num = 0; dep_num < idx[i].control_deps.size(); ++dep_num) {
      const auto& dep = idx[i].control_deps[dep_num];
      const int their_subgraph_id = subgraph_assignment[dep];
      if (subgraph_id != -1 && their_subgraph_id == -1) {
        // Not in any subgraph, use FusedOpOutHelper
        auto& info = subgraphs[subgraph_id];
        size_t node_id = info.subgraph_node->control_deps.size();
        info.subgraph_node->control_deps.emplace_back(new_nodes[dep]);
        auto helper_node = op::MakeNode("_FusedOpOutHelper",
                                        "FusedOp_" + new_nodes[i]->attrs.name + "_outhelper",
                                        nullptr,
                                        nullptr,
                                        nullptr);
        helper_node->attrs.parsed =
          FusedOpHelperParamPtr(new FusedOpHelperParam(
                nnvm::get<FusedOpPtr>(info.subgraph_node->attrs.parsed),
                node_id));
        new_nodes[i]->control_deps.insert(new_nodes[i]->control_deps.begin() + dep_num,
                                          std::move(helper_node));
      } else if (their_subgraph_id != subgraph_id &&
                 their_subgraph_id != -1) {
        auto& info = subgraphs[their_subgraph_id];
        const auto& subgraph_idx = info.graph.indexed_graph();
        uint32_t node_id = subgraph_idx.node_id(new_nodes[dep].get());
        auto helper_node = op::MakeNode("_FusedOpHelper",
                                        info.subgraph_node->attrs.name + "_"
                                        + idx[i].source->attrs.name + "_helper",
                                        nullptr,
                                        nullptr,
                                        nullptr);
        helper_node->attrs.parsed =
          FusedOpHelperParamPtr(new FusedOpHelperParam(
                nnvm::get<FusedOpPtr>(info.subgraph_node->attrs.parsed),
                node_id));
        new_nodes[i]->control_deps.insert(new_nodes[i]->control_deps.begin() + dep_num,
                                          std::move(helper_node));
      }
    }
  }
  for (auto& info : subgraphs) {
    const auto& idx = info.graph.indexed_graph();
    const auto& input_nodes = idx.input_nodes();
    std::vector<nnvm::NodeEntry> subgraph_inputs;
    subgraph_inputs.reserve(info.subgraph_node->inputs.size());
    for (const int input : input_nodes) {
      for (size_t i = 0; i < info.input_nodes.size(); ++i) {
        const auto& input_ptr = info.input_nodes[i].get();
        if (input_ptr == idx[input].source) {
          subgraph_inputs.emplace_back(info.subgraph_node->inputs[i]);
        }
      }
    }
    info.subgraph_node->inputs.swap(subgraph_inputs);
    std::string name;
    for (size_t i = 0; i < idx.num_nodes(); ++i) {
      if (idx[i].source->op() != nullptr) {
        name += idx[i].source->op()->name + "_";
      }
    }
    info.subgraph_node->attrs.name = name;
  }
  return ret;
}

Graph FusePointwise(const Graph &g, const size_t num_forward_outputs) {
  auto start = std::chrono::steady_clock::now();
  auto [subset_assignment, num_subsets] = GetCompatibleSubsets(g, num_forward_outputs,  // NOLINT(*)
                                                               IsFusionCompatible,
                                                               IsInputsOnlyCompatible);
  Graph ret = CopyAndReplaceSubgraphs(g, subset_assignment, num_subsets,
                                      CreateSubgraphNode);
  auto end = std::chrono::steady_clock::now();
  if (dmlc::GetEnv("MXNET_RTC_VERBOSE", false)) {
    auto diff = end - start;
    LOG(INFO) << "Pointwise fusion graph pass took: "
              << std::chrono::duration<double, std::milli>(diff).count()
              << "ms.";
  }
  return ret;
}
#endif  // MXNET_USE_CUDA

}  // namespace exec
}  // namespace mxnet

