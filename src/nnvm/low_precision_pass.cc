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
 * \file low_precision_pass.cc
 * \brief Return new graph with amp_cast and amp_multicast operators added wherever required
 */

#include <nnvm/node.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <mxnet/base.h>
#include <algorithm>
#include <functional>

namespace mxnet {
using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::Graph;

// create a node for operator : op_name with name : node_name
static NodePtr CreateNode(std::string op_name, std::string node_name) {
  NodePtr node = Node::Create();
  node->attrs.name = node_name;
  if (op_name == "nullptr") {
    node->attrs.op = nullptr;
    // ugly workaround because VariableParam is not exposed
    node->attrs.parsed = nnvm::Symbol::CreateVariable(node->attrs.name)
                             .outputs[0]
                             .node->attrs.parsed;
  } else {
    node->attrs.op = Op::Get(op_name);
  }
  return node;
}

static NodePtr InsertNode(std::string op_name, std::string node_name, NodePtr current,
                          NodeEntry previous) {
    NodePtr node = CreateNode(op_name, node_name);
    node->inputs.emplace_back(previous);
    current->inputs.emplace_back(NodeEntry{node, 0, 0});
    return node;
}

// get suffix for a node entry so that it can be used for amp_cast/amp_multicast node name
static std::string GetSuffix(const nnvm::NodeEntry &node_entry,
                             const std::unordered_map<Node*, NodePtr> &mirror_map) {
  static const auto &flist_outputs =
      nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  std::string suffix = "";
  NodePtr mirror_node = mirror_map.at(node_entry.node.get());
  if (mirror_node->op() != nullptr) {
      auto list_output_names_func = flist_outputs.get(node_entry.node->op(), nullptr);
      if (list_output_names_func != nullptr) {
          std::vector<std::string> names = list_output_names_func(node_entry.node->attrs);
          suffix = "_" + names[node_entry.index];
      } else {
          suffix = "_" + std::to_string(node_entry.index);
      }
  }
  return suffix;
}

// add amp_cast node between curr_node and input
static void AddCastNode(const nnvm::NodeEntry &e, const std::string &suffix,
                        const nnvm::NodeEntry &input, const std::string dtype,
                        nnvm::NodeEntryMap<NodeEntry> *mirror_entry_map,
                        NodePtr curr_node) {
  NodePtr cast_node =
      InsertNode("amp_cast", e.node->attrs.name + suffix + "_amp_cast_" + dtype,
                 curr_node, input);
  cast_node->attrs.dict["dtype"] = dtype;
  cast_node->op()->attr_parser(&(cast_node->attrs));
  (*mirror_entry_map)[e] = NodeEntry{std::move(cast_node), 0, e.version};
  return;
}

// add amp_multicast node between curr_node and inputs
static void AddMultiCastNode(const std::vector<NodeEntry> &inputs,
                             const std::string &node_name,
                             const std::unordered_map<Node *, NodePtr> &mirror_map,
                             NodePtr curr_node) {
  NodePtr node =
      CreateNode("amp_multicast",
                 inputs[0].node->attrs.name + node_name + "_amp_multicast");
  for (const auto &node_entry : inputs) {
    NodePtr mirror_node = mirror_map.at(node_entry.node.get());
    NodeEntry mirror_entry = NodeEntry{std::move(mirror_node), node_entry.index,
                                       node_entry.version};
    node->inputs.emplace_back(mirror_entry);
  }
  node->attrs.dict["num_outputs"] = std::to_string(inputs.size());
  node->op()->attr_parser(&(node->attrs));
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    const auto &e = inputs[i];
    curr_node->inputs.emplace_back(
        NodeEntry{node, static_cast<uint32_t>(i), e.version});
  }
  return;
}

static bool CheckConditionalFP32(
    const std::unordered_map<
        std::string, std::unordered_map<std::string, std::vector<std::string>>>
        &conditional_fp32_ops,
    const std::unordered_set<std::string> &excluded_syms, NodePtr node) {
  if (node->is_variable() || (excluded_syms.count(node->attrs.name) > 0) ||
      conditional_fp32_ops.count(node->op()->name) == 0) {
    return false;
  } else {
    // Iterate through all conditional ops
    auto it = conditional_fp32_ops.find(node->op()->name);
    if (it != conditional_fp32_ops.end()) {
      auto it_params = it->second;
      // For each param name, iterate through param values to check
      // if the provided param name is equal to any of the values
      for (auto it_param = it_params.begin(); it_param != it_params.end();
           it_param++) {
        auto param_key = node->attrs.dict.find(it_param->first);
        if (param_key != node->attrs.dict.end()) {
          auto it_param_vals = it_param->second;
          if (std::find(it_param_vals.begin(), it_param_vals.end(),
                        param_key->second) != it_param_vals.end()) {
            return true;
          }
        }
      }
    }
    return false;
  }
}

Graph ReducePrecision(Graph &&src) {
  const auto target_dtype_ops =
      src.GetAttr<std::unordered_set<std::string>>("target_dtype_ops");
  const auto fp32_ops =
      src.GetAttr<std::unordered_set<std::string>>("fp32_ops");
  const auto widest_dtype_ops =
      src.GetAttr<std::unordered_set<std::string>>("widest_dtype_ops");
  const auto target_dtype = src.GetAttr<int>("target_dtype");
  const auto excluded_syms = src.GetAttr<std::unordered_set<std::string>>("excluded_syms");
  const auto conditional_fp32_ops = src.GetAttr<std::unordered_map<
      std::string, std::unordered_map<std::string, std::vector<std::string>>>>(
      "conditional_fp32_ops");

  CHECK(target_dtype == mshadow::kFloat16)
      << "Only float16 target_dtype is supported yet";

  // Additional data structures to share common cast node inputs among different nodes
  std::unordered_map<Node *, NodePtr> mirror_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_fp32_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_target_dtype_map;

  // Visit nodes in a topologically sorted order
  DFSVisit(src.outputs, [&](const NodePtr &node) {
    NodePtr new_node = Node::Create(*node);
    new_node->inputs.clear();

    /* 1. for node which needs to run in FP32 mode, add amp_cast operators
     * (to fp32) after its inputs
     * 2. for node which needs to run in FP16 mode, add amp_cast operators
     * (to target_dtype) after its inputs
     * 3. for nodes which need to run in widest dtype among its inputs, add
     * amp_multicast operators between op and its inputs
     * 4. for nodes which need to run in FP32 mode, based on a specific condition,
     * check the condition, and if true add amp_cast (to fp32) after its inputs
     * 4. for other nodes, create copy node and add it to the mirror_map
     */
    if (!node->is_variable() && fp32_ops.count(node->op()->name) > 0 &&
        excluded_syms.count(node->attrs.name) == 0) {
      for (const auto& node_entry : node->inputs) {
        if (mirror_fp32_map.count(node_entry)) {
          new_node->inputs.emplace_back(mirror_fp32_map[node_entry]);
        } else {
          NodePtr mirror_node = mirror_map.at(node_entry.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
          std::string suffix = GetSuffix(node_entry, mirror_map);
          AddCastNode(node_entry, suffix, mirror_entry, "float32", &mirror_fp32_map,
                      new_node);
        }
      }
    } else if (!node->is_variable() &&
               target_dtype_ops.count(node->op()->name) > 0 &&
               excluded_syms.count(node->attrs.name) == 0) {
      for (const auto& node_entry : node->inputs) {
        if (mirror_target_dtype_map.count(node_entry)) {
          new_node->inputs.emplace_back(mirror_target_dtype_map[node_entry]);
        } else {
          NodePtr mirror_node = mirror_map.at(node_entry.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
          std::string suffix = GetSuffix(node_entry, mirror_map);
          AddCastNode(node_entry, suffix, mirror_entry, "float16",
                      &mirror_target_dtype_map, new_node);
        }
      }
    } else if (!node->is_variable() &&
               widest_dtype_ops.count(node->op()->name) > 0 &&
               excluded_syms.count(node->attrs.name) == 0) {
      CHECK(node->inputs.size() > 0)
          << "Please check the symbol. node name: " << node->attrs.name
          << "op name " << node->op()->name << " has no inputs."
          << "It is likely that something went wrong during symbolic construction.";
      const auto &e = node->inputs[0];
      std::string suffix = GetSuffix(e, mirror_map);
      AddMultiCastNode(node->inputs, suffix, mirror_map, new_node);
    } else if (CheckConditionalFP32(conditional_fp32_ops, excluded_syms, node)) {
      for (const auto& node_entry : node->inputs) {
        if (mirror_fp32_map.count(node_entry)) {
          new_node->inputs.emplace_back(mirror_fp32_map[node_entry]);
        } else {
          NodePtr mirror_node = mirror_map.at(node_entry.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
          std::string suffix = GetSuffix(node_entry, mirror_map);
          AddCastNode(node_entry, suffix, mirror_entry, "float32", &mirror_fp32_map,
                      new_node);
        }
      }
    } else {
      for (const auto& node_entry : node->inputs) {
        NodePtr mirror_node = mirror_map.at(node_entry.node.get());
        new_node->inputs.emplace_back(mirror_node, node_entry.index, node_entry.version);
      }
    }
    mirror_map[node.get()] = std::move(new_node);
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
      outputs.emplace_back(mirror_map.at(e.node.get()), e.index, e.version);
  }

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

NNVM_REGISTER_PASS(ReducePrecision)
    .describe("add cast layers for low precision inference")
    .set_body(ReducePrecision)
    .set_change_graph(true);
}  // namespace mxnet
