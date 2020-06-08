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
using nnvm::ObjectPtr;
using nnvm::NodeEntry;
using nnvm::Graph;

// create a node for operator : op_name with name : node_name
static ObjectPtr CreateNode(std::string op_name, std::string node_name) {
  ObjectPtr node = Node::Create();
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

static ObjectPtr InsertNode(std::string op_name, std::string node_name, ObjectPtr current,
                          NodeEntry previous) {
    ObjectPtr node = CreateNode(op_name, node_name);
    node->inputs.emplace_back(previous);
    if (current) current->inputs.emplace_back(NodeEntry{node, 0, 0});
    return node;
}

// get suffix for a node entry so that it can be used for amp_cast/amp_multicast node name
static std::string GetSuffix(const nnvm::NodeEntry &node_entry,
                             const std::unordered_map<Node*, ObjectPtr> &mirror_map) {
  static const auto &flist_outputs =
      nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  std::string suffix = "";
  ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
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
                        ObjectPtr curr_node) {
  ObjectPtr cast_node =
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
                             const std::unordered_map<Node *, ObjectPtr> &mirror_map,
                             ObjectPtr curr_node) {
  ObjectPtr node =
      CreateNode("amp_multicast",
                 inputs[0].node->attrs.name + node_name + "_amp_multicast");
  for (const auto &node_entry : inputs) {
    ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
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
    const std::unordered_set<std::string> &excluded_syms, ObjectPtr node) {
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
  static auto& fmutate_inputs = Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  static auto& infertype = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
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
  const auto data_name_types = src.GetAttr<std::unordered_map<std::string, int>>("data_name_types");
  const auto cast_optional_params = src.GetAttr<int>("cast_optional_params");

  CHECK(target_dtype == mshadow::kFloat16 || target_dtype == mshadow::kBfloat16)
      << "Only float16 and bfloat16 target_dtype is supported yet," << target_dtype;

  std::string target_dtype_str = "float32";
  if (target_dtype == mshadow::kFloat16) {
    target_dtype_str = "float16";
  } else if (target_dtype == mshadow::kBfloat16) {
    target_dtype_str = "bfloat16";
  }

  // Additional data structures to share common cast node inputs among different nodes
  std::unordered_map<Node *, ObjectPtr> mirror_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_fp32_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_target_dtype_map;

  // Visit nodes in a topologically sorted order
  DFSVisit(src.outputs, [&](const ObjectPtr &node) {
    ObjectPtr new_node = Node::Create(*node);
    new_node->inputs.clear();
    std::vector<uint32_t> mutable_inputs;
    if (fmutate_inputs.count(node->op()) != 0) {
      mutable_inputs = fmutate_inputs[node->op()](node->attrs);
    }
    /* 1. for node which needs to run in FP32 mode, add amp_cast operators
     * (to fp32) after its inputs
     * 2. for node which needs to run in LP16 mode, add amp_cast operators
     * (to target_dtype) after its inputs
     * 3. for nodes which need to run in widest dtype among its inputs, add
     * amp_multicast operators between op and its inputs
     * 4. for nodes which need to run in FP32 mode, based on a specific condition,
     * check the condition, and if true add amp_cast (to fp32) after its inputs
     * 4. for other nodes, create copy node and add it to the mirror_map
     */
    if ((!node->is_variable() && fp32_ops.count(node->op()->name) > 0) ||
        (excluded_syms.count(node->attrs.name) > 0)) {
      // Add output entry to fp32_map
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const auto out_entry = NodeEntry(node, i, 0);
        mirror_fp32_map[out_entry] = NodeEntry(new_node, i, 0);
      }
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const auto &node_entry = node->inputs[i];
        if (mirror_fp32_map.count(node_entry)) {
          new_node->inputs.emplace_back(mirror_fp32_map[node_entry]);
        } else if (node_entry.node->is_variable()) {
          // For variable, assume they are already fp32
          ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
          new_node->inputs.emplace_back(mirror_node, node_entry.index, node_entry.version);
        } else {
          ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
          std::string suffix = GetSuffix(node_entry, mirror_map);
          AddCastNode(node_entry, suffix, mirror_entry, "float32", &mirror_fp32_map, new_node);
        }
      }
    } else if (!node->is_variable() && target_dtype_ops.count(node->op()->name) > 0 &&
               excluded_syms.count(node->attrs.name) == 0) {
      std::vector<int> in_types(node->inputs.size(), -1);
      std::vector<int> out_types(node->num_outputs(), -1);
      if (infertype.count(node->op())) {
        // Try to infertype with target dtype. And add output entry to mirror_target_dtype_map or
        // mirror_fp32_map based on infered result.
        in_types[0] = target_dtype;
        bool infer_type_success = infertype[node->op()](node->attrs, &in_types, &out_types);
        CHECK(infer_type_success == true);
        for (size_t i = 0; i < node->num_outputs(); ++i) {
          const auto out_entry = NodeEntry(node, i, 0);
          if (out_types[i] == target_dtype) {
            mirror_target_dtype_map[out_entry] = NodeEntry(new_node, i, 0);
          } else if (out_types[i] == 0) {
            mirror_fp32_map[out_entry] = NodeEntry(new_node, i, 0);
          }
        }
      }
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const auto &node_entry = node->inputs[i];
        if (mirror_target_dtype_map.count(node_entry)) {
          new_node->inputs.emplace_back(mirror_target_dtype_map[node_entry]);
        } else if ((cast_optional_params && node_entry.node->is_variable() &&
                    !data_name_types.count(node_entry.node->attrs.name)) ||
                   (std::find(mutable_inputs.begin(), mutable_inputs.end(), i) !=
                    mutable_inputs.end()) ||
                   !(in_types[i] == target_dtype || in_types[i] == -1)) {
          // Here's some rules that not insert amp_cast for inputs:
          // 1. cast_optional_params is True, node_entry.node is variable and its not the data of
          //    the network. This is network params that offline converted to target dtype.
          // 2. Mutable inputs.
          // 3. Even the input[0] is target dtype, some operations still require float32 for other
          //    inputs. For example, Batchnorm.
          ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
          const auto mirror_entry = NodeEntry(mirror_node, node_entry.index, node_entry.version);
          new_node->inputs.push_back(mirror_entry);
          if ((cast_optional_params && node_entry.node->is_variable())) {
            // Node is target dtype
            mirror_target_dtype_map[node_entry] = mirror_entry;
          }
        } else {
          ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
          std::string suffix = GetSuffix(node_entry, mirror_map);
          AddCastNode(node_entry, suffix, mirror_entry, target_dtype_str, &mirror_target_dtype_map,
                      new_node);
        }
      }
    } else if (!node->is_variable() &&
               widest_dtype_ops.count(node->op()->name) > 0 &&
               excluded_syms.count(node->attrs.name) == 0) {
      CHECK(node->inputs.size() > 0)
          << "Please check the symbol. node name: " << node->attrs.name
          << "op name " << node->op()->name << " has no inputs."
          << "It is likely that something went wrong during symbolic construction.";
      CHECK_EQ(mutable_inputs.size(), 0)
          << "can't handle the widest_dtype_ops with mutable inputs.";
      int out_dtype = target_dtype;
      bool have_unknown_dtype = false;
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        // Try to infer output dtype based on input dtype
        if (!mirror_target_dtype_map.count(node->inputs[i])
            && !mirror_fp32_map.count(node->inputs[i])) {
          have_unknown_dtype = true;
          break;
        } else if (mirror_fp32_map.count(node->inputs[i])) {
          out_dtype = mshadow::kFloat32;
        }
      }
      if (have_unknown_dtype) {
        // We can't infer all dtype for inputs, so we need to add AddMultiCastNode here.
        const auto &e = node->inputs[0];
        std::string suffix = GetSuffix(e, mirror_map);
        AddMultiCastNode(node->inputs, suffix, mirror_map, new_node);
      } else {
        for (size_t i = 0; i < node->num_outputs(); ++i) {
          const auto out_entry = NodeEntry(node, i, 0);
          if (out_dtype == target_dtype) {
            mirror_target_dtype_map[out_entry] = NodeEntry(new_node, i, 0);
          } else {
            mirror_fp32_map[out_entry] = NodeEntry(new_node, i, 0);
          }
        }
        // we know all dtype from inputs, then we can use amp_cast instead.
        for (size_t i = 0; i < node->inputs.size(); ++i) {
          const auto &node_entry = node->inputs[i];
          if (out_dtype == target_dtype) {
            if (mirror_target_dtype_map.count(node_entry)) {
              new_node->inputs.emplace_back(mirror_target_dtype_map[node_entry]);
            } else {
              ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
              NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
              std::string suffix = GetSuffix(node_entry, mirror_map);
              AddCastNode(node_entry, suffix, mirror_entry, target_dtype_str,
                          &mirror_target_dtype_map, new_node);
            }
          } else {
            if (mirror_fp32_map.count(node_entry)) {
              new_node->inputs.emplace_back(mirror_fp32_map[node_entry]);
            } else {
              ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
              NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
              std::string suffix = GetSuffix(node_entry, mirror_map);
              AddCastNode(node_entry, suffix, mirror_entry, "float32", &mirror_fp32_map, new_node);
            }
          }
        }
      }
    } else if (CheckConditionalFP32(conditional_fp32_ops, excluded_syms, node)) {
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        const auto out_entry = NodeEntry(node, i, 0);
        mirror_fp32_map[out_entry] = NodeEntry(new_node, i, 0);
      }
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const auto &node_entry = node->inputs[i];
        if (mirror_fp32_map.count(node_entry)) {
          new_node->inputs.emplace_back(mirror_fp32_map[node_entry]);
        } else if (std::find(mutable_inputs.begin(), mutable_inputs.end(), i) !=
                   mutable_inputs.end()) {
          // Can't insert amp_cast for this inputs. Such op have to handle fp32 inputs itself.
          ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
          new_node->inputs.emplace_back(mirror_node, node_entry.index, node_entry.version);
        } else {
          ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, node_entry.index, node_entry.version};
          std::string suffix = GetSuffix(node_entry, mirror_map);
          AddCastNode(node_entry, suffix, mirror_entry, "float32", &mirror_fp32_map, new_node);
        }
      }
    } else {
      if (node->inputs.size() && (mirror_fp32_map.count(node->inputs[0]) ||
                                  mirror_target_dtype_map.count(node->inputs[0]))) {
        // If we know the dtype of input[0], then we will try to infer the dtype of its output, and
        // add the result to mirror_target_dtype_map or mirror_fp32_map.
        const int in_type =
            mirror_target_dtype_map.count(node->inputs[0]) ? target_dtype : mshadow::kFloat32;
        std::vector<int> in_types(node->inputs.size(), -1);
        std::vector<int> out_types(node->num_outputs(), -1);
        if (infertype.count(node->op())) {
          in_types[0] = in_type;
          bool infer_type_success = infertype[node->op()](node->attrs, &in_types, &out_types);
          if (infer_type_success) {
            for (size_t i = 0; i < node->num_outputs(); ++i) {
              const auto out_entry = NodeEntry(node, i, 0);
              if (out_types[i] == target_dtype) {
                mirror_target_dtype_map[out_entry] = NodeEntry(new_node, i, 0);
              } else if (out_types[i] == 0) {
                mirror_fp32_map[out_entry] = NodeEntry(new_node, i, 0);
              }
            }
          }
        }
      }
      for (const auto& node_entry : node->inputs) {
        ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
        new_node->inputs.emplace_back(mirror_node, node_entry.index, node_entry.version);
      }
    }
    mirror_map[node.get()] = std::move(new_node);
  });

  std::vector<NodeEntry> outputs;
  for (const auto &e : src.outputs) {
    if (mirror_fp32_map.count(e)) {
      outputs.emplace_back(mirror_fp32_map[e]);
    } else {
      ObjectPtr mirror_node = mirror_map.at(e.node.get());
      NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
      std::string suffix = GetSuffix(e, mirror_map);
      AddCastNode(e, suffix, mirror_entry, "float32", &mirror_fp32_map, nullptr);
      outputs.emplace_back(mirror_fp32_map[e]);
    }
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
