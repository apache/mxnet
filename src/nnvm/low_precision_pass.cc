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
#include "../operator/tensor/amp_cast.h"

namespace mxnet {
using nnvm::Graph;
using nnvm::Node;
using nnvm::NodeEntry;
using nnvm::NodeEntryEqual;
using nnvm::ObjectPtr;

static const char* dtype_name(const int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return "float32";
    case mshadow::kFloat16:
      return "float16";
    case mshadow::kBfloat16:
      return "bfloat16";
    default:
      LOG(FATAL) << "Unsupported type";
      return "";
  }
}

// create a node for operator : op_name with name : node_name
static ObjectPtr CreateNode(std::string op_name, std::string node_name) {
  ObjectPtr node   = Node::Create();
  node->attrs.name = node_name;
  if (op_name == "nullptr") {
    node->attrs.op = nullptr;
    // ugly workaround because VariableParam is not exposed
    node->attrs.parsed =
        nnvm::Symbol::CreateVariable(node->attrs.name).outputs[0].node->attrs.parsed;
  } else {
    node->attrs.op = Op::Get(op_name);
  }
  return node;
}

static ObjectPtr InsertNode(std::string op_name,
                            std::string node_name,
                            ObjectPtr current,
                            NodeEntry previous) {
  ObjectPtr node = CreateNode(op_name, node_name);
  node->inputs.emplace_back(previous);
  if (current)
    current->inputs.emplace_back(NodeEntry{node, 0, 0});
  return node;
}

// get suffix for a node entry so that it can be used for amp_cast/amp_multicast node name
static std::string GetSuffix(const nnvm::NodeEntry& node_entry,
                             const std::unordered_map<Node*, ObjectPtr>& mirror_map) {
  static const auto& flist_outputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  std::string suffix               = "";
  ObjectPtr mirror_node            = mirror_map.at(node_entry.node.get());
  if (mirror_node->op() != nullptr) {
    auto list_output_names_func = flist_outputs.get(node_entry.node->op(), nullptr);
    if (list_output_names_func != nullptr) {
      std::vector<std::string> names = list_output_names_func(node_entry.node->attrs);
      suffix                         = "_" + names[node_entry.index];
    } else {
      suffix = "_" + std::to_string(node_entry.index);
    }
  }
  return suffix;
}

// add amp_cast node between curr_node and input
static void AddCastNode(const nnvm::NodeEntry& e,
                        const std::string& suffix,
                        const nnvm::NodeEntry& input,
                        int target_dtype,
                        nnvm::NodeEntryMap<NodeEntry>* mirror_entry_map,
                        ObjectPtr curr_node) {
  const std::string dtype = dtype_name(target_dtype);
  ObjectPtr cast_node =
      InsertNode("amp_cast", e.node->attrs.name + suffix + "_amp_cast_" + dtype, curr_node, input);
  cast_node->attrs.dict["dtype"] = dtype;
  cast_node->op()->attr_parser(&(cast_node->attrs));
  (*mirror_entry_map)[e] = NodeEntry{std::move(cast_node), 0, e.version};
  return;
}

// add amp_multicast node between curr_node and inputs
static void AddMultiCastNode(const std::vector<NodeEntry>& inputs,
                             const std::string& node_name,
                             const std::unordered_map<Node*, ObjectPtr>& mirror_map,
                             ObjectPtr curr_node) {
  ObjectPtr node =
      CreateNode("amp_multicast", inputs[0].node->attrs.name + node_name + "_amp_multicast");
  for (const auto& node_entry : inputs) {
    ObjectPtr mirror_node = mirror_map.at(node_entry.node.get());
    NodeEntry mirror_entry =
        NodeEntry{std::move(mirror_node), node_entry.index, node_entry.version};
    node->inputs.emplace_back(mirror_entry);
  }
  node->attrs.dict["num_outputs"] = std::to_string(inputs.size());
  node->op()->attr_parser(&(node->attrs));
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    const auto& e = inputs[i];
    curr_node->inputs.emplace_back(NodeEntry{node, static_cast<uint32_t>(i), e.version});
  }
  return;
}

static bool CheckConditionalFP32(
    const std::unordered_map<std::string,
                             std::unordered_map<std::string, std::vector<std::string>>>&
        conditional_fp32_ops,
    const std::unordered_set<std::string>& excluded_syms,
    ObjectPtr node) {
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
      for (auto& it_param : it_params) {
        auto param_key = node->attrs.dict.find(it_param.first);
        if (param_key != node->attrs.dict.end()) {
          auto it_param_vals = it_param.second;
          if (std::find(it_param_vals.begin(), it_param_vals.end(), param_key->second) !=
              it_param_vals.end()) {
            return true;
          }
        }
      }
    }
    return false;
  }
}

// returns true when the original node was used, and false when it had to be cast to f32
bool keep_original_input_or_cast_it_to_f32(
    const ObjectPtr& new_node,
    const NodeEntry& old_node_input_entry,
    const std::unordered_map<Node*, ObjectPtr>& mirror_map,
    nnvm::NodeEntryMap<NodeEntry>* mirror_fp32_map,
    const nnvm::NodeEntryMap<NodeEntry>& mirror_target_dtype_map) {
  if (mirror_fp32_map->count(old_node_input_entry)) {
    new_node->inputs.emplace_back(mirror_fp32_map->at(old_node_input_entry));
    return false;
  } else if (mirror_target_dtype_map.count(old_node_input_entry) &&
             mirror_target_dtype_map.at(old_node_input_entry).node->op() != Op::Get("amp_cast") &&
             old_node_input_entry.node->is_variable() == false) {
    // we never cast variables to f32 (even the ones with lp16 dtype), as they are the part of the
    // original graph. We only cast the nodes, that we cast to lp16 ourself
    const ObjectPtr new_node_input_node = mirror_map.at(old_node_input_entry.node.get());
    const NodeEntry new_node_input_node_entry =
        NodeEntry{new_node_input_node, old_node_input_entry.index, old_node_input_entry.version};
    const std::string suffix = GetSuffix(old_node_input_entry, mirror_map);
    AddCastNode(old_node_input_entry,
                suffix,
                new_node_input_node_entry,
                mshadow::kFloat32,
                mirror_fp32_map,
                new_node);
    return false;
  } else {
    const ObjectPtr new_node_input_node = mirror_map.at(old_node_input_entry.node.get());
    const NodeEntry new_node_input_node_entry =
        NodeEntry{new_node_input_node, old_node_input_entry.index, old_node_input_entry.version};
    new_node->inputs.emplace_back(new_node_input_node_entry);
    return true;
  }
}

// Here we assume, that the nodes that are already cast to low precision, originally had f32 output
void keep_original_node(const ObjectPtr& old_node,
                        const ObjectPtr& new_node,
                        const std::unordered_map<Node*, ObjectPtr>& mirror_map,
                        nnvm::NodeEntryMap<NodeEntry>* mirror_fp32_map,
                        const nnvm::NodeEntryMap<NodeEntry>& mirror_target_dtype_map) {
  for (const auto& old_node_input_entry : old_node->inputs) {
    keep_original_input_or_cast_it_to_f32(
        new_node, old_node_input_entry, mirror_map, mirror_fp32_map, mirror_target_dtype_map);
  }
}

bool try_low_precision(const ObjectPtr& old_node,
                       const ObjectPtr& new_node,
                       const int target_dtype,
                       const std::unordered_map<Node*, ObjectPtr>& mirror_map,
                       nnvm::NodeEntryMap<NodeEntry>* mirror_fp32_map,
                       nnvm::NodeEntryMap<NodeEntry>* mirror_target_dtype_map) {
  static auto& infertype      = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static auto& fmutate_inputs = Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");

  std::vector<int> in_types(old_node->inputs.size(), -1);
  std::vector<int> out_types(old_node->num_outputs(), -1);
  in_types[0] = target_dtype;

  if (infertype.count(old_node->op()) == 0 ||
      infertype[old_node->op()](old_node->attrs, &in_types, &out_types) == false) {
    return false;
  }

  std::vector<uint32_t> mutable_inputs;
  if (fmutate_inputs.count(old_node->op()) != 0) {
    mutable_inputs = fmutate_inputs[old_node->op()](old_node->attrs);
  }
  for (size_t i = 0; i < old_node->inputs.size(); ++i) {
    if ((in_types[i] == target_dtype) &&
        std::find(mutable_inputs.begin(), mutable_inputs.end(), i) != mutable_inputs.end()) {
      return false;
    }
  }

  for (size_t i = 0; i < old_node->inputs.size(); ++i) {
    const auto& old_node_input_entry = old_node->inputs[i];
    if (in_types[i] == target_dtype) {
      if (mirror_target_dtype_map->count(old_node_input_entry)) {
        new_node->inputs.emplace_back(mirror_target_dtype_map->at(old_node_input_entry));
      } else {
        const ObjectPtr& mirror_node = mirror_map.at(old_node_input_entry.node.get());
        const NodeEntry& mirror_entry =
            NodeEntry{mirror_node, old_node_input_entry.index, old_node_input_entry.version};
        const std::string& suffix = GetSuffix(old_node_input_entry, mirror_map);
        AddCastNode(old_node_input_entry,
                    suffix,
                    mirror_entry,
                    target_dtype,
                    mirror_target_dtype_map,
                    new_node);
      }
    } else {
      CHECK(keep_original_input_or_cast_it_to_f32(new_node,
                                                  old_node_input_entry,
                                                  mirror_map,
                                                  mirror_fp32_map,
                                                  *mirror_target_dtype_map) ||
            in_types[i] == mshadow::kFloat32 || in_types[i] == -1);
    }
  }

  for (size_t i = 0; i < old_node->num_outputs(); ++i) {
    const auto out_entry = NodeEntry(old_node, i, 0);
    if (out_types[i] == target_dtype) {
      (*mirror_target_dtype_map)[out_entry] = NodeEntry(new_node, i, 0);
    }
  }

  return true;
}

bool try_widest_dtype_node(const ObjectPtr& old_node,
                           const ObjectPtr& new_node,
                           const int target_dtype,
                           const std::unordered_map<Node*, ObjectPtr>& mirror_map,
                           nnvm::NodeEntryMap<NodeEntry>* mirror_target_dtype_map) {
  static auto& infertype      = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static auto& fmutate_inputs = Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");

  if (fmutate_inputs.count(old_node->op()) != 0 &&
      fmutate_inputs[old_node->op()](old_node->attrs).size() != 0) {
    return false;
  }

  const auto& is_lp = [&](const NodeEntry& input) {
    return mirror_target_dtype_map->count(input) &&
           mirror_target_dtype_map->at(input).node->op() != Op::Get("amp_cast");
  };

  if (!std::any_of(old_node->inputs.begin(), old_node->inputs.end(), is_lp)) {
    return false;
  }
  if (std::all_of(old_node->inputs.begin(), old_node->inputs.end(), is_lp)) {
    for (const NodeEntry& old_node_input_entry : old_node->inputs) {
      new_node->inputs.emplace_back(mirror_target_dtype_map->at(old_node_input_entry));
    }
    // if we cannot infer the output type correctly, we cannot assume it is lp16
    std::vector<int> in_types(old_node->inputs.size(), target_dtype);
    std::vector<int> out_types(old_node->num_outputs(), -1);
    if (infertype.count(old_node->op())) {
      if (infertype[old_node->op()](old_node->attrs, &in_types, &out_types) == true) {
        for (size_t i = 0; i < old_node->num_outputs(); ++i) {
          if (out_types[i] == target_dtype) {
            const auto out_entry                  = NodeEntry(old_node, i, 0);
            (*mirror_target_dtype_map)[out_entry] = NodeEntry(new_node, i, 0);
          }
        }
      }
    }
  } else {
    const std::string& suffix = GetSuffix(old_node->inputs[0], mirror_map);
    AddMultiCastNode(old_node->inputs, suffix, mirror_map, new_node);
  }
  return true;
}

Graph ReducePrecision(Graph&& src) {
  const auto target_dtype_ops = src.GetAttr<std::unordered_set<std::string>>("target_dtype_ops");
  const auto fp32_ops         = src.GetAttr<std::unordered_set<std::string>>("fp32_ops");
  const auto widest_dtype_ops = src.GetAttr<std::unordered_set<std::string>>("widest_dtype_ops");
  const auto target_dtype     = src.GetAttr<int>("target_dtype");
  const auto excluded_syms    = src.GetAttr<std::unordered_set<std::string>>("excluded_syms");
  const auto conditional_fp32_ops = src.GetAttr<
      std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>>(
      "conditional_fp32_ops");
  const auto data_name_types = src.GetAttr<std::unordered_map<std::string, int>>("data_name_types");
  const auto cast_optional_params = src.GetAttr<int>("cast_optional_params");

  CHECK(target_dtype == mshadow::kFloat16 || target_dtype == mshadow::kBfloat16)
      << "Only float16 and bfloat16 target_dtype is supported yet," << target_dtype;

  // Additional data structures to share common cast node inputs among different nodes
  std::unordered_map<Node*, ObjectPtr> mirror_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_fp32_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_target_dtype_map;
  nnvm::NodeEntryMap<std::vector<Node*>> output_nodes_of_variable;

  // Visit nodes in a topologically sorted order
  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) {
    ObjectPtr new_node = Node::Create(*old_node);
    new_node->inputs.clear();
    /* 1. for node which needs to run in FP32 mode, add amp_cast operators
     * (to fp32) after its inputs
     * 2. for node which needs to run in LP16 mode, add amp_cast operators
     * (to target_dtype) after its inputs
     * 3. for nodes which need to run in widest dtype among its inputs, add
     * amp_multicast operators between op and its inputs
     * 4. for other nodes, create copy node and add it to the mirror_map
     */
    if ((!old_node->is_variable() && fp32_ops.count(old_node->op()->name) > 0) ||
        (excluded_syms.count(old_node->attrs.name) > 0) ||
        CheckConditionalFP32(conditional_fp32_ops, excluded_syms, old_node)) {
      keep_original_node(old_node, new_node, mirror_map, &mirror_fp32_map, mirror_target_dtype_map);
    } else if (!old_node->is_variable() && target_dtype_ops.count(old_node->op()->name) > 0) {
      if (!try_low_precision(old_node,
                             new_node,
                             target_dtype,
                             mirror_map,
                             &mirror_fp32_map,
                             &mirror_target_dtype_map)) {
        keep_original_node(
            old_node, new_node, mirror_map, &mirror_fp32_map, mirror_target_dtype_map);
      }
    } else if (!old_node->is_variable() && widest_dtype_ops.count(old_node->op()->name) > 0) {
      if (!try_widest_dtype_node(
              old_node, new_node, target_dtype, mirror_map, &mirror_target_dtype_map))
        keep_original_node(
            old_node, new_node, mirror_map, &mirror_fp32_map, mirror_target_dtype_map);
    } else {
      bool try_lp = std::any_of(
          old_node->inputs.begin(), old_node->inputs.end(), [&](const auto& old_node_input_entry) {
            return mirror_target_dtype_map.count(old_node_input_entry);
          });

      const NodeEntry new_node_entry = NodeEntry(new_node, 0, 0);
      if (old_node->op() == Op::Get("amp_cast")) {
        const op::AMPCastParam& param = nnvm::get<op::AMPCastParam>(old_node->attrs.parsed);
        if (param.dtype == target_dtype) {
          if (mirror_target_dtype_map.count(old_node->inputs[0]) == 0) {
            mirror_target_dtype_map[old_node->inputs[0]] = new_node_entry;
          }
          mirror_target_dtype_map[NodeEntry(old_node, 0, 0)] = new_node_entry;
        } else if (try_lp && param.dtype == mshadow::kFloat32 &&
                   mirror_fp32_map.count(old_node->inputs[0]) == 0) {
          mirror_fp32_map[old_node->inputs[0]] = new_node_entry;
        }
        try_lp = false;
      } else if ((old_node->is_variable() && old_node->attrs.dict.count("__dtype__") > 0 &&
                  old_node->attrs.dict.at("__dtype__") == std::to_string(target_dtype)) ||
                 (data_name_types.count(old_node->attrs.name) &&
                  data_name_types.at(old_node->attrs.name) == target_dtype)) {
        mirror_target_dtype_map[NodeEntry(old_node, 0, 0)] = new_node_entry;
      }

      // handle operators from LP16_FP32_FUNCS - operators that can run on lp16, but only when
      // inputs are already cast
      bool runs_lp16 = try_lp && try_low_precision(old_node,
                                                   new_node,
                                                   target_dtype,
                                                   mirror_map,
                                                   &mirror_fp32_map,
                                                   &mirror_target_dtype_map);
      if (!runs_lp16) {
        keep_original_node(
            old_node, new_node, mirror_map, &mirror_fp32_map, mirror_target_dtype_map);
      }
    }
    mirror_map[old_node.get()] = std::move(new_node);
    for (const NodeEntry& old_node_input_entry : old_node->inputs) {
      const auto& old_node_input_node = old_node_input_entry.node;
      if (old_node_input_node->op() == Op::Get("amp_cast") &&
          old_node_input_node->inputs[0].node->is_variable() &&
          data_name_types.count(old_node_input_node->inputs[0].node->attrs.name) == 0 &&
          nnvm::get<op::AMPCastParam>(old_node_input_node->attrs.parsed).dtype == target_dtype) {
        output_nodes_of_variable[old_node_input_node->inputs[0]].push_back(old_node.get());
      } else if (old_node_input_node->is_variable() &&
                 data_name_types.count(old_node_input_node->attrs.name) == 0 &&
                 (old_node->op() != Op::Get("amp_cast") ||
                  nnvm::get<op::AMPCastParam>(old_node->attrs.parsed).dtype != target_dtype)) {
        output_nodes_of_variable[old_node_input_entry].push_back(old_node.get());
      }
    }
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    const ObjectPtr& mirror_node = mirror_map.at(e.node.get());
    const NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
    if (mirror_target_dtype_map.count(e)) {
      const std::string& suffix = GetSuffix(e, mirror_map);
      AddCastNode(e, suffix, mirror_entry, mshadow::kFloat32, &mirror_fp32_map, nullptr);
      outputs.emplace_back(mirror_fp32_map[e]);
    } else {
      outputs.emplace_back(mirror_entry);
    }
  }

  std::vector<Node*> target_dtype_variable_nodes;
  if (cast_optional_params) {
    for (auto& kv : output_nodes_of_variable) {
      const NodeEntry old_variable_node_entry = kv.first;
      if (!mirror_target_dtype_map.count(old_variable_node_entry)) {
        continue;
      }

      bool is_used_with_and_without_cast          = false;
      const ObjectPtr& new_variable_node          = mirror_map[old_variable_node_entry.node.get()];
      const std::vector<Node*>& old_variable_outs = kv.second;
      for (Node* const old_node : old_variable_outs) {
        Node* const new_node = mirror_map[old_node].get();
        for (const NodeEntry new_node_input_entry : new_node->inputs) {
          // if input of this node is directly the variable, and not the amp_cast or amp_multicast,
          // then this variable cannot be cast offline
          if (new_node_input_entry.node == new_variable_node) {
            is_used_with_and_without_cast = true;
            break;
          }
        }
        if (is_used_with_and_without_cast) {
          break;
        }
      }

      if (is_used_with_and_without_cast ||
          data_name_types.count(old_variable_node_entry.node->attrs.name)) {
        continue;
      }

      const NodeEntryEqual is_equal;
      const NodeEntry new_variable_node_entry = NodeEntry{
          new_variable_node, old_variable_node_entry.index, old_variable_node_entry.version};
      for (Node* const old_node : old_variable_outs) {
        Node* const new_node  = mirror_map[old_node].get();
        bool skipped_amp_cast = false;
        for (NodeEntry& new_node_input_entry : new_node->inputs) {
          if (new_node_input_entry.node->op() == Op::Get("amp_cast") &&
              is_equal(new_node_input_entry.node->inputs[0], new_variable_node_entry)) {
            new_node_input_entry = new_variable_node_entry;
            skipped_amp_cast     = true;
            break;
          }
        }
        CHECK(skipped_amp_cast);
      }
      target_dtype_variable_nodes.push_back(new_variable_node.get());
    }
  }

  Graph ret;
  ret.outputs = std::move(outputs);

  const nnvm::IndexedGraph& idx = ret.indexed_graph();
  const auto& input_nodes       = idx.input_nodes();
  nnvm::DTypeVector arg_types(input_nodes.size(), -1);
  for (const auto& new_variable_node : target_dtype_variable_nodes) {
    const auto id    = idx.node_id(new_variable_node);
    const auto found = std::find(input_nodes.begin(), input_nodes.end(), id);
    CHECK(found != input_nodes.end());

    const auto arg_idx = found - input_nodes.begin();
    arg_types[arg_idx] = target_dtype;
  }
  ret.attrs["arg_types"] = std::make_shared<dmlc::any>(std::move(arg_types));

  return ret;
}

NNVM_REGISTER_PASS(ReducePrecision)
    .describe("add cast layers for low precision inference")
    .set_body(ReducePrecision)
    .set_change_graph(true);
}  // namespace mxnet
