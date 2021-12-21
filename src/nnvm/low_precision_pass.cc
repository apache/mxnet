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
static std::string GetSuffix(const nnvm::NodeEntry& node_entry) {
  static const auto& flist_outputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  std::string suffix               = "";
  if (node_entry.node->op() != nullptr) {
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

struct MappedNodeEntry {
  enum class DType { Unchanged, Target, Unknown };

  NodeEntry as_unchanged() {
    if (dtype == DType::Unchanged) {
      return entry;
    }
    if (fp32_entry.node == nullptr) {
      cast(mshadow::kFloat32);
      CHECK(fp32_entry.node);
    }
    return fp32_entry;
  }

  NodeEntry as_lp(int target_dtype) {
    // low precision
    if (dtype == DType::Target) {
      return entry;
    }
    if (lp16_entry.node == nullptr) {
      cast(target_dtype);
      CHECK(lp16_entry.node);
    }
    return lp16_entry;
  }

 private:
  void cast(int target_dtype) {
    CHECK((target_dtype == mshadow::kFloat32 && fp32_entry.node == nullptr &&
           dtype != DType::Unchanged) ||
          (target_dtype != mshadow::kFloat32 && lp16_entry.node == nullptr &&
           dtype != DType::Target));

    const std::string dt_name = dtype_name(target_dtype);
    const std::string suffix  = GetSuffix(entry);

    ObjectPtr cast_node = InsertNode(
        "amp_cast", entry.node->attrs.name + suffix + "_amp_cast_" + dt_name, nullptr, entry);
    cast_node->attrs.dict["dtype"] = dt_name;
    cast_node->op()->attr_parser(&(cast_node->attrs));

    const NodeEntry cast_node_entry = NodeEntry{std::move(cast_node), 0, 0};
    if (target_dtype == mshadow::kFloat32) {
      fp32_entry = cast_node_entry;
    } else {
      lp16_entry = cast_node_entry;
    }
  }

 public:
  DType dtype = DType::Unchanged;
  NodeEntry entry;
  NodeEntry lp16_entry;  // associated target_dtype amp_cast entry (if any)
  NodeEntry fp32_entry;  // associated fp32 amp_cast entry (if any)
};

using EntryMap_t = nnvm::NodeEntryMap<MappedNodeEntry>;
using NodeMap_t  = std::unordered_map<Node*, ObjectPtr>;

// Here we assume, that the nodes that are already cast to low precision, originally had f32 output
void keep_original_node(const ObjectPtr& old_node,
                        const NodeMap_t& node_map,
                        EntryMap_t* const entry_map) {
  const ObjectPtr& new_node = node_map.at(old_node.get());
  for (const auto& old_node_input_entry : old_node->inputs) {
    auto& mapped_node_entry = entry_map->at(old_node_input_entry);
    new_node->inputs.push_back(mapped_node_entry.as_unchanged());
  }
}

bool try_low_precision(const ObjectPtr& old_node,
                       const int target_dtype,
                       const NodeMap_t& node_map,
                       EntryMap_t* const entry_map) {
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

  const ObjectPtr& new_node = node_map.at(old_node.get());
  for (size_t i = 0; i < old_node->inputs.size(); ++i) {
    auto& mapped_input_entry = entry_map->at(old_node->inputs[i]);
    if (in_types[i] == target_dtype) {
      new_node->inputs.push_back(mapped_input_entry.as_lp(target_dtype));
    } else {
      new_node->inputs.push_back(mapped_input_entry.as_unchanged());
      CHECK(mapped_input_entry.fp32_entry.node == nullptr || in_types[i] == mshadow::kFloat32 ||
            in_types[i] == -1);
    }
  }

  for (size_t i = 0; i < old_node->num_outputs(); ++i) {
    const auto out_entry = NodeEntry(old_node, i, 0);
    if (out_types[i] == target_dtype) {
      entry_map->at(out_entry).dtype = MappedNodeEntry::DType::Target;
    }
  }

  return true;
}

bool try_widest_dtype_node(const ObjectPtr& old_node,
                           const int target_dtype,
                           const NodeMap_t& node_map,
                           EntryMap_t* entry_map) {
  static auto& infertype      = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static auto& fmutate_inputs = Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  const ObjectPtr& new_node   = node_map.at(old_node.get());

  if (fmutate_inputs.count(old_node->op()) != 0 &&
      fmutate_inputs[old_node->op()](old_node->attrs).size() != 0) {
    return false;
  }

  const auto& is_lp = [&](const NodeEntry& input) {
    return entry_map->at(input).dtype == MappedNodeEntry::DType::Target;
  };
  const auto& is_unknown = [&](const NodeEntry& input) {
    return entry_map->at(input).dtype == MappedNodeEntry::DType::Unknown;
  };

  const bool is_any_input_lp = std::any_of(old_node->inputs.begin(), old_node->inputs.end(), is_lp);
  const bool is_any_input_unknown =
      std::any_of(old_node->inputs.begin(), old_node->inputs.end(), is_unknown);
  if (!is_any_input_lp && !is_any_input_unknown) {
    return false;
  }
  if (std::all_of(old_node->inputs.begin(), old_node->inputs.end(), is_lp)) {
    for (const NodeEntry& old_node_input_entry : old_node->inputs) {
      new_node->inputs.push_back(entry_map->at(old_node_input_entry).entry);
    }
    // if we cannot infer the output type correctly, we cannot assume it is lp16
    std::vector<int> in_types(old_node->inputs.size(), target_dtype);
    std::vector<int> out_types(old_node->num_outputs(), -1);
    if (infertype.count(old_node->op())) {
      if (infertype[old_node->op()](old_node->attrs, &in_types, &out_types) == true) {
        for (size_t i = 0; i < old_node->num_outputs(); ++i) {
          if (out_types[i] == target_dtype) {
            const NodeEntry& out_entry     = NodeEntry(old_node, i, 0);
            entry_map->at(out_entry).dtype = MappedNodeEntry::DType::Target;
          }
        }
      }
    }
  } else {
    const std::string& node_name =
        old_node->inputs[0].node->attrs.name + GetSuffix(old_node->inputs[0]) + "_amp_multicast";
    ObjectPtr multicast_node = CreateNode("amp_multicast", node_name);
    for (const auto& old_node_entry : old_node->inputs) {
      multicast_node->inputs.push_back(entry_map->at(old_node_entry).entry);
    }
    multicast_node->attrs.dict["num_outputs"] = std::to_string(old_node->inputs.size());
    multicast_node->op()->attr_parser(&(multicast_node->attrs));

    for (uint32_t i = 0; i < old_node->inputs.size(); ++i) {
      new_node->inputs.emplace_back(multicast_node, i, 0);
    }
    for (uint32_t i = 0; i < new_node->num_outputs(); ++i) {
      const NodeEntry& old_out_entry     = NodeEntry(old_node, i, 0);
      entry_map->at(old_out_entry).dtype = MappedNodeEntry::DType::Unknown;
    }
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
  NodeMap_t node_map;
  EntryMap_t entry_map;
  nnvm::NodeEntryMap<std::vector<Node*>> output_nodes_of_variable;

  const auto& add_old_node_entry = [&](const NodeEntry& old_node_entry,
                                       const ObjectPtr& new_next_node) {
    // map the entry
    const ObjectPtr& old_node          = old_node_entry.node;
    const ObjectPtr& new_node          = node_map.at(old_node.get());
    MappedNodeEntry& mapped_node_entry = entry_map[old_node_entry];
    mapped_node_entry.entry = NodeEntry(new_node, old_node_entry.index, old_node_entry.version);

    if (old_node->op() == Op::Get("amp_cast")) {
      const NodeEntry& amp_cast_input_node_entry = old_node_entry.node->inputs[0];
      const op::AMPCastParam& param = nnvm::get<op::AMPCastParam>(old_node->attrs.parsed);
      if (param.dtype == target_dtype) {
        entry_map.at(amp_cast_input_node_entry).lp16_entry = mapped_node_entry.entry;
        mapped_node_entry.lp16_entry                       = mapped_node_entry.entry;
      } else if (param.dtype == mshadow::kFloat32) {
        entry_map.at(amp_cast_input_node_entry).fp32_entry = mapped_node_entry.entry;
        mapped_node_entry.fp32_entry                       = mapped_node_entry.entry;
      }
    } else if ((old_node->is_variable() && old_node->attrs.dict.count("__dtype__") > 0 &&
                old_node->attrs.dict.at("__dtype__") == std::to_string(target_dtype)) ||
               (data_name_types.count(old_node->attrs.name) &&
                data_name_types.at(old_node->attrs.name) == target_dtype)) {
      mapped_node_entry.lp16_entry = mapped_node_entry.entry;
    }

    // mark whether new_next_node use the model parameters directly (or through an amp_cast)
    if (!new_next_node) {
      return;
    }
    if (((old_node->op() == Op::Get("amp_cast") &&
          nnvm::get<op::AMPCastParam>(old_node->attrs.parsed).dtype == target_dtype) ||
         old_node->op() == Op::Get("amp_multicast")) &&
        old_node->inputs[0].node->is_variable() &&
        data_name_types.count(old_node->inputs[0].node->attrs.name) == 0) {
      const NodeEntry& amp_cast_input_node_entry = old_node_entry.node->inputs[0];
      output_nodes_of_variable[amp_cast_input_node_entry].push_back(new_next_node.get());
    } else if (old_node->is_variable() && data_name_types.count(old_node->attrs.name) == 0 &&
               new_next_node->op() != Op::Get("amp_multicast") &&
               (new_next_node->op() != Op::Get("amp_cast") ||
                nnvm::get<op::AMPCastParam>(new_next_node->attrs.parsed).dtype != target_dtype)) {
      output_nodes_of_variable[old_node_entry].push_back(new_next_node.get());
    }
  };

  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) {
    ObjectPtr new_node = Node::Create(*old_node);
    new_node->inputs.clear();

    for (const NodeEntry& old_node_input_entry : old_node->inputs) {
      add_old_node_entry(old_node_input_entry, new_node);
    }
    node_map[old_node.get()] = std::move(new_node);
  });
  for (const auto& output_node_entry : src.outputs) {
    add_old_node_entry(output_node_entry, nullptr);
  }

  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) {
    if (old_node->op() == Op::Get("amp_cast") || old_node->op() == Op::Get("amp_multicast")) {
      const ObjectPtr& new_node = node_map.at(old_node.get());
      for (const auto& in_ne : old_node->inputs) {
        const ObjectPtr& new_in_node = node_map.at(in_ne.node.get());
        new_node->inputs.emplace_back(new_in_node, in_ne.index, in_ne.version);
      }
      return;
    }
    /* 1. for node which needs to run in FP32 mode, add amp_cast operators
     * (to fp32) after its inputs
     * 2. for node which needs to run in LP16 mode, add amp_cast operators
     * (to target_dtype) after its inputs
     * 3. for nodes which need to run in widest dtype among its inputs, add
     * amp_multicast operators between op and its inputs
     * 4. for other nodes, create copy node and add it to the node_map
     */
    if ((!old_node->is_variable() && fp32_ops.count(old_node->op()->name) > 0) ||
        (excluded_syms.count(old_node->attrs.name) > 0) ||
        CheckConditionalFP32(conditional_fp32_ops, excluded_syms, old_node)) {
      keep_original_node(old_node, node_map, &entry_map);
    } else if (!old_node->is_variable() && target_dtype_ops.count(old_node->op()->name) > 0) {
      if (!try_low_precision(old_node, target_dtype, node_map, &entry_map)) {
        keep_original_node(old_node, node_map, &entry_map);
      }
    } else if (!old_node->is_variable() && widest_dtype_ops.count(old_node->op()->name) > 0) {
      if (!try_widest_dtype_node(old_node, target_dtype, node_map, &entry_map))
        keep_original_node(old_node, node_map, &entry_map);
    } else {
      bool try_lp =
          (std::any_of(old_node->inputs.begin(),
                       old_node->inputs.end(),
                       [&](const auto& old_node_input_entry) {
                         return entry_map.at(old_node_input_entry).dtype ==
                                    MappedNodeEntry::DType::Target ||
                                entry_map.at(old_node_input_entry).lp16_entry.node != nullptr;
                       }) &&
           old_node->op() != Op::Get("amp_cast") && old_node->op() != Op::Get("amp_multicast"));

      // handle operators from LP16_FP32_FUNCS - operators that can run on lp16, but only when
      // inputs are already cast
      bool runs_lp16 = try_lp && try_low_precision(old_node, target_dtype, node_map, &entry_map);
      if (!runs_lp16) {
        keep_original_node(old_node, node_map, &entry_map);
      }
    }
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    MappedNodeEntry& mirror_entry = entry_map.at(e);
    outputs.push_back(mirror_entry.as_unchanged());
  }

  std::vector<Node*> target_dtype_variable_nodes;
  if (cast_optional_params) {
    const NodeEntryEqual is_equal;
    for (auto& kv : output_nodes_of_variable) {
      const NodeEntry& old_variable_node_entry = kv.first;
      if (entry_map.at(old_variable_node_entry).lp16_entry.node == nullptr) {
        continue;
      }
      bool is_used_with_and_without_cast          = false;
      const NodeEntry& new_variable_node_entry    = entry_map.at(old_variable_node_entry).entry;
      const std::vector<Node*>& new_variable_outs = kv.second;
      for (Node* const new_node : new_variable_outs) {
        for (const NodeEntry new_node_input_entry : new_node->inputs) {
          // if input of this node is directly the variable, and not the amp_cast or amp_multicast,
          // then this variable cannot be cast offline
          if (is_equal(new_node_input_entry, new_variable_node_entry)) {
            is_used_with_and_without_cast = true;
            break;
          }
        }
        if (is_used_with_and_without_cast) {
          break;
        }
      }
      if (is_used_with_and_without_cast ||
          data_name_types.count(new_variable_node_entry.node->attrs.name)) {
        continue;
      }
      for (Node* const new_node : new_variable_outs) {
        bool skipped_amp_cast = false;
        for (NodeEntry& new_node_input_entry : new_node->inputs) {
          if (new_node_input_entry.node->op() == Op::Get("amp_cast") &&
              is_equal(new_node_input_entry.node->inputs[0], new_variable_node_entry)) {
            new_node_input_entry = new_variable_node_entry;
            skipped_amp_cast     = true;
            break;
          } else if (new_node_input_entry.node->op() == Op::Get("amp_multicast")) {
            const auto& found = std::find_if(
                new_node_input_entry.node->inputs.begin(),
                new_node_input_entry.node->inputs.end(),
                [&](const NodeEntry& ne) { return is_equal(ne, new_variable_node_entry); });
            if (found != new_node_input_entry.node->inputs.end()) {
              skipped_amp_cast = true;
              break;
            }
          }
        }
        CHECK(skipped_amp_cast);
      }
      target_dtype_variable_nodes.push_back(new_variable_node_entry.node.get());
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
