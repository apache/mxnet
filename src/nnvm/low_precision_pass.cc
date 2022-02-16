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
#include "../operator/operator_common.h"

namespace mxnet {
using nnvm::Graph;
using nnvm::Node;
using nnvm::NodeEntry;
using nnvm::ObjectPtr;

bool is_cast_op(const nnvm::Op* const op) {
  return op && (op == Op::Get("amp_cast") || op == Op::Get("Cast"));
}

class MappedNodeEntry {
 public:
  MappedNodeEntry(const NodeEntry node_entry, const int original_dtype)
      : entry(std::move(node_entry)), original_dtype(original_dtype) {
    dtype = original_dtype;
  }

  void convert(const int new_dtype) {
    CHECK_EQ(dtype, original_dtype);  // dtype should be changed only once
    dtype = new_dtype;
  }

  NodeEntry as_original() {
    return as_type(original_dtype);
  }

  NodeEntry as_type(const int target_dtype) {
    if (dtype == target_dtype) {
      return entry;
    }
    NodeEntry& cast_entry = casts[original_dtype];
    if (cast_entry.node == nullptr) {
      cast_entry = cast(original_dtype);
      CHECK(cast_entry.node);
    }
    return cast_entry;
  }

  bool has_dtype_entry(const int target_dtype) {
    return dtype == target_dtype || casts.count(target_dtype) > 0;
  }

  bool can_be_cast_offline_to(const int target_dtype) {
    return casts.count(target_dtype) > 0;
  }

 private:
  ObjectPtr CreateCastNode(const std::string& op_name, const std::string& node_name) {
    CHECK(op_name.size() > 0);

    ObjectPtr node   = Node::Create();
    node->attrs.name = node_name;
    node->attrs.op   = Op::Get(op_name);
    node->inputs.emplace_back(entry);
    return node;
  }

  NodeEntry cast(const int new_dtype) {
    CHECK(new_dtype == nnvm::kBfloat16 || new_dtype == nnvm::kFloat16 ||
          new_dtype == nnvm::kFloat32);  // TODO: support every type

    const std::string dt_name        = mxnet::op::type_string(new_dtype);
    const std::string suffix         = "_" + std::to_string(entry.index);
    const std::string cast_node_name = entry.node->attrs.name + suffix + "_amp_cast_" + dt_name;
    ObjectPtr cast_node              = CreateCastNode("amp_cast", cast_node_name);
    cast_node->attrs.dict["dtype"]   = dt_name;
    cast_node->op()->attr_parser(&(cast_node->attrs));
    return NodeEntry{std::move(cast_node), 0, 0};
  }

 public:
  const NodeEntry entry;
  const int original_dtype;  // original dtype of the entry

 private:
  int dtype;  // current dtype of the entry
  std::unordered_map<int, NodeEntry> casts;
};

using EntryMap_t     = nnvm::NodeEntryMap<MappedNodeEntry>;
using NodeMap_t      = std::unordered_map<Node*, ObjectPtr>;
using NodeEntrySet_t = std::unordered_set<NodeEntry, nnvm::NodeEntryHash, nnvm::NodeEntryEqual>;

static void keep_original_node(const ObjectPtr& old_node,
                               const NodeMap_t& node_map,
                               EntryMap_t* const entry_map) {
  const ObjectPtr& new_node = node_map.at(old_node.get());
  for (const auto& old_node_input_entry : old_node->inputs) {
    auto& mapped_node_entry = entry_map->at(old_node_input_entry);
    new_node->inputs.push_back(mapped_node_entry.as_original());
  }
}

static bool try_low_precision(const int target_dtype,
                              const ObjectPtr& old_node,
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

  if (fmutate_inputs.count(old_node->op()) != 0) {
    std::vector<uint32_t> mutable_inputs = fmutate_inputs[old_node->op()](old_node->attrs);
    for (size_t i = 0; i < old_node->inputs.size(); ++i) {
      if (in_types[i] == target_dtype) {
        if (std::find(mutable_inputs.begin(), mutable_inputs.end(), i) != mutable_inputs.end()) {
          return false;
        }
      }
    }
  }

  const ObjectPtr& new_node = node_map.at(old_node.get());
  for (size_t i = 0; i < old_node->inputs.size(); ++i) {
    auto& mapped_input_entry = entry_map->at(old_node->inputs[i]);
    new_node->inputs.push_back(mapped_input_entry.as_type(in_types[i]));
  }

  for (size_t i = 0; i < old_node->num_outputs(); ++i) {
    // TODO use actual outputs (take version into account)
    const auto out_entry = NodeEntry(old_node, i, 0);
    entry_map->at(out_entry).convert(out_types[i]);
  }

  return true;
}

static bool try_widest_dtype_node(const ObjectPtr& old_node,
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
    return entry_map->at(input).original_dtype == target_dtype;
  };
  const bool is_any_input_lp = std::any_of(old_node->inputs.begin(), old_node->inputs.end(), is_lp);
  if (!is_any_input_lp) {
    return false;
  }

  if (std::all_of(old_node->inputs.begin(), old_node->inputs.end(), is_lp)) {
    for (const NodeEntry& old_node_input_entry : old_node->inputs) {
      new_node->inputs.push_back(entry_map->at(old_node_input_entry).entry);
    }
    std::vector<int> in_types(old_node->inputs.size(), target_dtype);
    std::vector<int> out_types(old_node->num_outputs(), -1);
    CHECK(infertype.count(old_node->op()));
    CHECK(infertype[old_node->op()](old_node->attrs, &in_types, &out_types));

    // TODO use actual outputs (take version into account)
    for (size_t i = 0; i < old_node->num_outputs(); ++i) {
      const NodeEntry& out_entry = NodeEntry(old_node, i, 0);
      entry_map->at(out_entry).convert(out_types[i]);
    }
  } else {
    for (const NodeEntry& old_node_input_entry : old_node->inputs) {
      NodeEntry new_node_input_entry = entry_map->at(old_node_input_entry).as_original();
      new_node->inputs.push_back(std::move(new_node_input_entry));
    }
    // no need to update outputs in the entry_map since this node is unchanged
  }
  return true;
}

Graph ReducePrecision(Graph&& src) {
  const auto target_dtype        = src.GetAttr<int>("target_dtype");
  const auto cast_params_offline = src.GetAttr<int>("cast_params_offline");
  const auto& input_names        = src.GetAttr<std::unordered_set<std::string>>("input_names");
  const auto& target_dtype_ops   = src.GetAttr<std::unordered_set<std::string>>("target_dtype_ops");
  const auto& fp32_ops           = src.GetAttr<std::unordered_set<std::string>>("fp32_ops");
  const auto& widest_dtype_ops   = src.GetAttr<std::unordered_set<std::string>>("widest_dtype_ops");
  const auto& excluded_syms      = src.GetAttr<std::unordered_set<std::string>>("excluded_syms");
  const auto& dtypes             = src.GetAttr<nnvm::DTypeVector>("dtype");

  CHECK(target_dtype == mshadow::kFloat16 || target_dtype == mshadow::kBfloat16)
      << "Only float16 and bfloat16 target_dtype is supported yet," << target_dtype;

  const nnvm::IndexedGraph& src_idx = src.indexed_graph();
  CHECK_EQ(dtypes.size(), src_idx.num_node_entries());
  for (const int dtype : dtypes) {
    CHECK_NE(dtype, -1) << "Infer type failed with full information about input types";
  }

  NodeMap_t node_map;
  EntryMap_t entry_map;
  std::unordered_map<Node*, NodeEntrySet_t> node_out_entries;
  std::unordered_map<Node*, std::unordered_map<Node*, NodeEntry>> param_out_nodes;

  // gather information about node entries and build a new graph
  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) {
    ObjectPtr new_node = Node::Create(*old_node);
    new_node->inputs.clear();
    for (const NodeEntry& old_ne : old_node->inputs) {
      const ObjectPtr& new_input_node = node_map[old_ne.node.get()];
      const NodeEntry new_ne          = NodeEntry(new_input_node, old_ne.index, old_ne.version);
      const uint32_t entry_id         = src_idx.entry_id(old_ne);
      const int original_ne_dtype     = dtypes[entry_id];

      entry_map.emplace(old_ne, MappedNodeEntry(new_ne, original_ne_dtype));
      node_out_entries[new_input_node.get()].insert(new_ne);
      if (new_input_node->is_variable() && input_names.count(new_input_node->attrs.name) == 0) {
        // using old_ne to later get its mapped version
        param_out_nodes[new_input_node.get()][new_node.get()] = old_ne;
      }
    }
    node_map[old_node.get()] = std::move(new_node);
  });

  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) {
    if (old_node->is_variable() || old_node->op() == Op::Get("amp_multicast") ||
        is_cast_op(old_node->op())) {
      const ObjectPtr& new_node = node_map.at(old_node.get());
      for (const auto& in_ne : old_node->inputs) {
        const ObjectPtr& new_in_node = node_map.at(in_ne.node.get());
        new_node->inputs.emplace_back(new_in_node, in_ne.index, in_ne.version);
      }
      return;
    }

    if (fp32_ops.count(old_node->op()->name) > 0 || excluded_syms.count(old_node->attrs.name) > 0) {
      keep_original_node(old_node, node_map, &entry_map);
    } else if (target_dtype_ops.count(old_node->op()->name) > 0) {
      if (!try_low_precision(target_dtype, old_node, node_map, &entry_map)) {
        keep_original_node(old_node, node_map, &entry_map);
      }
    } else if (widest_dtype_ops.count(old_node->op()->name) > 0) {
      if (!try_widest_dtype_node(old_node, target_dtype, node_map, &entry_map))
        keep_original_node(old_node, node_map, &entry_map);
    } else {
      // handle operators that can run on lp16, but only when inputs are already cast
      const auto& has_lp_inputs = [&](const auto& old_node_input_entry) {
        return entry_map.at(old_node_input_entry).has_dtype_entry(target_dtype);
      };
      bool runs_lp = false;
      if (std::any_of(old_node->inputs.begin(), old_node->inputs.end(), has_lp_inputs)) {
        runs_lp = try_low_precision(target_dtype, old_node, node_map, &entry_map);
      }
      if (!runs_lp) {
        keep_original_node(old_node, node_map, &entry_map);
      }
    }
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    outputs.push_back(entry_map.at(e).as_original());
  }

  if (cast_params_offline) {
    for (const auto& [param, outs] : param_out_nodes) {
      const auto& can_be_cast_offline = [&](const auto& node_x_old_ne) {
        return entry_map.at(node_x_old_ne.second).can_be_cast_offline_to(target_dtype);
      };
      if (std::all_of(outs.begin(), outs.end(), can_be_cast_offline)) {
        nnvm::NodeEntryEqual are_equal;
        for (const auto& [node, old_ne] : outs) {
          bool skipped_amp_cast       = false;
          MappedNodeEntry& mapped_ne  = entry_map.at(old_ne);
          const NodeEntry& ne_to_skip = mapped_ne.as_type(target_dtype);
          for (NodeEntry& node_entry : node->inputs) {
            if (are_equal(node_entry, ne_to_skip)) {
              node_entry       = mapped_ne.entry;
              skipped_amp_cast = true;
              break;
            }
          }
          CHECK(skipped_amp_cast);
        }
      }
    }
  }

  Graph ret;
  ret.outputs = std::move(outputs);

  // const nnvm::IndexedGraph& idx = ret.indexed_graph();
  // const auto& input_nodes       = idx.input_nodes();
  // nnvm::DTypeVector arg_types(input_nodes.size(), -1);

  // ret.attrs["params_to_cast_offline"] =
  //     std::make_shared<dmlc::any>(std::move(params_to_cast_offline));

  return ret;
}

NNVM_REGISTER_PASS(ReducePrecision)
    .describe("add cast layers for low precision inference")
    .set_body(ReducePrecision)
    .set_change_graph(true);
}  // namespace mxnet