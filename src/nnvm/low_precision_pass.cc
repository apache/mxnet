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
#include "operator/operator_common.h"
#include "common/utils.h"

namespace mxnet {
using nnvm::Graph;
using nnvm::Node;
using nnvm::NodeEntry;
using nnvm::ObjectPtr;

bool IsCastOp(const nnvm::Op* const op) {
  return op && (op == Op::Get("amp_cast") || op == Op::Get("Cast"));
}
/*!
 * \brief Before the model conversion, node entries of the original graph are mapped to the
 * equivalent node entries in the new graph that will be then converted to a mixed precision graph.
 * This class wraps a mapped NodeEntry from the new graph, providing a transparent interface for
 * acquiring versions of the wrapped entry with a specific dtype, adding a casting nodes to the
 * graph when needed (one for each unique dtype that was requested).
 */
class MappedNodeEntry {
 public:
  MappedNodeEntry(NodeEntry node_entry, const int original_dtype)
      : entry(std::move(node_entry)), original_dtype(original_dtype) {
    dtype = original_dtype;
  }

  /*!
   * \brief Converts the dtype of this NodeEntry. This should be called after a node has been
   * converted and dtypes of its outputs may have changed
   */
  void UpdateDTypeAfterConversion(const int new_dtype) {
    CHECK(dtype == original_dtype || dtype == new_dtype);  // dtype should be changed only once
    CHECK(entry.node->op());
    CHECK_NE(new_dtype, -1);
    dtype = new_dtype;
  }

  /*!
   * \brief If dtype of this NodeEntry was not changed, returns the mapped entry. Otherwise returns
   * a NodeEntry to the node which casts to the original dtype of this NodeEntry.
   */
  const NodeEntry& AsOriginal() {
    return AsType(original_dtype);
  }

  /*!
   * \brief If dtype of this NodeEntry matches the specified dtype, returns the mapped entry.
   * Otherwise returns a NodeEntry to the node which casts to that type.
   */
  const NodeEntry& AsType(const int target_dtype, const bool can_add_cast = true) {
    if (dtype == target_dtype || target_dtype == -1) {
      return entry;
    }
    NodeEntry& cast_entry = casts[target_dtype];
    if (cast_entry.node == nullptr) {
      CHECK(can_add_cast);
      cast_entry = Cast(target_dtype);
      CHECK(cast_entry.node);
    }
    return cast_entry;
  }

  /*! \brief Returns whether this entry has the specified dtype or an existing cast to that dtype */
  bool HasDTypeEntry(const int target_dtype) const {
    CHECK_NE(target_dtype, -1);

    return dtype == target_dtype || casts.count(target_dtype) > 0;
  }

  /*!
   * \brief Returns whether this entry can be cast to a specific dtype. This should be called on
   * input entires of a node before its conversion.
   */
  bool CanBeCastTo(const int target_dtype) {
    CHECK_NE(target_dtype, -1);

    static const auto& amp_cast_op = Op::Get("amp_cast");
    static const auto& infertype   = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType")[amp_cast_op];
    nnvm::NodeAttrs dummy_atts;
    dummy_atts.dict["dtype"] = mxnet::op::type_string(target_dtype);
    amp_cast_op->attr_parser(&dummy_atts);

    std::vector<int> in_types  = {dtype};
    std::vector<int> out_types = {-1};
    return infertype(dummy_atts, &in_types, &out_types);
  }

  /*! \brief Returns whether this NodeEntry (of a parameter) can be cast offline */
  bool CanBeCastOfflineTo(const int target_dtype) const {
    CHECK(entry.node->is_variable());
    CHECK_NE(target_dtype, -1);

    return casts.count(target_dtype) > 0;
  }

 private:
  ObjectPtr CreateCastNode(const std::string& op_name, const std::string& node_name) {
    CHECK_GT(op_name.size(), 0);

    ObjectPtr node   = Node::Create();
    node->attrs.name = node_name;
    node->attrs.op   = Op::Get(op_name);
    node->inputs.emplace_back(entry);
    return node;
  }

  NodeEntry Cast(const int target_dtype) {
    CHECK(CanBeCastTo(target_dtype));

    const std::string dt_name        = mxnet::op::type_string(target_dtype);
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
using NodesEntries_t = std::unordered_map<Node*, NodeEntrySet_t>;
using DstNodes_t     = std::unordered_map<Node*, std::unordered_map<Node*, NodeEntry>>;

/*! \brief Makes sure the node in the new graph will work with the same precision as in the original
 * graph */
static void KeepOriginalNode(const ObjectPtr& old_node,
                             const NodeMap_t& node_map,
                             EntryMap_t* const entry_map) {
  const ObjectPtr& new_node = node_map.at(old_node.get());
  for (const auto& old_ne : old_node->inputs) {
    new_node->inputs.push_back(entry_map->at(old_ne).AsOriginal());
  }
}

/*! \brief Tries to convert the node to low precision. Returns whether the node has been
 * successfully converted
 */
static bool TryLowPrecision(const int target_dtype,
                            const ObjectPtr& old_node,
                            const NodeMap_t& node_map,
                            const NodesEntries_t& nodes_entries,
                            EntryMap_t* const entry_map) {
  static const auto& infertype      = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static const auto& fmutate_inputs = Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");

  std::vector<int> in_types(old_node->inputs.size(), -1);
  bool has_lp_input = false;
  for (int i = 0; i < old_node->inputs.size(); ++i) {
    if (entry_map->at(old_node->inputs[i]).HasDTypeEntry(target_dtype)) {
      in_types[i]  = target_dtype;
      has_lp_input = true;
    }
  }
  if (!has_lp_input) {
    // when inputs are not already in low precision, assume the first input should be in low
    // precision in order to convert this op
    in_types[0] = target_dtype;
  }

  // infer types of other inputs
  std::vector<int> out_types(old_node->num_outputs(), -1);
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

  for (size_t i = 0; i < old_node->inputs.size(); ++i) {
    MappedNodeEntry& mapped_ne = entry_map->at(old_node->inputs[i]);
    // if this tensor needs a cast, check whether MappedNodeEntry can actually cast it
    if (in_types[i] != -1 && !mapped_ne.HasDTypeEntry(in_types[i]) &&
        !mapped_ne.CanBeCastTo(in_types[i])) {
      return false;
    }
  }

  const ObjectPtr& new_node = node_map.at(old_node.get());
  for (size_t i = 0; i < old_node->inputs.size(); ++i) {
    new_node->inputs.push_back(entry_map->at(old_node->inputs[i]).AsType(in_types[i]));
  }

  for (const NodeEntry& old_ne : nodes_entries.at(old_node.get())) {
    entry_map->at(old_ne).UpdateDTypeAfterConversion(out_types[old_ne.index]);
  }

  return true;
}

/*! \brief Tries to convert the node to low precision if all of its inputs already have the correct
 * dtype. Otherwise keeps the node unchanged.
 */
static void HandleWidestDtypeNode(const int target_dtype,
                                  const ObjectPtr& old_node,
                                  const NodeMap_t& node_map,
                                  const NodesEntries_t& nodes_entries,
                                  EntryMap_t* const entry_map) {
  static const auto& infertype = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");

  // gather info about current dtypes of inputs
  // if there is already at least one input with target dtype, we try converting to low precision
  bool try_lp = false;
  std::vector<int> in_types(old_node->inputs.size(), -1);
  for (int i = 0; i < old_node->inputs.size(); ++i) {
    if (entry_map->at(old_node->inputs[i]).HasDTypeEntry(target_dtype)) {
      in_types[i] = target_dtype;  // set only lp inputs
      try_lp      = true;
    }
  }

  if (try_lp) {
    // run infertype, to see what other input types this op needs with the current lp inputs
    std::vector<int> out_types(old_node->num_outputs(), -1);
    try_lp = (infertype.count(old_node->op()) > 0 &&
              infertype[old_node->op()](old_node->attrs, &in_types, &out_types));

    if (try_lp) {
      // if we have to add casts to inputs, this op shouldn't run in low precision
      for (int i = 0; i < old_node->inputs.size(); ++i) {
        const NodeEntry& old_input_ne = old_node->inputs[i];
        if (in_types[i] != -1 && !entry_map->at(old_input_ne).HasDTypeEntry(in_types[i])) {
          try_lp = false;
          break;
        }
      }
      if (try_lp && TryLowPrecision(target_dtype, old_node, node_map, nodes_entries, entry_map)) {
        return;
      }
    }
  }
  KeepOriginalNode(old_node, node_map, entry_map);
}
/*!
 * \brief Tries to convert the node to low precision if some of its inputs already are converted.
 * Otherwise keeps the node unchanged.
 */
void HandleDTypeNeutralNode(const int target_dtype,
                            const ObjectPtr& old_node,
                            const NodeMap_t& node_map,
                            const NodesEntries_t& nodes_entries,
                            EntryMap_t* const entry_map) {
  const auto& is_lp = [&](const auto& old_ne) {
    return entry_map->at(old_ne).HasDTypeEntry(target_dtype);
  };
  if (!std::any_of(old_node->inputs.begin(), old_node->inputs.end(), is_lp) ||
      !TryLowPrecision(target_dtype, old_node, node_map, nodes_entries, entry_map)) {
    KeepOriginalNode(old_node, node_map, entry_map);
  }
}

/* \brief Decides which prameters can be cast offline and removes redundant cast nodes from the
 * graph */
static void RemoveParamCasts(const int target_dtype,
                             const std::string& offline_param_cast_attr,
                             const NodeMap_t& node_map,
                             const DstNodes_t& old_param_dst_nodes,
                             EntryMap_t* entry_map) {
  for (const auto& [old_param, old_param_dsts] : old_param_dst_nodes) {
    const ObjectPtr& new_param      = node_map.at(old_param);
    const auto& can_be_cast_offline = [&](const std::pair<Node*, NodeEntry>& old_param_dst) {
      const ObjectPtr& param_dst_node        = node_map.at(old_param_dst.first);
      const MappedNodeEntry& param_mapped_ne = entry_map->at(old_param_dst.second);
      for (const NodeEntry& node_entry : param_dst_node->inputs) {
        if (node_entry.node == new_param) {
          return false;
        }
      }
      return param_mapped_ne.CanBeCastOfflineTo(target_dtype);
    };

    if (std::all_of(old_param_dsts.begin(), old_param_dsts.end(), can_be_cast_offline)) {
      nnvm::NodeEntryEqual are_equal;
      for (const auto& [old_dst_node, old_ne] : old_param_dsts) {
        MappedNodeEntry& mapped_ne      = entry_map->at(old_ne);
        const NodeEntry& new_ne_to_skip = mapped_ne.AsType(target_dtype, false);
        const ObjectPtr& new_dst_node   = node_map.at(old_dst_node);
        bool skipped_amp_cast           = false;
        for (NodeEntry& new_ne : new_dst_node->inputs) {
          if (are_equal(new_ne, new_ne_to_skip)) {
            new_ne           = mapped_ne.entry;
            skipped_amp_cast = true;
            break;
          }
        }
        CHECK(skipped_amp_cast);
      }
      new_param->attrs.dict[offline_param_cast_attr] = mxnet::op::type_string(target_dtype);
    }
  }
}

Graph ReducePrecision(Graph&& src) {
  const auto target_dtype             = src.GetAttr<int>("target_dtype");
  const auto cast_params_offline      = src.GetAttr<int>("cast_params_offline");
  const auto& offline_param_cast_attr = src.GetAttr<std::string>("offline_param_cast_attr");
  const auto& input_names             = src.GetAttr<std::unordered_set<std::string>>("input_names");
  const auto& target_dtype_ops = src.GetAttr<std::unordered_set<std::string>>("target_dtype_ops");
  const auto& fp32_ops         = src.GetAttr<std::unordered_set<std::string>>("fp32_ops");
  const auto& widest_dtype_ops = src.GetAttr<std::unordered_set<std::string>>("widest_dtype_ops");
  auto src_dtypes              = src.GetAttr<nnvm::DTypeVector>("dtype");  // copy, not reference

  CHECK(target_dtype == mshadow::kFloat16 || target_dtype == mshadow::kBfloat16)
      << "Only float16 and bfloat16 target_dtype is supported yet," << target_dtype;

  const nnvm::IndexedGraph& src_idx = src.indexed_graph();
  CHECK_EQ(src_dtypes.size(), src_idx.num_node_entries());
  for (const int src_dtype : src_dtypes) {
    CHECK_NE(src_dtype, -1) << "Infer type failed with full information about input types";
  }

  NodeMap_t node_map;
  EntryMap_t entry_map;
  NodesEntries_t nodes_entries;
  DstNodes_t old_param_dst_nodes;

  const auto& register_node_entry =
      [&](const NodeEntry& old_ne, const ObjectPtr& old_dst_node, const ObjectPtr& new_dst_node) {
        // new_dst_node is the node that should own `old_ne` equivalent as one of its input
        const uint32_t entry_id       = src_idx.entry_id(old_ne);
        const int original_ne_dtype   = src_dtypes[entry_id];
        const ObjectPtr& old_src_node = old_ne.node;
        const ObjectPtr& new_src_node = node_map.at(old_src_node.get());
        const NodeEntry new_ne        = NodeEntry(new_src_node, old_ne.index, old_ne.version);

        entry_map.emplace(old_ne, MappedNodeEntry(new_ne, original_ne_dtype));

        // register which nodes use parameters
        nodes_entries[old_src_node.get()].insert(old_ne);
        if (new_dst_node && old_src_node->is_variable() &&
            input_names.count(old_src_node->attrs.name) == 0) {
          CHECK(old_dst_node);
          old_param_dst_nodes[old_src_node.get()][old_dst_node.get()] = old_ne;
        }
      };

  // gather information about node entries and build a new graph
  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) {
    ObjectPtr new_node = Node::Create(*old_node);
    new_node->inputs.clear();
    for (const NodeEntry& old_ne : old_node->inputs) {
      register_node_entry(old_ne, old_node, new_node);
    }
    node_map.emplace(old_node.get(), std::move(new_node));
  });
  for (const NodeEntry& old_out_ne : src.outputs) {
    register_node_entry(old_out_ne, nullptr, nullptr);
  }

  // convert the model
  const auto convert_node_fn = [&](const ObjectPtr& old_node) {
    if (old_node->is_variable() || old_node->op() == Op::Get("amp_multicast") ||
        IsCastOp(old_node->op())) {
      const ObjectPtr& new_node = node_map.at(old_node.get());
      for (const auto& old_ne : old_node->inputs) {
        const ObjectPtr& new_in_node = node_map.at(old_ne.node.get());
        new_node->inputs.emplace_back(new_in_node, old_ne.index, old_ne.version);
      }
      return;
    }
    auto opt_constraints =
        common::flag_attr_accumulate<OptConstraint_int_t>(old_node->attrs, OPT_CONSTRAINT_ATTR);
    if (fp32_ops.count(old_node->op()->name) > 0 ||
        (opt_constraints & static_cast<OptConstraint_int_t>(OptConstraint::DisableAMP))) {
      KeepOriginalNode(old_node, node_map, &entry_map);
    } else if (target_dtype_ops.count(old_node->op()->name) > 0) {
      if (!TryLowPrecision(target_dtype, old_node, node_map, nodes_entries, &entry_map)) {
        LOG(WARNING) << "Low precision conversion failure. Node '" + old_node->attrs.name +
                            "' will not be converted.";
        KeepOriginalNode(old_node, node_map, &entry_map);
      }
    } else if (widest_dtype_ops.count(old_node->op()->name) > 0) {
      HandleWidestDtypeNode(target_dtype, old_node, node_map, nodes_entries, &entry_map);
    } else {
      HandleDTypeNeutralNode(target_dtype, old_node, node_map, nodes_entries, &entry_map);
    }
  };

  // Because some nodes depend on casts present in the graph, the order of visited nodes will
  // determine whether some nodes are converted or not. To avoid this, first we make a virtual
  // conversion pass in order to have all the necessary casts already present (in the
  // MappedNodeEntry instances) during the second (true) conversion pass

  // virtual conversion pass
  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) {
    convert_node_fn(old_node);
    node_map[old_node.get()]->inputs.clear();  // make this pass "virtual" by removing edges
  });
  // true conversion pass
  DFSVisit(src.outputs, [&](const ObjectPtr& old_node) { convert_node_fn(old_node); });

  std::vector<NodeEntry> outputs;
  for (const auto& out_ne : src.outputs) {
    outputs.push_back(entry_map.at(out_ne).AsOriginal());
  }

  if (cast_params_offline) {
    RemoveParamCasts(
        target_dtype, offline_param_cast_attr, node_map, old_param_dst_nodes, &entry_map);
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
