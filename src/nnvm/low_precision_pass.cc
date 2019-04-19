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

NodePtr CreateNode(std::string op_name, std::string node_name) {
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

NodePtr InsertNode(std::string op_name, std::string node_name, NodePtr current,
                   NodeEntry previous) {
  NodePtr node = CreateNode(op_name, node_name);
  node->inputs.emplace_back(previous);
  current->inputs.emplace_back(NodeEntry{node, 0, 0});
  return node;
}

std::string GetSuffix(const nnvm::NodeEntry &e,
                      const std::unordered_map<Node*, NodePtr>& mirror_map) {
  static const auto &flist_outputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  std::string suffix = "";
  NodePtr mirror_node = mirror_map.at(e.node.get());
  if (mirror_node->op() != nullptr) {
    auto list_output_names_func = flist_outputs.get(e.node->op(), nullptr);
    if (list_output_names_func != nullptr) {
      std::vector<std::string> names = list_output_names_func(e.node->attrs);
      suffix = "_" + names[e.index];
    } else {
      suffix = "_" + std::to_string(e.index);
    }
  }
  return suffix;
}

void AddCastNode(const nnvm::NodeEntry &e, const std::string &suffix,
                 const nnvm::NodeEntry &input,
                 const std::string dtype,
                 nnvm::NodeEntryMap<NodeEntry>* mirror_entry_map,
                 NodePtr curr_node) {
  NodePtr cast_node =
      InsertNode("amp_cast", e.node->attrs.name + suffix + "_cast", curr_node,
                 input);
  cast_node->attrs.dict["dtype"] = dtype;
  cast_node->op()->attr_parser(&(cast_node->attrs));
  mirror_entry_map[e] = NodeEntry{cast_node, 0, e.version};
  return;
}

void AddMultiCastNode(const std::vector<NodeEntry>& inputs, const std::string& node_name,
                      const std::unordered_map<Node*, NodePtr>& mirror_map,
                      nnvm::NodeEntryMap<NodeEntry>* mirror_entry_map,
                      NodePtr curr_node) {
  NodePtr node = CreateNode("amp_multicast", node_name);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &e = inputs[i];
    NodePtr mirror_node = mirror_map.at(e.node.get());
    NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
    node->inputs.emplace_back(mirror_entry);
  }
  node->attrs.dict["num_outputs"] = std::to_string(inputs.size());
  node->op()->attr_parser(&(node->attrs));
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    const auto &e = inputs[i];
    curr_node->inputs.emplace_back(NodeEntry{node, static_cast<uint32_t>(i), e.version});
  }
  return;
}

Graph ReducePrecision(Graph&& src) {
  static const auto &flist_outputs = nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  const auto target_dtype_ops =
      src.GetAttr<std::unordered_set<std::string>>("target_dtype_ops");
  const auto fp32_ops =
      src.GetAttr<std::unordered_set<std::string>>("fp32_ops");
  const auto widest_dtype_ops =
      src.GetAttr<std::unordered_set<std::string>>("widest_dtype_ops");
  const auto conditional_fp32_ops =
      src.GetAttr<std::unordered_set<std::string>>("conditional_fp32_ops");
  const auto target_dtype = src.GetAttr<int>("target_dtype");
  CHECK(target_dtype == mshadow::kFloat16) << "Only float16 target dtype is supported";
  std::unordered_map<Node*, NodePtr> mirror_map;
  nnvm::NodeEntryMap<NodeEntry> mirror_entry_map;
  DFSVisit(src.outputs, [&](const NodePtr &node) {
    NodePtr new_node = Node::Create();
    *new_node = *node;
    new_node->inputs.clear();
    if (!node->is_variable() && fp32_ops.count(node->op()->name) > 0) {
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const auto &e = node->inputs[i];
        if (mirror_entry_map.count(e)) {
          new_node->inputs.emplace_back(mirror_entry_map[e]);
        } else {
          NodePtr mirror_node = mirror_map.at(e.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
          std::string suffix = GetSuffix(e, mirror_map);
          AddCastNode(e, suffix, mirror_entry, "float32", &mirror_entry_map, new_node);
        }
      }
    } else if (!node->is_variable() && target_dtype_ops.count(node->op()->name) > 0) {
      for (size_t i = 0; i < node->inputs.size(); ++i) {
        const auto &e = node->inputs[i];
        if (mirror_entry_map.count(e)) {
          new_node->inputs.emplace_back(mirror_entry_map[e]);
        } else {
          NodePtr mirror_node = mirror_map.at(e.node.get());
          NodeEntry mirror_entry = NodeEntry{mirror_node, e.index, e.version};
          std::string suffix = GetSuffix(e, mirror_map);
          AddCastNode(e, suffix, mirror_entry, "float16", &mirror_entry_map, new_node);
        }
      }
    } else if (!node->is_variable() && widest_dtype_ops.count(node->op()->name) > 0) {
        CHECK(node->inputs.size() > 0)
            << "Please check the symbol. node name " << node << node->attrs.name
            << "op name " << node->op()->name << "has no inputs";
        const auto &e = node->inputs[0];
        std::string suffix = GetSuffix(e, mirror_map);
        AddMultiCastNode(node->inputs, suffix, mirror_map, &mirror_entry_map, new_node);
    }
    mirror_map[node.get()] = std::move(new_node);
  });

  std::vector<NodeEntry> outputs;
  for (const auto& e : src.outputs) {
    outputs.emplace_back(NodeEntry{mirror_map.at(e.node.get()), e.index, e.version});
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
