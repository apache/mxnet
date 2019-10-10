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
 * \file ir.cc
 * \author Junru Shao
 */
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include <nnvm/graph.h>

#include "../../../include/bridge/legacy_nnvm.h"
#include "../../../include/ir.h"
#include "../../../include/op/attrs/nn.h"

namespace mxnet {
namespace v3 {
namespace bridge {
namespace legacy_nnvm {

using ir::Array;
using ir::CallNode;
using ir::Expr;
using ir::Function;
using ir::FunctionNode;
using ir::LetNode;
using ir::NodeRef;
using ir::TupleGetItemNode;
using ir::TupleNode;
using ir::Var;
using ir::VarNode;

NNVMCapsule NNVMCapsule::make(const nnvm::NodeAttrs &attrs) {
  auto node = ir::make_node<NNVMCapsuleNode>();
  node->attrs = attrs;
  return NNVMCapsule(node);
}

static void PrintIndexedGraph(const nnvm::Graph &g) {
  const auto &idx = g.indexed_graph();
  std::unordered_set<int> input_nodes(idx.input_nodes().begin(),
                                      idx.input_nodes().end());
  std::cout << idx.num_nodes() << " nodes, " << input_nodes.size()
            << " input nodes" << std::endl;
  int n_nodes = idx.num_nodes();
  for (int i = 0, input_cnt = 0; i < n_nodes; ++i) {
    const nnvm::Node *node = idx[i].source;
    const nnvm::Op *op = node->op();
    std::string op_name = op ? op->name : "None";
    if (input_nodes.count(i)) {
      input_cnt += 1;
      op_name = (op ? op->name + " [input " : "[input ") +
                std::to_string(input_cnt) + "]";
    } else {
      op_name = op ? op->name : "None";
    }
    std::cout << "  i = " << i << ", op = " << op_name
              << ", #(input node entries) = " << idx[i].inputs.size()
              << std::endl;
    int j_cnt = 0;
    for (const auto &attr : node->attrs.dict) {
      std::cout << "    " << attr.first << " = " << attr.second << std::endl;
    }
    for (const nnvm::IndexedGraph::NodeEntry &j : idx[i].inputs) {
      std::cout << "    input entry #" << ++j_cnt
                << ", entry_id = " << idx.entry_id(j)
                << ", (node_id = " << j.node_id << ", index = " << j.index
                << ", version = " << j.version << ")" << std::endl;
    }
    for (int j_cnt = 0, n_out = node->num_outputs(); j_cnt < n_out; ++j_cnt) {
      uint32_t entry_id = idx.entry_id(i, j_cnt);
      std::cout << "    output entry #" << j_cnt + 1
                << ", entry_id = " << entry_id << std::endl;
    }
  }
  std::cout << idx.outputs().size() << " output node entries: " << std::endl;
  int j_cnt = 0;
  for (const nnvm::IndexedGraph::NodeEntry &j : idx.outputs()) {
    std::cout << "  output entry #" << ++j_cnt
              << ", entry_id = " << idx.entry_id(j)
              << ", (node_id = " << j.node_id << ", index = " << j.index
              << ", version = " << j.version << ")" << std::endl;
  }
}

Function NNVMToRelay(const nnvm::Graph &g) {
  PrintIndexedGraph(g);
  const auto &idx = g.indexed_graph();
  int n_nodes = idx.num_nodes();
  // maps: node -> var
  std::vector<Var> node2var(n_nodes);
  // maps: (node, output_index) -> var
  std::vector<std::vector<Var>> entries(n_nodes);
  // maps: node -> #outputs of the node
  std::vector<int> n_outputs(n_nodes);
  for (int node_id = 0, input_cnt = 0, compute_cnt = 0; node_id < n_nodes;
       ++node_id) {
    const nnvm::Node *node = idx[node_id].source;
    int n_out = node->num_outputs();
    n_outputs[node_id] = n_out;
    std::string name = node->is_variable()
                           ? "arg_" + std::to_string(++input_cnt)
                           : "x_" + std::to_string(++compute_cnt);
    Var var = node2var[node_id] = VarNode::make(name, {});
    std::vector<Var> &outputs = entries[node_id];
    if (n_out == 1) {
      outputs.push_back(var);
    } else {
      outputs.reserve(n_out);
      for (int i = 0; i < n_out; ++i) {
        outputs.push_back(VarNode::make(name + "#" + std::to_string(i), {}));
      }
    }
  }
  // Create the let list
  std::vector<std::pair<Var, Expr>> let_list;
  for (int node_id = 0; node_id < n_nodes; ++node_id) {
    const Var &var = node2var[node_id];
    const nnvm::IndexedGraph::Node &node = idx[node_id];
    int n_out = n_outputs[node_id];
    const auto &src = node.source;
    if (src->is_variable()) {
      CHECK_EQ(n_out, 1) << "InternalError: internal assumption violation";
      continue;
    }
    // Create call_args
    Array<Expr> call_args;
    for (const nnvm::IndexedGraph::NodeEntry &input : node.inputs) {
      CHECK_LT((int)input.node_id, node_id)
          << "InternalError: IndexedGraph is not topo-sorted";
      call_args.push_back(entries[input.node_id][input.index]);
    }
    // Add a CallNode
    let_list.push_back({var, ConvertCall(src->op(), src->attrs, call_args)});
    // Add logic for de-tuple
    if (n_out > 1) {
      for (int index = 0; index < n_out; ++index) {
        let_list.push_back(std::make_pair(entries[node_id][index],
                                          TupleGetItemNode::make(var, index)));
      }
    }
  }
  // Find input arguments to the function
  Array<Var> params;
  for (int node_id = 0; node_id < n_nodes; ++node_id) {
    const nnvm::Node *node = idx[node_id].source;
    if (node->is_variable()) {
      params.push_back(node2var[node_id]);
    }
  }
  // Find outputs of the function
  Expr body;
  {
    // 1) Find outputs
    Array<Expr> outputs;
    for (const nnvm::IndexedGraph::NodeEntry &j : idx.outputs()) {
      outputs.push_back(entries[j.node_id][j.index]);
    }
    CHECK(!outputs.empty()) << "InternalError: NNVM graph has no output";
    if (outputs.size() == 1) {
      body = outputs[0];
    } else {
      body = TupleNode::make(std::move(outputs));
    }
    // 2) Construct the body out of let-list
    for (; !let_list.empty(); let_list.pop_back()) {
      const std::pair<Var, Expr> &last = let_list.back();
      body = LetNode::make(last.first, last.second, body);
    }
  }
  // Then we are able to construct the function
  return FunctionNode::make(std::move(params), std::move(body), {}, {}, {});
}

}  // namespace legacy_nnvm
}  // namespace bridge
}  // namespace v3
}  // namespace mxnet
#endif
