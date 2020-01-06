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
 * \brief Use the Mixed Precision Model to infer the dtypes of
 * unknown input nodes
 */

#include <nnvm/node.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <mxnet/base.h>
#include <algorithm>
#include <functional>
#include "../common/utils.h"
#include "../operator/tensor/amp_cast.h"

namespace mxnet {
using nnvm::Graph;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using dmlc::any;
using mxnet::op::AMPCastParam;

// If a var node is not visited, visit it and set inferred_dtype_result as result_dtype,
// If already visited compare the result_dtype with existing inferred_dtype_result
static void CheckAndUpdateInferredDtypes(
    const nnvm::DTypeVector &inferred_dtypes, const nnvm::IndexedGraph &idx,
    const NodeEntry &node_entry,
    mshadow::TypeFlag result_dtype,
    std::unordered_map<std::string, mshadow::TypeFlag> *visited_vars,
    nnvm::DTypeVector *inferred_dtype_result) {
  const NodePtr &input_node = node_entry.node;
  if (!visited_vars->count(input_node->attrs.name)) {
    if ((*inferred_dtype_result)[idx.entry_id(node_entry)] == -1) {
      (*visited_vars)[input_node->attrs.name] = result_dtype;
      (*inferred_dtype_result)[idx.entry_id(node_entry)] = result_dtype;
    }
  } else {
    auto it = visited_vars->find(input_node->attrs.name);
    CHECK(it != visited_vars->end());
    if (it->second != result_dtype) {
      (*inferred_dtype_result)[idx.entry_id(node_entry)] =
          inferred_dtypes[idx.entry_id(node_entry)];
    }
  }
}

// Graph pass to infer unknown nodes which are input nodes
// as FP16 if possible
Graph AMPInferUnknown(Graph &&src) {
  const nnvm::DTypeVector &inferred_dtypes =
      src.GetAttr<nnvm::DTypeVector>("inferred_dtypes");
  const int target_dtype = src.GetAttr<int>("target_dtype");
  CHECK(target_dtype == mshadow::kFloat16)
      << "Only float16 target_dtype is supported yet";

  nnvm::DTypeVector inferred_dtype_result(inferred_dtypes);
  const nnvm::IndexedGraph &idx = src.indexed_graph();

  std::unordered_map<std::string, mshadow::TypeFlag> visited_vars;

  // Visits all nodes which are amp_cast and amp_multicast,
  // and check if inputs to these nodes are variables.
  // If input nodes are variables, set dtype for these inputs
  // and check for conflicts if an input node goes to two cast nodes
  DFSVisit(src.outputs, [&](const NodePtr &node) {
    if (!node->is_variable()) {
      std::string op_name = node->op()->name;

      if (op_name == "amp_cast") {
        // for amp_cast set inferred_dtypes for input_nodes and add
        // to visited_vars, if a var is being visited second time
        // and already has dtype set, make sure the dtype inferred again
        // is same, otherwise reset dtype to original dtype
        for (const NodeEntry &node_entry : node->inputs) {
          const NodePtr &input_node = node_entry.node;
          if (input_node->is_variable() &&
              (node->attrs.dict.find("dtype") != node->attrs.dict.end())) {
            const AMPCastParam &param =
                nnvm::get<AMPCastParam>(node->attrs.parsed);
            CHECK(param.dtype != -1)
                << "amp_cast node shouldn't have unknown dtype";
            CheckAndUpdateInferredDtypes(inferred_dtypes, idx, node_entry,
                                         static_cast<mshadow::TypeFlag>(param.dtype),
                                         &visited_vars, &inferred_dtype_result);
          }
        }
      } else if (op_name == "amp_multicast") {
        // for amp_multicast, for non var input nodes, keep track of biggest dtype.
        // If the biggest dtype is same as target_dtype, set this for the input_var nodes
        // if it is not already set
        mshadow::TypeFlag max_dtype = static_cast<mshadow::TypeFlag>(target_dtype);
        for (const NodeEntry& node_entry : node->inputs) {
          const NodePtr& input_node = node_entry.node;
          if (!input_node->is_variable()) {
            // if one input is not a variable then don't infer the dtype of other
            // input node dtypes
            max_dtype = mshadow::kFloat32;
          }
        }
        if (max_dtype == target_dtype) {
          for (const NodeEntry &node_entry : node->inputs) {
            const NodePtr &input_node = node_entry.node;
            if (input_node->is_variable()) {
              CheckAndUpdateInferredDtypes(inferred_dtypes, idx, node_entry,
                                           max_dtype, &visited_vars,
                                           &inferred_dtype_result);
            }
          }
        }
      }
    }
  });

  Graph ret;
  ret.attrs["inferred_dtype_result"] =
      std::make_shared<dmlc::any>(std::move(inferred_dtype_result));
  ret.outputs = std::move(src.outputs);
  return ret;
}

NNVM_REGISTER_PASS(AMPInferUnknown)
    .describe("Infer dtypes of different nodes for the mixed precision model")
    .set_body(AMPInferUnknown)
    .set_change_graph(true)
    .provide_graph_attr("inferred_dtypes");
}  // namespace mxnet
