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
 *  Copyright (c) 2021 by Contributors
 * \file quantize_graph_pass.h
 * \brief
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZE_GRAPH_PASS_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZE_GRAPH_PASS_H_

#include <mxnet/op_attr_types.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include "quantize_v2-inl.h"
#include "../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::ObjectPtr;
using nnvm::NodeEntry;
using nnvm::Graph;

inline ObjectPtr CreateNode(std::string op_name, std::string node_name) {
  ObjectPtr node = Node::Create();
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

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZE_GRAPH_PASS_H_
