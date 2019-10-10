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
 * \file legacy_nnvm.h
 * \author Junru Shao
 */
#pragma once
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
#include <nnvm/node.h>

#include "../ir.h"

namespace nnvm {
class Op;
class Graph;
}  // namespace nnvm

namespace mxnet {
namespace v3 {
namespace bridge {
namespace legacy_nnvm {

class NNVMCapsuleNode final : public ir::Node {
 public:
  nnvm::NodeAttrs attrs;
  void VisitAttrs(tvm::AttrVisitor *v) final {}
  static constexpr const char *_type_key = "mxnet.v3.bridge.NNVMCapsule";
  MX_V3_DEF_NODE_TYPE_INFO(NNVMCapsuleNode, ir::Node);
};

class NNVMCapsule final : public ir::NodeRef {
 public:
  MX_V3_DEF_NODE_REF_METHODS(NNVMCapsule, ir::NodeRef, NNVMCapsuleNode);
  static NNVMCapsule make(const nnvm::NodeAttrs &attrs);
};

ir::Call ConvertCall(const nnvm::Op *op, const nnvm::NodeAttrs &attrs,
                     const ir::Array<ir::Expr> &args);

ir::Function NNVMToRelay(const nnvm::Graph &g);

}  // namespace legacy_nnvm
}  // namespace bridge
}  // namespace v3
}  // namespace mxnet
#endif
