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
 * \file ir.h
 * \author Junru Shao
 */
#pragma once
#if MXNET_USE_TVM_OP && !defined MXNET_AMALGAMATION
// This is a compatibility layer between MXNet v3 and Relay
// We will borrow basically everything from TVM/Relay to here.

#include <tvm/attrs.h>
#include <tvm/ir.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/node/container.h>
#include <tvm/node/memory.h>
#include <tvm/node/node.h>
#include <tvm/relay/base.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/module.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/type.h>

namespace mxnet {
namespace v3 {
namespace ir {

using tvm::Array;
using tvm::Attrs;
using tvm::AttrsNode;
using tvm::Downcast;
using tvm::GetRef;
using tvm::Integer;
using tvm::IntImm;
using tvm::make_node;
using tvm::Map;
using tvm::MapNode;
using tvm::Node;
using tvm::NodePtr;
using tvm::NullValue;

using tvm::relay::DataType;
using tvm::relay::IndexExpr;
using tvm::relay::NodeEqual;
using tvm::relay::NodeHash;
using tvm::relay::NodeRef;

// Relay Expression
using tvm::relay::Expr;
using tvm::relay::ExprNode;

using tvm::relay::FTVMCompute;
using tvm::relay::FTVMSchedule;
using tvm::relay::TOpPattern;
using tvm::relay::Op;
using tvm::relay::OpNode;

using tvm::relay::Tuple;
using tvm::relay::TupleNode;

using tvm::relay::Var;
using tvm::relay::VarNode;

using tvm::relay::GlobalVar;
using tvm::relay::GlobalVarNode;

using tvm::relay::Function;
using tvm::relay::FunctionNode;

using tvm::relay::Call;
using tvm::relay::CallNode;

using tvm::relay::Let;
using tvm::relay::LetNode;

using tvm::relay::If;
using tvm::relay::IfNode;

using tvm::relay::TupleGetItem;
using tvm::relay::TupleGetItemNode;

using tvm::relay::RefCreate;
using tvm::relay::RefCreateNode;

using tvm::relay::RefRead;
using tvm::relay::RefReadNode;

using tvm::relay::RefWrite;
using tvm::relay::RefWriteNode;

using tvm::relay::TempExpr;
using tvm::relay::TempExprNode;

// Relay Types
using tvm::relay::Kind;

using tvm::relay::Type;
using tvm::relay::TypeNode;

using tvm::relay::BaseTensorType;
using tvm::relay::BaseTensorTypeNode;

using tvm::relay::TensorType;
using tvm::relay::TensorTypeNode;

using tvm::relay::TypeVar;
using tvm::relay::TypeVarNode;

using tvm::relay::GlobalTypeVar;
using tvm::relay::GlobalTypeVarNode;

using tvm::relay::TypeCall;
using tvm::relay::TypeCallNode;

using tvm::relay::IncompleteType;
using tvm::relay::IncompleteTypeNode;

using tvm::relay::FuncType;
using tvm::relay::FuncTypeNode;

using tvm::relay::TupleType;
using tvm::relay::TupleTypeNode;

using tvm::relay::RefType;
using tvm::relay::RefTypeNode;

using tvm::relay::TypeConstraint;
using tvm::relay::TypeConstraintNode;

using tvm::relay::TypeRelation;
using tvm::relay::TypeRelationNode;

using tvm::relay::TypeReporter;

// Relay Functors
using tvm::relay::ExprFunctor;

}  // namespace ir
}  // namespace v3
}  // namespace mxnet

#define MX_V3_DEF_NODE_TYPE_INFO(TypeName, Parent) TVM_DECLARE_NODE_TYPE_INFO(TypeName, Parent)

#define MX_V3_DEF_BASE_NODE_INFO(TypeName, Parent) TVM_DECLARE_BASE_NODE_INFO(TypeName, Parent)

#define MX_V3_DEF_NODE_REF_METHODS(TypeName, BaseTypeName, NodeName)     \
  TypeName() {                                                         \
  }                                                                    \
  explicit TypeName(::tvm::NodePtr<::tvm::Node> n) : BaseTypeName(n) { \
  }                                                                    \
  NodeName* operator->() const {                                       \
    return static_cast<NodeName*>(node_.get());                        \
  }                                                                    \
  operator bool() const {                                              \
    return this->defined();                                            \
  }                                                                    \
  using ContainerType = NodeName;

#define MX_V3_DECLARE_ATTRS TVM_DECLARE_ATTRS

#define MX_V3_ATTR_FIELD TVM_ATTR_FIELD

#define MX_V3_REGISTER_NODE_TYPE TVM_REGISTER_NODE_TYPE

#define MX_V3_REGISTER_OP RELAY_REGISTER_OP

#define MX_V3_ADD_FILELINE TVM_ADD_FILELINE
#endif
