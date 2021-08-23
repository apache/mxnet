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
 * \file node.h
 * \brief Definitions and helper macros for IR/AST nodes.
 *
 *  The node folder contains base utilities for IR/AST nodes,
 *  invariant of which specific language dialect.
 *
 *  We implement AST/IR nodes as sub-classes of runtime::Object.
 *  The base class Node is just an alias of runtime::Object.
 *
 *  Besides the runtime type checking provided by Object,
 *  node folder contains additional functionalities such as
 *  reflection and serialization, which are important features
 *  for building a compiler infra.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_NODE_NODE_H_
#define MXNET_NODE_NODE_H_

#include <mxnet/runtime/c_runtime_api.h>
#include <mxnet/runtime/object.h>
#include <mxnet/runtime/memory.h>

#include <string>
#include <vector>
#include <utility>
#include <type_traits>

namespace mxnet {

using runtime::TypeIndex;
using runtime::Object;
// We strictly restrict ObjectPtr to ::mxnet::runtime
// as it may conflict with ::nnvm::ObjectPtr
// using runtime::ObjectPtr;
using runtime::ObjectRef;
using runtime::GetRef;
using runtime::Downcast;
using runtime::ObjectHash;
using runtime::ObjectEqual;
using runtime::make_object;

}  // namespace mxnet

#endif  // MXNET_NODE_NODE_H_
