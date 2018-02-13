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
#include "./elemwise_binary_op.h"


namespace mxnet {
namespace op {

bool ElemwiseBinaryOp::SparseSparseWithDenseResult(const nnvm::NodeAttrs& attrs,
                                                   const int dev_mask,
                                                   DispatchMode* dispatch_mode,
                                                   std::vector<int> *in_attrs,
                                                   std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), 1U) << " in operator " << attrs.name;
  const auto& lhs_stype = in_attrs->at(0);
  const auto& rhs_stype = in_attrs->at(1);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  const bool invalid_ctx = dev_mask != mshadow::cpu::kDevMask;
  const auto dispatch_ex = invalid_ctx ?
                           DispatchMode::kFComputeFallback : DispatchMode::kFComputeEx;
  if (!dispatched && (lhs_stype == kDefaultStorage || rhs_stype == kDefaultStorage)) {
    // dns, dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && lhs_stype == kRowSparseStorage && rhs_stype == kRowSparseStorage) {
    // rsp, rsp -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode, dispatch_ex);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

bool ElemwiseBinaryOp::BackwardUseInStorageType(const nnvm::NodeAttrs& attrs,
                                                const int dev_mask,
                                                DispatchMode* dispatch_mode,
                                                std::vector<int> *in_attrs,
                                                std::vector<int> *out_attrs) {
  using namespace common;
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 2U);
  bool dispatched = false;
  const bool invalid_ctx = dev_mask != mshadow::cpu::kDevMask;
  const auto dispatch_ex = invalid_ctx ? DispatchMode::kFComputeFallback :
                           DispatchMode::kFComputeEx;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    if (common::ContainsOnlyStorage(*in_attrs, kRowSparseStorage)
      && common::ContainsOnlyStorage(*out_attrs, kRowSparseStorage)) {
      dispatched = storage_type_assign(out_attrs, kRowSparseStorage,
                                       dispatch_mode, dispatch_ex);
    }
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

}  // namespace op
}  // namespace mxnet
