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
 * Copyright (c) 2018 by Contributors
 * \file gradient_multiplier_op.cc
 * \brief
 * \author Istvan Fehervari
*/
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

static bool BinaryScalarStorageType(const nnvm::NodeAttrs& attrs,
                                    const int dev_mask,
                                    DispatchMode* dispatch_mode,
                                    std::vector<int> *in_attrs,
                                    std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const auto in_stype = in_attrs->at(0);
  auto &out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && (in_stype == kDefaultStorage)) {
    // dense -> dense
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && in_stype == kRowSparseStorage) {
    // row sparse -> row sparse
    dispatched = storage_type_assign(&out_stype, kRowSparseStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
    // FComputeEx can handle dns output on cpu, too
    if (dev_mask == cpu::kDevMask && out_stype == kDefaultStorage) {
      DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
      dispatched = true;
    }
  }
  if (!dispatched && in_stype == kCSRStorage) {
    // csr -> csr
    dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
    // FComputeEx can handle dns output on cpu, too
    if (dev_mask == cpu::kDevMask && out_stype == kDefaultStorage) {
      DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
      dispatched = true;
    }
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

MXNET_OPERATOR_REGISTER_UNARY(_contrib_gradientmultiplier)
.describe(R"code(This operator implements the gradient multiplier function.
In forward pass it acts as an identity transform. During backpropagation it
multiplies the gradient from the subsequent level by a scalar factor lambda and passes it to
the preceding layer.
)code" ADD_FILELINE)
.set_attr_parser([](NodeAttrs* attrs) {
    attrs->parsed = std::stod(attrs->dict["scalar"]);
  })
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1, false, true, true>)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_contrib_backward_gradientmultiplier"})
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.add_argument("scalar", "float", "lambda multiplier");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_contrib_backward_gradientmultiplier)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", BinaryScalarStorageType)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BinaryScalarOp::ComputeEx<cpu, op::mshadow_op::mul>);

}  // namespace op
}  // namespace mxnet
