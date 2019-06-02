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
 * \file elemwise_binary_scalar_op_logic.cc
 * \brief CPU Implementation of binary scalar logic functions.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_BINARY_SCALAR_LOGIC(__name$, __kernel$)                      \
  MXNET_OPERATOR_REGISTER_BINARY_SCALAR(__name$)                                             \
  .set_attr<FInferStorageType>("FInferStorageType", BinaryScalarLogicStorageType<__kernel$>) \
  .set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<cpu>", BinaryScalarOp::LogicComputeEx<cpu, __kernel$>)

template<typename OP>
static bool BinaryScalarLogicStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const auto in_stype = in_attrs->at(0);
  auto &out_stype = out_attrs->at(0);
  bool dispatched = false;
  const double alpha = nnvm::get<double>(attrs.parsed);
  bool is_sparse = OP::Map(static_cast<double>(0), alpha) == 0;
  if (!dispatched && in_stype == kDefaultStorage) {
    // dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && in_stype == kRowSparseStorage && is_sparse) {
    // rsp -> rsp
    dispatched = storage_type_assign(&out_stype, kRowSparseStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched && in_stype == kCSRStorage && is_sparse) {
    // csr -> csr
    dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}


MXNET_OPERATOR_REGISTER_BINARY_SCALAR_LOGIC(_equal_scalar, mshadow_op::eq)
.add_alias("_npi_equal_scalar")
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_EqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR_LOGIC(_not_equal_scalar, mshadow_op::ne)
.add_alias("_npi_not_equal_scalar")
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_NotEqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR_LOGIC(_greater_scalar, mshadow_op::gt)
.add_alias("_npi_greater_scalar")
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_GreaterScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR_LOGIC(_greater_equal_scalar, mshadow_op::ge)
.add_alias("_npi_greater_equal_scalar")
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_GreaterEqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR_LOGIC(_lesser_scalar, mshadow_op::lt)
.add_alias("_npi_less_scalar")
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LesserScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR_LOGIC(_lesser_equal_scalar, mshadow_op::le)
.add_alias("_npi_less_equal_scalar")
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LesserEqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_logical_and_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::logical_and>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LogicalAndScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_logical_or_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::logical_or>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LogicalOrScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_logical_xor_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::logical_xor>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LogicalXorScalar");

}  // namespace op
}  // namespace mxnet
