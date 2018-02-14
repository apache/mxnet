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
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op-inl.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(elemwise_add, op::mshadow_op::plus)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_add)
.add_alias("_add").add_alias("_plus").add_alias("_Plus")
.describe(R"code(Adds arguments element-wise.

The storage type of ``elemwise_add`` output depends on storage types of inputs

   - elemwise_add(row_sparse, row_sparse) = row_sparse
   - elemwise_add(csr, csr) = csr
   - otherwise, ``elemwise_add`` generates output with default storage

)code")
.set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_add"});

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_grad_add, op::mshadow_op::plus);

NNVM_REGISTER_OP(_backward_add)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseNone<
  cpu, mshadow_op::identity, mshadow_op::identity>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      ElemwiseBinaryOp::BackwardUseNoneEx<cpu, mshadow_op::identity,
                      mshadow_op::identity>)
.set_attr<FInferStorageType>("FInferStorageType",
                             ElemwiseStorageType<1, 2, true, true, true>);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(elemwise_sub, op::mshadow_op::minus)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_sub)
.add_alias("_sub").add_alias("_minus").add_alias("_Minus")
.describe(R"code(Subtracts arguments element-wise.

The storage type of ``elemwise_sub`` output depends on storage types of inputs

   - elemwise_sub(row_sparse, row_sparse) = row_sparse
   - elemwise_sub(csr, csr) = csr
   - otherwise, ``elemwise_sub`` generates output with default storage

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sub"});

NNVM_REGISTER_OP(_backward_sub)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseNone<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BackwardUseNoneEx<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FInferStorageType>("FInferStorageType",
                             ElemwiseStorageType<1, 2, true, true, true>);

MXNET_OPERATOR_REGISTER_BINARY(elemwise_mul)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_mul)
.describe(R"code(Multiplies arguments element-wise.

The storage type of ``elemwise_mul`` output depends on storage types of inputs

   - elemwise_mul(default, default) = default
   - elemwise_mul(row_sparse, row_sparse) = row_sparse
   - elemwise_mul(default, row_sparse) = default
   - elemwise_mul(row_sparse, default) = default
   - elemwise_mul(csr, csr) = csr
   - otherwise, ``elemwise_mul`` generates output with default storage

)code")
.set_attr<FInferStorageType>("FInferStorageType",
                             ElemwiseBinaryOp::AllowLRDenseInputWithSparseOutputStorageType<
                               false, false>)  // 0 * nan or nan * 0 -> nan, so rsp * dns -> dns
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      ElemwiseBinaryOp::ComputeDnsLRValueEx<cpu, op::mshadow_op::mul, true, true>)
.set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */
                              [](const NodeAttrs& attrs) {
                                return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                              })
.add_alias("_mul").add_alias("_Mul")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseBinaryOp::BackwardUseInStorageType)
.set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */
                              [](const NodeAttrs& attrs) {
                                return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                              })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::right, mshadow_op::left>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BackwardUseInEx<
  cpu, mshadow_op::right, mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(elemwise_div, op::mshadow_op::div)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_div)
.describe(R"code(Divides arguments element-wise.

The storage type of ``elemwise_div`` output is always dense

)code")
.add_alias("_div").add_alias("_Div")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::div_grad, mshadow_op::div_rgrad>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BackwardUseInEx<
  cpu, mshadow_op::div_grad, mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_mod)
.add_alias("_Mod")
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod"});

NNVM_REGISTER_OP(_backward_mod)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
