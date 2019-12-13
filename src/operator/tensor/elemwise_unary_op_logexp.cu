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
 * \file elemwise_unary_op_logexp.cu
 * \brief GPU Implementation of unary log and exp functions.
 */
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {

// exp
NNVM_REGISTER_OP(exp)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::exp>);

// log
NNVM_REGISTER_OP(log)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log>);

// log10
NNVM_REGISTER_OP(log10)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log10>);

// log2
NNVM_REGISTER_OP(log2)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log2>);

NNVM_REGISTER_OP(_backward_log)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log_grad> >);

NNVM_REGISTER_OP(_backward_log10)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log10_grad> >);

NNVM_REGISTER_OP(_backward_log2)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log2_grad> >);

// log1p
NNVM_REGISTER_OP(log1p)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::log1p>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::log1p>);

NNVM_REGISTER_OP(_backward_log1p)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::log1p_grad> >);

// expm1
NNVM_REGISTER_OP(expm1)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::Compute<gpu, mshadow_op::expm1>)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::ComputeEx<gpu, mshadow_op::expm1>);

NNVM_REGISTER_OP(_backward_expm1)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<
  gpu, unary_bwd<mshadow_op::exp> >);

}  // namespace op
}  // namespace mxnet
