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
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_plus_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_minus_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::minus>);

NNVM_REGISTER_OP(_rminus_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rminus>);

NNVM_REGISTER_OP(_mul_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, op::mshadow_op::mul>);

NNVM_REGISTER_OP(_backward_mul_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, op::mshadow_op::mul>);

NNVM_REGISTER_OP(_div_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::div>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, op::mshadow_op::div>);

NNVM_REGISTER_OP(_backward_div_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::div>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, op::mshadow_op::div>);

NNVM_REGISTER_OP(_rdiv_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rdiv>);

NNVM_REGISTER_OP(_backward_rdiv_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<gpu,
  mshadow_op::rdiv_grad>);

NNVM_REGISTER_OP(_mod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_mod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<
  gpu, mshadow_op::mod_grad>);

NNVM_REGISTER_OP(_rmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rmod>);

NNVM_REGISTER_OP(_backward_rmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<
  gpu, mshadow_op::rmod_grad>);

}  // namespace op
}  // namespace mxnet
