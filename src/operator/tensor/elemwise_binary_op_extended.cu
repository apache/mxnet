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

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_power)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<gpu, mshadow_op::power>);

NNVM_REGISTER_OP(_backward_power)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BackwardUseIn<gpu,
  mshadow_op::power_grad, mshadow_op::power_rgrad>);

NNVM_REGISTER_OP(_maximum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<gpu, mshadow_op::maximum>);

NNVM_REGISTER_OP(_backward_maximum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BackwardUseIn<gpu, mshadow_op::ge,
  mshadow_op::lt>);

NNVM_REGISTER_OP(_minimum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<gpu, mshadow_op::minimum>);

NNVM_REGISTER_OP(_backward_minimum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BackwardUseIn<gpu, mshadow_op::le,
  mshadow_op::gt>);

NNVM_REGISTER_OP(_hypot)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Compute<gpu, mshadow_op::hypot>);

NNVM_REGISTER_OP(_backward_hypot)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BackwardUseIn<gpu,
  mshadow_op::hypot_grad_left, mshadow_op::hypot_grad_right>);

}  // namespace op
}  // namespace mxnet
