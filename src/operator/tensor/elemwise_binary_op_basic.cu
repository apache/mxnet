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
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(elemwise_add)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_grad_add)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_backward_add)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseNoneWithHalf2<gpu, mshadow_op::identity,
                    mshadow_op::identity>);

NNVM_REGISTER_OP(elemwise_sub)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<
  gpu, op::mshadow_op::minus>);

NNVM_REGISTER_OP(_backward_sub)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseNoneWithHalf2<gpu, mshadow_op::identity,
                    mshadow_op::negation>);

NNVM_REGISTER_OP(elemwise_mul)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::mul>);

NNVM_REGISTER_OP(_backward_mul)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::right,
                    mshadow_op::left>);

NNVM_REGISTER_OP(elemwise_div)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::div>);

NNVM_REGISTER_OP(_backward_div)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::div_grad,
                    mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(_mod)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_mod)
.set_attr<FCompute>("FCompute<gpu>",
  ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
