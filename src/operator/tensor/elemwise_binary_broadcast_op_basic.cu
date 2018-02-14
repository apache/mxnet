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
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(broadcast_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_backward_broadcast_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseNone<gpu, mshadow_op::identity,
                                                                mshadow_op::identity>);

NNVM_REGISTER_OP(broadcast_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, op::mshadow_op::minus>);

NNVM_REGISTER_OP(_backward_broadcast_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseNone<gpu, mshadow_op::identity,
                                                                mshadow_op::negation>);

NNVM_REGISTER_OP(broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, op::mshadow_op::mul>);

NNVM_REGISTER_OP(_backward_broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::right,
                                                                mshadow_op::left>);

NNVM_REGISTER_OP(broadcast_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, op::mshadow_op::div>);

NNVM_REGISTER_OP(_backward_broadcast_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::div_grad,
                                                                mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(broadcast_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_broadcast_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::mod_grad,
                                                                  mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
