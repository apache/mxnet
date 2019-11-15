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
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_broadcast_op_extended.cu
 * \brief GPU Implementation of extended functions for elementwise binary broadcast operator.
 */

#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_copysign)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::copysign>);

NNVM_REGISTER_OP(_npi_lcm)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastIntCompute<gpu, mshadow_op::lcm>);

NNVM_REGISTER_OP(_npi_bitwise_xor)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastIntCompute<gpu, mshadow_op::bitwise_xor>);

NNVM_REGISTER_OP(_backward_npi_copysign)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::copysign_grad,
                                                                  mshadow_op::copysign_rgrad>);

NNVM_REGISTER_OP(_npi_arctan2)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::arctan2>);

NNVM_REGISTER_OP(_backward_npi_arctan2)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::arctan2_grad,
                                                                  mshadow_op::arctan2_rgrad>);
NNVM_REGISTER_OP(_npi_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::hypot>);

NNVM_REGISTER_OP(_backward_npi_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::hypot_grad_left,
                                                                  mshadow_op::hypot_grad_right>);
NNVM_REGISTER_OP(_npi_copysign_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::copysign>);

NNVM_REGISTER_OP(_npi_rcopysign_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rcopysign>);

NNVM_REGISTER_OP(_backward_npi_copysign_scalar)
.set_attr<FCompute>("FCompute<gpu>",
                    BinaryScalarOp::Backward<gpu, mshadow_op::copysign_grad>);

NNVM_REGISTER_OP(_backward_npi_rcopysign_scalar)
.set_attr<FCompute>("FCompute<gpu>",
                    BinaryScalarOp::Backward<gpu, mshadow_op::rcopysign_grad>);

NNVM_REGISTER_OP(_npi_arctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::arctan2>);

NNVM_REGISTER_OP(_backward_npi_arctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::arctan2_grad>);

NNVM_REGISTER_OP(_npi_rarctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rarctan2>);

NNVM_REGISTER_OP(_backward_npi_rarctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rarctan2_grad>);

NNVM_REGISTER_OP(_npi_lcm_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::ComputeInt<gpu, mshadow_op::lcm>);

NNVM_REGISTER_OP(_npi_bitwise_xor_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::ComputeInt<gpu, mshadow_op::bitwise_xor>);

NNVM_REGISTER_OP(_npi_ldexp)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::ldexp>);

NNVM_REGISTER_OP(_npi_ldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::ldexp>);

NNVM_REGISTER_OP(_npi_rldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rldexp>);

NNVM_REGISTER_OP(_backward_npi_ldexp)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::ldexp_grad,
                                                                  mshadow_op::ldexp_rgrad>);

NNVM_REGISTER_OP(_backward_npi_ldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<gpu, mshadow_op::ldexp_grad>);

NNVM_REGISTER_OP(_backward_npi_rldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<gpu, mshadow_op::rldexp_grad>);

}  // namespace op
}  // namespace mxnet
