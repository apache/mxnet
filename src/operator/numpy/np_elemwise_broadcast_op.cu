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
 * \file np_elemwise_broadcast_op.cu
 * \brief GPU Implementation of basic functions for elementwise binary broadcast operator.
 */
#include "./np_elemwise_broadcast_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_npi_subtract)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, op::mshadow_op::minus>);

NNVM_REGISTER_OP(_npi_multiply)
.set_attr<FCompute>(
  "FCompute<gpu>",
  MixedBinaryBroadcastCompute<gpu, op::mshadow_op::mul, op::mshadow_op::mixed_mul,
                              op::mshadow_op::mixed_mul>);

NNVM_REGISTER_OP(_backward_npi_broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", MixedBinaryBackwardUseIn<gpu, mshadow_op::right,
                                                              mshadow_op::left>);

NNVM_REGISTER_OP(_npi_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_npi_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::power>);

NNVM_REGISTER_OP(_npi_copysign)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::copysign>);

NNVM_REGISTER_OP(_npi_lcm)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::lcm>);

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

NNVM_REGISTER_OP(_npi_add_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::minus>);

NNVM_REGISTER_OP(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rminus>);

NNVM_REGISTER_OP(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::mul>);

NNVM_REGISTER_OP(_npi_mod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_npi_rmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rmod>);

NNVM_REGISTER_OP(_npi_power_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::power>);

NNVM_REGISTER_OP(_npi_rpower_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rpower>);

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
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::lcm>);

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
