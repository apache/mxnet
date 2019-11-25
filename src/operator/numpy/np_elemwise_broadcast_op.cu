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

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_add)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<gpu>",
  NumpyBinaryBroadcastComputeWithBool<gpu, op::mshadow_op::plus, op::mshadow_op::mixed_plus,
                                      op::mshadow_op::mixed_plus>);
#else
.set_attr<FCompute>(
  "FCompute<gpu>",
  NumpyBinaryBroadcastComputeWithBool<gpu, op::mshadow_op::plus>);
#endif

NNVM_REGISTER_OP(_npi_subtract)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<gpu>",
  NumpyBinaryBroadcastCompute<gpu, op::mshadow_op::minus, op::mshadow_op::mixed_minus,
                              op::mshadow_op::mixed_rminus>);
#else
.set_attr<FCompute>(
  "FCompute<gpu>",
  NumpyBinaryBroadcastCompute<gpu, op::mshadow_op::minus>);
#endif

NNVM_REGISTER_OP(_npi_multiply)
#ifndef _WIN32
.set_attr<FCompute>(
  "FCompute<gpu>",
  NumpyBinaryBroadcastComputeWithBool<gpu, op::mshadow_op::mul, op::mshadow_op::mixed_mul,
                                      op::mshadow_op::mixed_mul>);
#else
.set_attr<FCompute>(
  "FCompute<gpu>",
  NumpyBinaryBroadcastComputeWithBool<gpu, op::mshadow_op::mul>);
#endif

NNVM_REGISTER_OP(_backward_npi_broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", NumpyBinaryBackwardUseIn<gpu, mshadow_op::right,
                                                              mshadow_op::left>);

NNVM_REGISTER_OP(_npi_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_npi_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::power>);

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

}  // namespace op
}  // namespace mxnet
