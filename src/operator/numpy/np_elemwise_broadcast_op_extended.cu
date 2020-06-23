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
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"copysign"});

NNVM_REGISTER_OP(_npi_lcm)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"lcm"});

NNVM_REGISTER_OP(_npi_bitwise_and)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastIntCompute<gpu, mshadow_op::bitwise_and>);

NNVM_REGISTER_OP(_npi_bitwise_xor)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"bitwise_xor"});

NNVM_REGISTER_OP(_npi_bitwise_or)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastIntCompute<gpu, mshadow_op::bitwise_or>);

NNVM_REGISTER_OP(_backward_npi_copysign)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"copysign_grad",
                                                                     "zero"});

NNVM_REGISTER_OP(_npi_arctan2)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"arctan2"});

NNVM_REGISTER_OP(_backward_npi_arctan2)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"arctan2_grad",
                                                                     "arctan2_rgrad"});

NNVM_REGISTER_OP(_npi_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"hypot"});

NNVM_REGISTER_OP(_backward_npi_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"hypot_grad_left",
                                                                     "hypot_grad_right"});
NNVM_REGISTER_OP(_npi_copysign_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"copysign"});

NNVM_REGISTER_OP(_npi_rcopysign_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rcopysign"});

NNVM_REGISTER_OP(_backward_npi_copysign_scalar)
.set_attr<FCompute>("FCompute<gpu>",
                    BinaryScalarRTCBackward{"copysign_grad"});

NNVM_REGISTER_OP(_npi_arctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"arctan2"});

NNVM_REGISTER_OP(_backward_npi_arctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"arctan2_grad"});

NNVM_REGISTER_OP(_npi_rarctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rarctan2"});

NNVM_REGISTER_OP(_backward_npi_rarctan2_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"rarctan2_grad"});

NNVM_REGISTER_OP(_npi_lcm_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"lcm"});

NNVM_REGISTER_OP(_npi_bitwise_and_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::ComputeInt<gpu, mshadow_op::bitwise_and>);

NNVM_REGISTER_OP(_npi_bitwise_xor_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"bitwise_xor"});

NNVM_REGISTER_OP(_npi_bitwise_or_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::ComputeInt<gpu, mshadow_op::bitwise_or>);

NNVM_REGISTER_OP(_npi_ldexp)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"ldexp"});

NNVM_REGISTER_OP(_npi_ldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"ldexp"});

NNVM_REGISTER_OP(_npi_rldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rldexp"});

NNVM_REGISTER_OP(_backward_npi_ldexp)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"ldexp_grad",
                                                                     "ldexp_rgrad"});

NNVM_REGISTER_OP(_backward_npi_ldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"ldexp_grad"});

NNVM_REGISTER_OP(_backward_npi_rldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"rldexp_grad"});

}  // namespace op
}  // namespace mxnet
