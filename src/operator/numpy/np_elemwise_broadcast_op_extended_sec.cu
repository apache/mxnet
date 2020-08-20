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
 * \file np_elemwise_broadcast_op_extended_sec.cu
 * \brief GPU Implementation of extended functions for elementwise binary broadcast operator. (Second extended file)
 */

#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_fmax)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"fmax"});

NNVM_REGISTER_OP(_backward_npi_fmax)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"greater_equal", "less"});

NNVM_REGISTER_OP(_npi_fmax_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"fmax"});

NNVM_REGISTER_OP(_backward_npi_fmax_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"greater_equal"});

NNVM_REGISTER_OP(_npi_fmin)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"fmin"});

NNVM_REGISTER_OP(_backward_npi_fmin)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"less_equal",
                                                                     "greater"});

NNVM_REGISTER_OP(_npi_fmin_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"fmin"});

NNVM_REGISTER_OP(_backward_npi_fmin_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"less_equal"});

NNVM_REGISTER_OP(_npi_fmod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"fmod"});

NNVM_REGISTER_OP(_backward_npi_fmod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"mod_grad",
                                                                     "mod_rgrad"});

NNVM_REGISTER_OP(_npi_fmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"fmod"});

NNVM_REGISTER_OP(_backward_npi_fmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"mod_grad"});

NNVM_REGISTER_OP(_npi_rfmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rfmod"});

NNVM_REGISTER_OP(_backward_npi_rfmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"rmod_grad"});

}  // namespace op
}  // namespace mxnet
