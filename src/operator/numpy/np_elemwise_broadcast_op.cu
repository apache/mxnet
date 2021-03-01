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
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"add"});

NNVM_REGISTER_OP(_backward_npi_broadcast_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"one", "one"});

NNVM_REGISTER_OP(_npi_subtract)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"sub"});

NNVM_REGISTER_OP(_backward_npi_broadcast_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"one", "negone"});

NNVM_REGISTER_OP(_npi_multiply)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"mul"});

NNVM_REGISTER_OP(_backward_npi_broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"right", "left"});

NNVM_REGISTER_OP(_npi_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"mod"});

NNVM_REGISTER_OP(_backward_npi_broadcast_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"mod_grad", "mod_rgrad"});

NNVM_REGISTER_OP(_npi_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"power"});

NNVM_REGISTER_OP(_backward_npi_broadcast_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"power_grad", "power_rgrad"});

NNVM_REGISTER_OP(_npi_add_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"add"});

NNVM_REGISTER_OP(_npi_subtract_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"sub"});

NNVM_REGISTER_OP(_npi_rsubtract_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rsub"});

NNVM_REGISTER_OP(_npi_multiply_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"mul"});

NNVM_REGISTER_OP(_npi_mod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"mod"});

NNVM_REGISTER_OP(_npi_rmod_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rmod"});

NNVM_REGISTER_OP(_npi_power_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"power"});

NNVM_REGISTER_OP(_npi_rpower_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rpow"});

}  // namespace op
}  // namespace mxnet
