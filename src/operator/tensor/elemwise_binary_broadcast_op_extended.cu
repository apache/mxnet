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
 * \file elemwise_binary_broadcast_op_extended.cu
 * \brief GPU Implementation of extended functions for elementwise binary broadcast operator.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(broadcast_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"power"});

NNVM_REGISTER_OP(_backward_broadcast_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"power_grad", "power_rgrad"});

NNVM_REGISTER_OP(broadcast_maximum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"max"});

NNVM_REGISTER_OP(_backward_broadcast_maximum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"greater_equal", "less"});

NNVM_REGISTER_OP(broadcast_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"min"});

NNVM_REGISTER_OP(_backward_broadcast_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"less_equal", "greater"});

NNVM_REGISTER_OP(broadcast_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"hypot"});

NNVM_REGISTER_OP(_backward_broadcast_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"hypot_grad_left",
                                                                     "hypot_grad_right"});

}  // namespace op
}  // namespace mxnet
