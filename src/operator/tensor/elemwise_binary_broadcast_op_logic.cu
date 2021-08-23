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
 * \file elemwise_binary_broadcast_op_logic.cu
 * \brief GPU Implementation of elementwise binary broadcast logical operators.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(broadcast_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"equal"});

NNVM_REGISTER_OP(broadcast_not_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"not_equal"});

NNVM_REGISTER_OP(broadcast_greater)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"greater"});

NNVM_REGISTER_OP(broadcast_greater_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"greater_equal"});

NNVM_REGISTER_OP(broadcast_lesser)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"less"});

NNVM_REGISTER_OP(broadcast_lesser_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"less_equal"});

NNVM_REGISTER_OP(broadcast_logical_and)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"logical_and"});

NNVM_REGISTER_OP(broadcast_logical_or)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"logical_or"});

NNVM_REGISTER_OP(broadcast_logical_xor)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"logical_xor"});

}  // namespace op
}  // namespace mxnet
