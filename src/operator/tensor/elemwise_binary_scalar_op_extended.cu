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
 * \file elemwise_binary_scalar_op_extended.cu
 * \brief GPU Implementation of extended binary scalar functions.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_maximum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"max"});

NNVM_REGISTER_OP(_backward_maximum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"greater_equal"});

NNVM_REGISTER_OP(_minimum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"min"});

NNVM_REGISTER_OP(_backward_minimum_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"less_equal"});

NNVM_REGISTER_OP(_power_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"power"});

NNVM_REGISTER_OP(_backward_power_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"power_grad"});

NNVM_REGISTER_OP(_rpower_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rpow"});

NNVM_REGISTER_OP(_backward_rpower_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"rpower_grad"});

NNVM_REGISTER_OP(_hypot_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"hypot"});

NNVM_REGISTER_OP(_backward_hypot_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"hypot_grad_left"});

NNVM_REGISTER_OP(smooth_l1)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"smooth_l1"});

NNVM_REGISTER_OP(_backward_smooth_l1)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"smooth_l1_grad"});

}  // namespace op
}  // namespace mxnet
