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
 * \file elemwise_binary_op_extended.cu
 * \brief GPU Implementation of binary function.
 */
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_power)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"power"});

NNVM_REGISTER_OP(_backward_power)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCBwdUseIn{"power_grad", "power_rgrad"});

NNVM_REGISTER_OP(_maximum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"max"});

NNVM_REGISTER_OP(_backward_maximum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCBwdUseIn{"greater_equal", "less"});

NNVM_REGISTER_OP(_minimum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"min"});

NNVM_REGISTER_OP(_backward_minimum)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCBwdUseIn{"less_equal", "greater"});

NNVM_REGISTER_OP(_hypot)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"hypot"});

NNVM_REGISTER_OP(_backward_hypot)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCBwdUseIn{"hypot_grad_left",
                                                               "hypot_grad_right"});

}  // namespace op
}  // namespace mxnet
