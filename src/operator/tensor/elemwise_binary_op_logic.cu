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
 * \file elemwise_binary_op_logic.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_equal)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"equal"});

NNVM_REGISTER_OP(_not_equal)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"not_equal"});

NNVM_REGISTER_OP(_greater)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"greater"});

NNVM_REGISTER_OP(_greater_equal)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"greater_equal"});

NNVM_REGISTER_OP(_lesser)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"less"});

NNVM_REGISTER_OP(_lesser_equal)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"less_equal"});

NNVM_REGISTER_OP(_logical_and)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"logical_and"});

NNVM_REGISTER_OP(_logical_or)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"logical_or"});

NNVM_REGISTER_OP(_logical_xor)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"logical_xor"});

}  // namespace op
}  // namespace mxnet
