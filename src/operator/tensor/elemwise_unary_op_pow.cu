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
 * \file elemwise_unary_op_pow.cu
 * \brief GPU Implementation of power (x^k for fixed k) functions.
 */
#include "./elemwise_binary_op.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {

// reciprocal
NNVM_REGISTER_OP(reciprocal)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"reciprocal"});

NNVM_REGISTER_OP(_backward_reciprocal)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_reciprocal"});

// square
NNVM_REGISTER_OP(square)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"square"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"square"});

NNVM_REGISTER_OP(_backward_square)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_square"});

// sqrt
NNVM_REGISTER_OP(sqrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"sqrt"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"sqrt"});

NNVM_REGISTER_OP(_backward_sqrt)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_sqrt"});

// rsqrt
NNVM_REGISTER_OP(rsqrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"rsqrt"});

NNVM_REGISTER_OP(_backward_rsqrt)
.set_attr<FCompute>("FCompute<gpu>",
  ElemwiseBinaryRTCCompute{"backward_rsqrt"});

// cbrt
NNVM_REGISTER_OP(cbrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"cbrt"})
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryRTCCompute{"cbrt"});


NNVM_REGISTER_OP(_backward_cbrt)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryRTCCompute{"backward_cbrt"});

// rcbrt
NNVM_REGISTER_OP(rcbrt)
.set_attr<FCompute>("FCompute<gpu>", UnaryRTCCompute{"rcbrt"});

NNVM_REGISTER_OP(_backward_rcbrt)
.set_attr<FCompute>("FCompute<gpu>",
  ElemwiseBinaryRTCCompute{"backward_rcbrt"});

}  // namespace op
}  // namespace mxnet
