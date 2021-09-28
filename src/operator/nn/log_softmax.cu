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
 * \file log_softmax.cu
 * \brief GPU Implementation of log_softmax
 */
#include "./softmax-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(log_softmax)
    .set_attr<FCompute>("FCompute<gpu>", SoftmaxRTCCompute{"log_softmax_fwd"});

NNVM_REGISTER_OP(_backward_log_softmax)
    .set_attr<FCompute>("FCompute<gpu>", SoftmaxRTCGradCompute{"op::left", "log_softmax_bwd"});

NNVM_REGISTER_OP(masked_log_softmax)
    .set_attr<FCompute>("FCompute<gpu>",
                        MaskedSoftmaxCompute<gpu, mxnet_op::log_softmax_fwd, true>);

NNVM_REGISTER_OP(_backward_masked_log_softmax)
    .set_attr<FCompute>("FCompute<gpu>",
                        MaskedSoftmaxGradCompute<gpu, mshadow_op::left, mxnet_op::log_softmax_bwd>);

}  // namespace op
}  // namespace mxnet
