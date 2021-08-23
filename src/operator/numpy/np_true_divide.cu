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
 * \file np_true_divide.cu
 * \brief GPU Implementation of true_divide operator.
 */

#include "./np_true_divide-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_true_divide)
.set_attr<FCompute>("FCompute<gpu>", TrueDivideBroadcastCompute<gpu>);

NNVM_REGISTER_OP(_backward_npi_broadcast_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCBackwardUseIn{"div_grad", "div_rgrad"});

NNVM_REGISTER_OP(_npi_true_divide_scalar)
.set_attr<FCompute>("FCompute<gpu>", TrueDivideScalarCompute<gpu, mshadow_op::true_divide>);

NNVM_REGISTER_OP(_npi_rtrue_divide_scalar)
.set_attr<FCompute>("FCompute<gpu>", TrueDivideScalarCompute<gpu, mshadow_op::rtrue_divide>);

}  // namespace op
}  // namespace mxnet
