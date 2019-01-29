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
 * Copyright (c) 2018 by Contributors
 * \file gradient_multiplier_op.cu
 * \brief
 * \author Istvan Fehervari
*/
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_gradientmultiplier)
.set_attr<FComputeEx>("FComputeEx<gpu>", UnaryOp::IdentityComputeEx<gpu>)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_contrib_backward_gradientmultiplier)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, op::mshadow_op::mul>);

}  // namespace op
}  // namespace mxnet
