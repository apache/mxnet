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
 * \file np_elemwise_binary_op.cu
 * \brief GPU Implementation for element-wise binary operators.
 */


#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(ldexp)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::ldexp>);

NNVM_REGISTER_OP(ldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::ldexp>);

NNVM_REGISTER_OP(rldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::rldexp>);

NNVM_REGISTER_OP(_backward_ldexp)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::ldexp_grad,
                                                              mshadow_op::ldexp_rgrad>);

NNVM_REGISTER_OP(_backward_ldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<gpu, mshadow_op::ldexp_grad>);

NNVM_REGISTER_OP(_backward_rldexp_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Backward<gpu, mshadow_op::rldexp_grad>);

}  // namespace op
}  // namespace mxnet