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
 * \file np_elemwise_broadcast_logic_op.cu
 * \brief GPU Implementation of basic functions for elementwise binary
 * broadcast logic operator.
 */
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {


#if MXNET_USE_TVM_OP == 0

#define MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(name)                                     \
  NNVM_REGISTER_OP(_npi_##name)                                                               \
  .set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastComputeLogic<gpu, mshadow_op::np_##name>)

#define MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(name)                               \
  NNVM_REGISTER_OP(_npi_##name##_scalar)                                                       \
  .set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::ComputeLogic<gpu, mshadow_op::np_##name>)

MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_LOGIC_GPU(less_equal);

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(not_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(greater);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(less);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(greater_equal);
MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR_LOGIC_GPU(less_equal);

#endif  // MXNET_USE_TVM_OP

}  // namespace op
}  // namespace mxnet
