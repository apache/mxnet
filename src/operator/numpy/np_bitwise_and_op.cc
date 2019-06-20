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
 * \file np_broadcast_reduce_op_index.cc
 * \brief CPU Implementation of elementwise bitwise add.
 */

#include "./np_bitwise_and_op-inl.h"
#include <mxnet/base.h>
#include "../mshadow_op.h" // mshadow operations
#include "../operator_common.h" // MakeZeroGradNodes
#include "../tensor/elemwise_binary_op.h" // ElemwiseShape, ElemwiseType
#include "../tensor/elemwise_binary_broadcast_op.h" // BinaryBroadcastCompute

//#include <mxnet/base.h>
//#include <mxnet/operator_util.h>
//#include <vector>
//#include "../mxnet_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_np_bitwise_and)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"x1", "x2"};
})
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::bitwise_and>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("x1", "NDArray-or-Symbol", "Input ndarray")
.add_argument("x2", "NDArray-or-Symbol", "Input ndarray");

} // namespace op
} // namespace mxnet