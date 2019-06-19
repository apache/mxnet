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

#include <mxnet/base.h>
#include "./np_bitwise_and_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_np_bitwise_and)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"x1", "x2"};
})
.set_attr<FCompute>("FCompute", ElemwiseBinaryOp::Compute<cpu, bitwise_and_forward>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("x1", "NDArray-or-Symbol", "Input ndarray")
.add_argument("x2", "NDArray-or-Symbol", "Input ndarray");

} // namespace op
} // namespace mxnet

