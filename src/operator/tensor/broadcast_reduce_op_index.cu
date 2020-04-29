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
 * \file broadcast_reduce_op_index.cu
 * \brief GPU Implementation of broadcast and reduce functions based on index.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(argmax)
.set_attr<FCompute>("FCompute<gpu>", ArgMax<gpu>);

NNVM_REGISTER_OP(argmin)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::minimum>);

// Legacy support
NNVM_REGISTER_OP(argmax_channel)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::maximum>);

NNVM_REGISTER_OP(pick)
.set_attr<FCompute>("FCompute<gpu>", PickOpForward<gpu>);


NNVM_REGISTER_OP(_backward_pick)
.set_attr<FCompute>("FCompute<gpu>", PickOpBackward<gpu>);

template<>
int DefineNumbWorkers<gpu>(const TShape &shape, int axis) {
  const auto nSteps = static_cast<uint32_t>(shape[axis]);
  const auto nThreads = shape.Size()/nSteps;
  if (nThreads > nSteps)
    return 1;

  // The formula used here is just a heuristic. Experimenting with different values,
  // we accumulated a lot of data for different(shape, axis, numbWorkers).
  // It turned out that they almost perfectly correspond to this formula.
  const auto a = static_cast<float>(nSteps)/nThreads;
  const auto b = log2f(a);
  const auto numbWorkers = pow(2, (b * 5 + 28)/11);
  return 2 * numbWorkers <= nSteps? numbWorkers : 1;
}

}  // namespace op
}  // namespace mxnet
