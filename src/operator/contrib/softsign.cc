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
 * \file softsign.cc
 * \brief CPU Implementation of softsign function.
 */
#include "../tensor/elemwise_unary_op.h"
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {
// softsign
MXNET_OPERATOR_REGISTER_UNARY(_contrib_softsign)
  .describe(R"code(Computes softsign of x element-wise.

.. math::
   y = x / (1 + abs(x))

)code" ADD_FILELINE)
  .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_contrib_softsign"})
  .set_attr<FCompute>("FCompute<cpu>",
                      UnaryLaunch<cpu, kernel_launch_op::softsign>);


MXNET_OPERATOR_REGISTER_BINARY(_backward_contrib_softsign)
.set_attr<FCompute>("FCompute<cpu>",
                    BinaryLaunch<cpu, kernel_launch_op::softsign_grad>);
}  // namespace op
}  // namespace mxnet
