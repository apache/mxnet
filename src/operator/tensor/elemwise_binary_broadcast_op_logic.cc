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
 * \file elemwise_binary_broadcast_op_logic.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_equal)
.describe(R"code(Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_equal(x, y) = [[ 0.,  0.,  0.],
                            [ 1.,  1.,  1.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::eq>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_not_equal)
.describe(R"code(Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
                                [ 0.,  0.,  0.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ne>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_greater)
.describe(R"code(Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_greater(x, y) = [[ 1.,  1.,  1.],
                              [ 0.,  0.,  0.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::gt>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_greater_equal)
.describe(R"code(Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
                                    [ 1.,  1.,  1.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ge>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_lesser)
.describe(R"code(Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
                             [ 0.,  0.,  0.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::lt>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_lesser_equal)
.describe(R"code(Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
                                   [ 1.,  1.,  1.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::le>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet
