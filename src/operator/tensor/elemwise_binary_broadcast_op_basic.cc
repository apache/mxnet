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
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_add)
.add_alias("broadcast_plus")
.describe(R"code(Returns element-wise sum of the input arrays with broadcasting.

`broadcast_plus` is an alias to the function `broadcast_add`.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_add(x, y) = [[ 1.,  1.,  1.],
                          [ 2.,  2.,  2.]]

   broadcast_plus(x, y) = [[ 1.,  1.,  1.],
                           [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_add"});

NNVM_REGISTER_OP(_backward_broadcast_add)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::identity>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_sub)
.add_alias("broadcast_minus")
.describe(R"code(Returns element-wise difference of the input arrays with broadcasting.

`broadcast_minus` is an alias to the function `broadcast_sub`.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                          [ 0.,  0.,  0.]]

   broadcast_minus(x, y) = [[ 1.,  1.,  1.],
                            [ 0.,  0.,  0.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_broadcast_sub"});

NNVM_REGISTER_OP(_backward_broadcast_sub)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::negation>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_mul)
.describe(R"code(Returns element-wise product of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_mul(x, y) = [[ 0.,  0.,  0.],
                          [ 1.,  1.,  1.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mul"});


NNVM_REGISTER_OP(_backward_broadcast_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::right,
                                                              mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_div)
.describe(R"code(Returns element-wise division of the input arrays with broadcasting.

Example::

   x = [[ 6.,  6.,  6.],
        [ 6.,  6.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_div(x, y) = [[ 3.,  3.,  3.],
                          [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::div>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_div"});

NNVM_REGISTER_OP(_backward_broadcast_div)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::div_grad,
                                                              mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_mod)
.describe(R"code(Returns element-wise modulo of the input arrays with broadcasting.

Example::

   x = [[ 8.,  8.,  8.],
        [ 8.,  8.,  8.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_mod(x, y) = [[ 0.,  0.,  0.],
                          [ 2.,  2.,  2.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_mod"});

NNVM_REGISTER_OP(_backward_broadcast_mod)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::mod_grad,
                                                                  mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
