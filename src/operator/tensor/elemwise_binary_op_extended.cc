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
#include "./elemwise_binary_op-inl.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_power, mshadow_op::power)
.add_alias("_Power")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_power"});

NNVM_REGISTER_OP(_backward_power)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::power_grad, mshadow_op::power_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_maximum, mshadow_op::maximum)
.add_alias("_Maximum")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_maximum"});

NNVM_REGISTER_OP(_backward_maximum)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<cpu, mshadow_op::ge,
  mshadow_op::lt>);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_minimum, mshadow_op::minimum)
.add_alias("_Minimum")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_minimum"});

NNVM_REGISTER_OP(_backward_minimum)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<cpu, mshadow_op::le,
  mshadow_op::gt>);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_hypot, mshadow_op::hypot)
.add_alias("_Hypot")
.describe(R"code(Given the "legs" of a right triangle, return its hypotenuse.

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_hypot" });

NNVM_REGISTER_OP(_backward_hypot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> > {{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<cpu,
  mshadow_op::hypot_grad_left, mshadow_op::hypot_grad_right>);

}  // namespace op
}  // namespace mxnet
