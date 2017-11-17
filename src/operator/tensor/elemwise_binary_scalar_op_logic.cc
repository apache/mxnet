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
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::eq>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_EqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_not_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::ne>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_NotEqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_greater_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::gt>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_GreaterScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_greater_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::ge>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_GreaterEqualScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_lesser_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::lt>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LesserScalar");

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_lesser_equal_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::le>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_alias("_LesserEqualScalar");

}  // namespace op
}  // namespace mxnet
