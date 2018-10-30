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
 * \file elemwise_binary_scalar_op_logic.cu
 * \brief GPU Implementation of binary scalar logic functions.
 */
#include "elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::eq>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::LogicComputeEx<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(_not_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::ne>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::LogicComputeEx<gpu, mshadow_op::ne>);

NNVM_REGISTER_OP(_greater_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::gt>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::LogicComputeEx<gpu, mshadow_op::gt>);

NNVM_REGISTER_OP(_greater_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::ge>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::LogicComputeEx<gpu, mshadow_op::ge>);

NNVM_REGISTER_OP(_lesser_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::lt>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::LogicComputeEx<gpu, mshadow_op::lt>);

NNVM_REGISTER_OP(_lesser_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::le>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::LogicComputeEx<gpu, mshadow_op::le>);

NNVM_REGISTER_OP(_logical_and_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::logical_and>);

NNVM_REGISTER_OP(_logical_or_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::logical_or>);

NNVM_REGISTER_OP(_logical_xor_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, mshadow_op::logical_xor>);

}  // namespace op
}  // namespace mxnet
