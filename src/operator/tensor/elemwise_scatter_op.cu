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
#include "./elemwise_binary_scalar_op.h"
#include "./elemwise_scatter_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_scatter_elemwise_div)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseScatterBinaryOp::Compute<gpu, op::mshadow_op::div>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseScatterBinaryOp::ComputeEx<gpu,
  op::mshadow_op::div>);

NNVM_REGISTER_OP(_scatter_plus_scalar)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseScatterBinaryScalarOp::Compute<gpu, op::mshadow_op::plus>)
.set_attr<FComputeEx>("FComputeEx<gpu>",
                      ElemwiseScatterBinaryScalarOp::ComputeEx<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_scatter_minus_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarOp::Compute<gpu, op::mshadow_op::minus>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarOp::ComputeEx<gpu, op::mshadow_op::minus>);

}  // namespace op
}  // namespace mxnet

