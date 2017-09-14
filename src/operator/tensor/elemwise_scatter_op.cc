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
#include "./elemwise_binary_op-inl.h"
#include "./elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_scatter_elemwise_div, mshadow::op::div)
.describe(R"code(Divides arguments element-wise.  If the inputs are sparse, then
only the values which exist in the sparse arrays are computed.  The 'missing' values are ignored.
For dense input, this operator behaves exactly like elemwise_div.

The storage type of ``scatter_div`` output depends on storage types of inputs

- scatter_div(row_sparse, row_sparse) = row_sparse
- otherwise, ``scatter_div`` generates output with default storage

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_scatter_plus_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow::op::plus>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BinaryScalarOp::ComputeEx<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_scatter_minus_scalar)
.set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow::op::minus>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BinaryScalarOp::ComputeEx<cpu, mshadow::op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

}  // namespace op
}  // namespace mxnet