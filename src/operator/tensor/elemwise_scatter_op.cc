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
#include "./elemwise_scatter_op.h"

namespace mxnet {
namespace op {

static bool StorageTypeRspOrDenseOutput(const nnvm::NodeAttrs &attrs,
                                        const Context &ctx,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  if((*in_attrs)[0] == kRowSparseStorage) {
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, 0, kRowSparseStorage);
    return true;
  }
  return ElemwiseStorageTypeDenseOutput<1>(attrs, ctx, in_attrs, out_attrs);
}

/*! \brief _scatter_elemwise_div */
MXNET_OPERATOR_REGISTER_BINARY(_scatter_elemwise_div)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseScatterBinaryOp::Compute<cpu, mshadow::op::div>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseScatterBinaryOp::ComputeEx<cpu, mshadow::op::div>)
.describe(R"code(Divides arguments element-wise.  If the left-hand-side input is 'row_sparse', then
only the values which exist in the left-hand sparse array are computed.  The 'missing' values
are ignored.

The storage type of ``_scatter_elemwise_div`` output depends on storage types of inputs

- _scatter_elemwise_div(row_sparse, row_sparse) = row_sparse
- _scatter_elemwise_div(row_sparse, dense) = row_sparse
- _scatter_elemwise_div(row_sparse, csr) = row_sparse
- otherwise, ``_scatter_elemwise_div`` behaves exactly like elemwise_div and generates output
with default storage

)code")
.set_attr<FInferStorageType>("FInferStorageType", StorageTypeRspOrDenseOutput)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

/*! \brief _scatter_plus_scalar */
MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_scatter_plus_scalar)
.describe(R"code(Adds a scalar to a tensor element-wise.  If the left-hand-side input is
'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.
The 'missing' values are ignored.

The storage type of ``_scatter_plus_scalar`` output depends on storage types of inputs

- _scatter_plus_scalar(row_sparse, scalar) = row_sparse
- _scatter_plus_scalar(csr, scalar) = csr
- otherwise, ``_scatter_plus_scalar`` behaves exactly like _plus_scalar and generates output
with default storage

)code")
.set_attr<FCompute>("FCompute<cpu>",
                    ElemwiseScatterBinaryScalarOp::Compute<cpu, mshadow::op::plus>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      ElemwiseScatterBinaryScalarOp::ComputeEx<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

/*! \brief _scatter_minus_scalar */
MXNET_OPERATOR_REGISTER_BINARY_SCALAR(_scatter_minus_scalar)
  .describe(R"code(Subtracts a scalar to a tensor element-wise.  If the left-hand-side input is
'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.
The 'missing' values are ignored.

The storage type of ``_scatter_minus_scalar`` output depends on storage types of inputs

- _scatter_minus_scalar(row_sparse, scalar) = row_sparse
- _scatter_minus_scalar(csr, scalar) = csr
- otherwise, ``_scatter_minus_scalar`` behaves exactly like _minus_scalar and generates output
with default storage

)code")
.set_attr<FCompute>("FCompute<cpu>",
                    ElemwiseScatterBinaryScalarOp::Compute<cpu, mshadow::op::minus>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      ElemwiseScatterBinaryScalarOp::ComputeEx<cpu, mshadow::op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

}  // namespace op
}  // namespace mxnet
