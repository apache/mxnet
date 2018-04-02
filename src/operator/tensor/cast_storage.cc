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
 * \file cast_storage.cc
 * \brief CPU Implementation of cast_storage operator.
 */

#include "./cast_storage-inl.h"
#include "../elemwise_op_common.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(CastStorageParam);
NNVM_REGISTER_OP(cast_storage)
.add_alias("_sparse_cast_storage")
.describe(R"code(Casts tensor storage type to the new type.

When an NDArray with default storage type is cast to csr or row_sparse storage,
the result is compact, which means:

- for csr, zero values will not be retained
- for row_sparse, row slices of all zeros will not be retained

The storage type of ``cast_storage`` output depends on stype parameter:

- cast_storage(csr, 'default') = default
- cast_storage(row_sparse, 'default') = default
- cast_storage(default, 'csr') = csr
- cast_storage(default, 'row_sparse') = row_sparse

Example::

    dense = [[ 0.,  1.,  0.],
             [ 2.,  0.,  3.],
             [ 0.,  0.,  0.],
             [ 0.,  0.,  0.]]

    # cast to row_sparse storage type
    rsp = cast_storage(dense, 'row_sparse')
    rsp.indices = [0, 1]
    rsp.values = [[ 0.,  1.,  0.],
                  [ 2.,  0.,  3.]]

    # cast to csr storage type
    csr = cast_storage(dense, 'csr')
    csr.indices = [1, 0, 2]
    csr.values = [ 1.,  2.,  3.]
    csr.indptr = [0, 1, 3, 3, 3]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CastStorageParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferStorageType>("FInferStorageType", CastStorageInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", CastStorageComputeEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(CastStorageParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
