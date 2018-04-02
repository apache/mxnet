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
 * \file dot.cc
 * \brief CPU Implementation of matrix dot
 */

#include "./dot-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DotParam);

NNVM_REGISTER_OP(dot)
.add_alias("_sparse_dot")  // alias for op registration under mxnet.ndarray.sparse
.describe(R"doc(Dot product of two arrays.

``dot``'s behavior depends on the input array dimensions:

- 1-D arrays: inner product of vectors
- 2-D arrays: matrix multiplication
- N-D arrays: a sum product over the last axis of the first input and the first
  axis of the second input

  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
  result array will have shape `(n,m,r,s)`. It is computed by::

    dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])

  Example::

    x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
    y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
    dot(x,y)[0,0,1,1] = 0
    sum(x[0,0,:]*y[:,1,1]) = 0

The storage type of ``dot`` output depends on storage types of inputs and transpose options:

- dot(csr, default) = default
- dot(csr.T, default) = row_sparse
- dot(csr, row_sparse) = default
- dot(default, csr) = csr
- otherwise, ``dot`` generates output with default storage

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", DotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", DotForwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", DotForward_<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", DotForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_dot"})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", DotBackwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", DotBackward_<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", DotBackwardEx<cpu>)
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(batch_dot)
.describe(R"doc(Batchwise dot product.

``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.

For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
`(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
which is computed by::

   batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", BatchDotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BatchDotForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_batch_dot"})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_batch_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BatchDotBackward_<cpu>);

}  // namespace op
}  // namespace mxnet
