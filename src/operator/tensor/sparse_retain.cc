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
 * \file sparse_retain.cc
 * \brief
*/

#include "./sparse_retain-inl.h"
namespace mxnet {
namespace op {

// Add prefix "_sparse_" to prevent it from being registered
// under mxnet.ndarray in python frontend as this op only
// accepts row-sparse format ndarrays. It will be registered
// under mxnet.ndarray.sparse with name retain.
NNVM_REGISTER_OP(_sparse_retain)
.describe(R"code(pick rows specified by user input index array from a row sparse matrix
and save them in the output sparse matrix.

Example::

  data = [[1, 2], [3, 4], [5, 6]]
  indices = [0, 1, 3]
  shape = (4, 2)
  rsp_in = row_sparse(data, indices)
  to_retain = [0, 3]
  rsp_out = retain(rsp_in, to_retain)
  rsp_out.values = [[1, 2], [5, 6]]
  rsp_out.indices = [0, 3]

The storage type of ``retain`` output depends on storage types of inputs

- retain(row_sparse, default) = row_sparse
- otherwise, ``retain`` is not supported

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", SparseRetainOpShape)
.set_attr<nnvm::FInferType>("FInferType", SparseRetainOpType)
.set_attr<FInferStorageType>("FInferStorageType", SparseRetainForwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseRetainOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_sparse_retain", n, ograds,
                               {n->inputs[sr::kIdx]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "The input array for sparse_retain operator.")
.add_argument("indices", "NDArray-or-Symbol", "The index array of rows ids that will be retained.");

NNVM_REGISTER_OP(_backward_sparse_retain)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", SparseRetainBackwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseRetainOpBackwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
