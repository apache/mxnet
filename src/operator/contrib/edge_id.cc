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
 * \file edge_id.cc
 * \brief CPU Implementation of edge_id op.
 */
#include "./edge_id-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_edge_id)
.describe(R"code(This operator implements the edge_id function for csr arrays,
where output[i] = input[u[i], v[i]] if input[u[i], v[i]] is a non-zero element of input,
otherwise output[i] will be -1. Both u and v should be 1D vectors.
Example::
  x = [[ 1, 0, 0 ],
       [ 0, 2, 0 ],
       [ 0, 0, 3 ]]
  u = [ 0, 0, 1, 1, 2, 2 ]
  v = [ 0, 1, 1, 2, 0, 2 ]
  edge_id(x, u, v) = [ 1, -1, 2, -1, -1, 3 ]

The storage type of ``edge_id`` output depends on storage types of inputs
  - quadratic(csr, default, default) = default
  - default and rsp inputs are not supported

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "u", "v"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", EdgeIDShape)
.set_attr<nnvm::FInferType>("FInferType", EdgeIDType)
.set_attr<FInferStorageType>("FInferStorageType", EdgeIDStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", EdgeIDForwardEx<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_argument("u", "NDArray-or-Symbol", "u ndarray")
.add_argument("v", "NDArray-or-Symbol", "v ndarray");

}  // namespace op
}  // namespace mxnet
