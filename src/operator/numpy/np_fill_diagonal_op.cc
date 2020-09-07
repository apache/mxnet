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
* Copyright (c) 2020 by Contributors
* \file np_fill_diagonal_op.cc
* \brief CPU implementation of numpy fill_diagonal operator
*/

#include "./np_fill_diagonal_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyFillDiagonalParam);

NNVM_REGISTER_OP(_npi_fill_diagonal)
.describe(R"code(Fill the main diagonal of the given array"
  "of any dimensionality.)code"
  ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) { return std::vector<std::string>{"a"};
})
.set_attr_parser(ParamParser<NumpyFillDiagonalParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyFillDiagonalOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& n) {
     return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyFillDiagonalForward<cpu>)
.add_argument("a", "NDArray-or-Symbol", "Source input")
.add_arguments(NumpyFillDiagonalParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
