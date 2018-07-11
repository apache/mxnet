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
* Copyright (c) 2015 by Contributors
* \file diag_op.cc
* \brief
* \author Istvan Fehervari
*/

#include "./diag_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DiagParam);

NNVM_REGISTER_OP(diag)
.describe(R"code(Extracts a diagonal or constructs a diagonal array.

``diag``'s behavior depends on the input array dimensions:

- 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero
- 2-D arrays: returns elements in the diagonal as a new 1-D array
- N-D arrays: not supported yet

Examples::

  x = [[1, 2, 3],
       [4, 5, 6]]

  diag(x) = [1, 5]

  diag(x, k=1) = [2, 6]

  diag(x, k=-1) = [4]

  x = [1, 2, 3]

  diag(x) = [[1, 0, 0],
             [0, 2, 0],
             [0, 0, 3]]

  diag(x, k=1) = [[0, 1, 0],
                  [0, 0, 2],
                  [0, 0, 0]]

  diag(x, k=-1) = [[0, 0, 0],
                   [1, 0, 0],
                   [0, 2, 0]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<DiagParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", DiagOpShape)
.set_attr<nnvm::FInferType>("FInferType", DiagOpType)
.set_attr<FCompute>("FCompute<cpu>", DiagOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_diag"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(DiagParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_diag)
.set_attr_parser(ParamParser<DiagParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", DiagOpBackward<cpu>);


}  // namespace op
}  // namespace mxnet
