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
* \brief CPU implementation of diag operator
* \author Istvan Fehervari, Zhijingcheng Yu
*/

#include "./diag_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DiagParam);

NNVM_REGISTER_OP(diag)
.describe(R"code(Extracts a diagonal or constructs a diagonal array.

``diag``'s behavior depends on the input array dimensions:

- 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero.
- N-D arrays: extracts the diagonals of the sub-arrays with axes specified by ``axis1`` and ``axis2``.
  The output shape would be decided by removing the axes numbered ``axis1`` and ``axis2`` from the
  input shape and appending to the result a new axis with the size of the diagonals in question.

  For example, when the input shape is `(2, 3, 4, 5)`, ``axis1`` and ``axis2`` are 0 and 2
  respectively and ``k`` is 0, the resulting shape would be `(3, 5, 2)`.

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

  x = [[[1, 2],
        [3, 4]],

       [[5, 6],
        [7, 8]]]

  diag(x) = [[1, 7],
             [2, 8]]

  diag(x, k=1) = [[3],
                  [4]]

  diag(x, axis1=-2, axis2=-1) = [[1, 4],
                                 [5, 8]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<DiagParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", DiagOpShape)
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
