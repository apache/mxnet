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
* Copyright (c) 2018 by Contributors
* \file trace_op.cc
* \brief
* \author Sam Skalicky
*/

#include "./trace_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(TraceParam);

NNVM_REGISTER_OP(trace)
.describe(R"code(Computes the trace of an  array.
Extracts the diagonals of the NDArray's sub-arrays with axes specified by ``axis1`` and ``axis2`` and sums the last dimension of the resulting diagonal.
The output shape would be decided by removing the axes numbered ``axis1`` and ``axis2`` from the input shape.

  For example, when the input shape is `(2, 3, 4, 5)`, ``axis1`` and ``axis2`` are 0 and 2
  respectively and ``k`` is 0, the resulting shape would be `(3, 5)`.

Examples::

  x = [[1, 2, 3],
       [4, 5, 6]]

  trace(x) = [6]

  trace(x, k=1) = [8]

  trace(x, k=-1) = [4]

  x = [[[1, 2],
        [3, 4]],

       [[5, 6],
        [7, 8]]]

  trace(x) = [8, 10]

  trace(x, k=1) = [3, 4]

  trace(x, axis1=-2, axis2=-1) = [5, 13]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<TraceParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", TraceOpShape)
.set_attr<nnvm::FInferType>("FInferType", TraceOpType)
.set_attr<FCompute>("FCompute<cpu>", TraceOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_trace"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(TraceParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_trace)
.set_attr_parser(ParamParser<TraceParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TraceOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
