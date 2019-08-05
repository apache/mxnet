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
* Copyright (c) 2019 by Contributors
* \file cumulative_op.cc
* \brief CPU implementation of cumulative operators
* \author Chaitanya Bapat
*/

#include "./cumulative_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(CumsumParam);

NNVM_REGISTER_OP(cumsum)
.describe(R"code(Finds the cumulative sum of the elements along a given axis.

Examples::

>>> x = mx.nd.array([[1,2,3], [4,5,6]])
>>> mx.nd.cumsum(x)
[  1.,   3.,   6.,  10.,  15.,  21.]
<NDArray 3 @cpu(0)>

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CumsumParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", CumSumOpShape)
.set_attr<nnvm::FInferType>("FInferType", CumsumOpType)
.set_attr<FCompute>("FCompute<cpu>", CumSumOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_cumsum"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray");
.add_arguments(CumsumParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_cumsum)
.set_attr_parser(ParamParser<CumsumParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CumsumOpBackward<cpu>);


}  // namespace op
}  // namespace mxnet
