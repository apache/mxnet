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
 * \file moments.cc
 * \brief Moments operator
 * \author Hao Jin
*/

#include "./moments-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MomentsParam);

NNVM_REGISTER_OP(moments)
.describe(R"code(
Calculate the mean and variance of `data`.

The mean and variance are calculated by aggregating the contents of data across axes.
If x is 1-D and axes = [0] this is just the mean and variance of a vector.

Example:

     x = [[1, 2, 3], [4, 5, 6]]
     mean, var = moments(data=x, axes=[0])
     mean = [2.5, 3.5, 4.5]
     var = [2.25, 2.25, 2.25]
     mean, var = moments(data=x, axes=[1])
     mean = [2.0, 5.0]
     var = [0.66666667, 0.66666667]
     mean, var = moments(data=x, axis=[0, 1])
     mean = [3.5]
     var = [2.9166667]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<MomentsParam>)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", MomentsShape)
.set_attr<nnvm::FInferType>("FInferType", MomentsType)
.set_attr<FCompute>("FCompute<cpu>", MomentsForward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_moments"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(MomentsParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_moments)
.set_attr_parser(ParamParser<MomentsParam>)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", MomentsBackward<cpu>);

}  // namespace op
}  // namespace mxnet
