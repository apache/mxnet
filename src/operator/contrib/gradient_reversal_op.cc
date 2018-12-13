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
 * \file gradient_reversal_op.cc
 * \brief
 * \author Istvan Fehervari
*/
#include "./gradient_reversal_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(GradientReversalParam);

NNVM_REGISTER_OP(_contrib_gradientreversal)
.describe(R"code(This operators implements the gradient reversal function.
In forward pass it acts as an identity tranform. During backpropagation it 
multiplies the gradient from the subsequent level by a negative factor and passes it to
the preceding layer.
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<GradientReversalParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", GradientReversalOpShape)
.set_attr<nnvm::FInferType>("FInferType", GradientReversalOpType)
.set_attr<FInferStorageType>("FInferStorageType", GradientReversalOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", GradientReversalOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_contrib_backward_gradientreversal"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(GradientReversalParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_backward_gradientreversal)
.set_attr_parser(ParamParser<GradientReversalParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", GradientReversalOpBackward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", GradientReversalOpForwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
