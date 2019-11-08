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
 * \file fold.cc
 * \brief CPU implementation of unfold operator
 * \author Istvan Fehervari
*/

#include "./fold-inl.h"
namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(UnfoldParam);

NNVM_REGISTER_OP(unfold)
.describe(R"code(Returns a new tensor with ``kernel_size``-sized slices from the input along dimension ``dim``. Step between the slices is given by ``stride``.

The size of dimension ``dim`` in the output will be ``(input.shape[dim] - kernel_size) / stride + 1``. 

Rest of the dimensions will be copied and an additional dimension with size ``kernel_size`` will be appended in the returned tensor.

)code")
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<UnfoldParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<mxnet::FInferShape>("FInferShape", UnfoldOpShape)
.set_attr<nnvm::FInferType>("FInferType", UnfoldOpType)
.set_attr<FCompute>("FCompute<cpu>", UnfoldOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_unfold"})
.add_argument("data", "NDArray-or-Symbol", "data")
.add_arguments(UnfoldParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_unfold)
.set_attr_parser(ParamParser<UnfoldParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", UnfoldOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
