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
 * \file slice_sum.cc
 * \brief Fused implementation of slice and sum operation
 */
#include "./slice_sum-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SliceSumParam);

NNVM_REGISTER_OP(slice_sum)
.describe(R"code(Computes the sum
of two tensors after slicing the second
tensor according to the axis, begin and end parameters.
Example:: see test_slice_sum in tests/python/unittest/test_operator.py

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "data2"};
})
.set_attr_parser(ParamParser<SliceSumParam>)
.set_attr<mxnet::FInferShape>("FInferShape", SliceSumOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", SliceSumOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_slice_sum"})
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_argument("data2", "NDArray-or-Symbol", "Source input2")
.add_arguments(SliceSumParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_slice_sum)
.set_attr<FCompute>("FCompute<cpu>", SliceSumOpBackward<cpu>)
.set_num_inputs(3)
.set_num_outputs(2);

}  // namespace op
}  // namespace mxnet
