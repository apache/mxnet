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
 * \file split_bias_act_red
 * \brief Implementation of fused split, bias addition, multiple activations and final
          reduction operation. Currently, split is set to 2, activations are fixed to 
          ['tanh', 'sigmoid'] and the final reduction is multiplication.
 */
#include "./split_bias_act_red-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SplitBiasActRedParam);

NNVM_REGISTER_OP(_split_bias_act_red)
.describe(R"code(Computes a split of the second
tensor along the axis dimension, adds the first tensor
to all the split tensors, applies provided activation
function to them and finally performs a reduction operation
across all split tensors.
Example:: see test_split_bias_act_red in tests/python/unittest/test_operator.py

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "data2"};
})
.set_attr_parser(ParamParser<SplitBiasActRedParam>)
.set_attr<mxnet::FInferShape>("FInferShape", SplitBiasActRedShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", SplitBiasActRedForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_split_bias_act_red"})
.add_argument("data", "NDArray-or-Symbol", "Source input")
.add_argument("data2", "NDArray-or-Symbol", "Source input2")
.add_arguments(SplitBiasActRedParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_split_bias_act_red)
.set_attr<FCompute>("FCompute<cpu>", SplitBiasActRedBackwardEx<cpu>)
.set_num_inputs(3)
.set_num_outputs(2);

}  // namespace op
}  // namespace mxnet
