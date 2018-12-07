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
 * \file gradcancel_op.cc
 * \brief CPU Implementation of gradcancel op
 */
#include "./gradient_cancel-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(GradCancelParam);

NNVM_REGISTER_OP(_contrib_gradcancel)
.describe(R"code(This operators implements the gradcancel function.
It does not do anything in the forward pass, and cancels gradients which
have an absolute value larger than the threshold.
Example::
  x = [[1, 2], [3, 4]]
  y = gradcancel(data=x, threshold=1.0)
  y = [[1, 2], [3, 4]]

The storage type of ``gradcancel`` output depends on storage types of inputs
  - gradcancel(csr, threshold) = csr
  - gradcancel(default, threshold) = default

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<GradCancelParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", GradCancelOpShape)
.set_attr<nnvm::FInferType>("FInferType", GradCancelOpType)
.set_attr<FInferStorageType>("FInferStorageType", GradCancelOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", GradCancelOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_contrib_backward_gradcancel"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(GradCancelParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_backward_gradcancel)
.set_attr_parser(ParamParser<GradCancelParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", GradCancelOpBackward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", GradCancelOpForwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
