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
 * \file quadratic_op.cc
 * \brief CPU Implementation of quadratic op
 */
#include "./quadratic_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(QuadraticParam);

NNVM_REGISTER_OP(_contrib_quadratic)
.describe(R"code(This operators implements the quadratic function:
.. math::
    f(x) = ax^2+bx+c
where :math:`x` is an input tensor and all operations
in the function are element-wise.
Example::
  x = [[1, 2], [3, 4]]
  y = quadratic(data=x, a=1, b=2, c=3)
  y = [[6, 11], [18, 27]]

The storage type of ``quadratic`` output depends on storage types of inputs
  - quadratic(csr, a, b, 0) = csr
  - quadratic(default, a, b, c) = default

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<QuadraticParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuadraticOpShape)
.set_attr<nnvm::FInferType>("FInferType", QuadraticOpType)
.set_attr<FInferStorageType>("FInferStorageType", QuadraticOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", QuadraticOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_contrib_backward_quadratic"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(QuadraticParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_backward_quadratic)
.set_attr_parser(ParamParser<QuadraticParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", QuadraticOpBackward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", QuadraticOpForwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
