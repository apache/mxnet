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
 * \file activation.cc
 * \brief activation op
 */
#include "./activation-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_OPERATOR_REGISTER_UNARY(Activation)
.describe(R"code(Applies an activation function element-wise to the input.

The following activation functions are supported:

- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<FCompute>("FCompute<cpu>", ActivationCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_activation"})
.add_arguments(ActivationParam::__FIELDS__());

MXNET_OPERATOR_REGISTER_BINARY(_backward_activation)
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<FCompute>("FCompute<cpu>", ActivationGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
