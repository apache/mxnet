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
 *  Copyright (c) 2016 by Contributors
 * \file optimizer_op.cc
 * \brief Optimizer operators
 * \author Haibin Lin
 */
#include "./adamw-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AdamWParam);

NNVM_REGISTER_OP(_contrib_adamw_update)
.describe(R"code(Update function for AdamW optimizer. AdamW is seen as a modification of
Adam by decoupling the weight decay from the optimization steps taken w.r.t. the loss function.

Adam update consists of the following steps, where g represents gradient and m, v
are 1st and 2nd order moment estimates (mean and variance).

.. math::

 g_t = \nabla J(W_{t-1})\\
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon } + wd W_{t-1})

It updates the weights using::

 m = beta1*m + (1-beta1)*grad
 v = beta2*v + (1-beta2)*(grad**2)
 w -= eta * (learning_rate * m / (sqrt(v) + epsilon) + w * wd)

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdamWParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", AdamWUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_arguments(AdamWParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
