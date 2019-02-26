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
#include "../optimizer_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AdamWParam);

template<template <typename xpu> class F>
inline void MPUpdateCPU(const nnvm::NodeAttrs& attrs,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  // copy to cpu and check NaN value
  TBlob scale_blob = inputs[inputs.size() - 1];
  MSHADOW_REAL_TYPE_SWITCH(scale_blob.type_flag_, DType, {
    float scalef = static_cast<float>(*scale_blob.dptr<DType>());
    if (!std::isfinite(scalef) || scalef == 0) return;
    std::vector<TBlob> inputs_wo_scale;
    size_t num_in = inputs.size();
    inputs_wo_scale.reserve(num_in - 1);
    for (size_t i = 0; i < num_in - 1; i++) inputs_wo_scale.emplace_back(inputs[i]);
    F<cpu>::Forward(attrs, ctx, inputs_wo_scale, req, outputs, scalef);
  });
}

NNVM_REGISTER_OP(_contrib_mp_adamw_update)
.describe(R"code(Update function for multi-precision AdamW optimizer.

AdamW is seen as a modification of Adam by decoupling the weight decay from the
optimization steps taken w.r.t. the loss function.

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

Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
the update is skipped.
)code" ADD_FILELINE)
.set_num_inputs(6)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdamWParam>)
.set_attr<nnvm::FInferShape>("FInferShape", MPUpdateInferShape<2, 1, 6>)
.set_attr<nnvm::FInferType>("FInferType", MPUpdateInferType<2, 1, 6>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3, 4};
  })
.set_attr<FCompute>("FCompute<cpu>", MPUpdateCPU<MPAdamWUpdate>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_argument("rescale_grad", "NDArray-or-Symbol",
              "Rescale gradient to rescale_grad * grad. If NaN, the update is skipped.")
.add_arguments(AdamWParam::__FIELDS__());

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

Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
the update is skipped.
)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdamWParam>)
.set_attr<nnvm::FInferShape>("FInferShape", MPUpdateInferShape<4, 1, 5>)
.set_attr<nnvm::FInferType>("FInferType", MPUpdateInferType<4, 1, 5>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", MPUpdateCPU<AdamWUpdate>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_argument("rescale_grad", "NDArray-or-Symbol",
              "Rescale gradient to rescale_grad * grad. If NaN, the update is skipped.")
.add_arguments(AdamWParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
