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
 *  Copyright (c) 2018 by Contributors
 * \file adamw.cc
 * \brief Optimizer operators
 * \author Haibin Lin, Moises Hernandez, Andrei Ivanov
 */
#include "./adamw-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AdamWParam);
DMLC_REGISTER_PARAMETER(MultiAdamWParam);

NNVM_REGISTER_OP(_mp_adamw_update)
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
.set_attr<mxnet::FInferShape>("FInferShape", MPUpdateInferShape<2, 1, 6>)
.set_attr<nnvm::FInferType>("FInferType", MPUpdateInferType<2, 1, 6>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3, 4};
  })
.set_attr<FCompute>("FCompute<cpu>", MPUpdate<cpu, MPAdamWUpdate<cpu>>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_argument("rescale_grad", "NDArray-or-Symbol",
              "Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, "
              "the update is skipped.")
.add_arguments(AdamWParam::__FIELDS__());

NNVM_REGISTER_OP(_adamw_update)
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
.set_attr<mxnet::FInferShape>("FInferShape", MPUpdateInferShape<4, 1, 5>)
.set_attr<nnvm::FInferType>("FInferType", MPUpdateInferType<4, 1, 5>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", MPUpdate<cpu, AdamWUpdate<cpu>>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_argument("rescale_grad", "NDArray-or-Symbol",
              "Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, "
              "the update is skipped.")
.add_arguments(AdamWParam::__FIELDS__());

template<>
void GetScaleFloat<cpu>(mshadow::Stream<cpu> *s, const TBlob &scale_blob, float *pScalef) {
  MSHADOW_REAL_TYPE_SWITCH(scale_blob.type_flag_, DType,
    *pScalef = static_cast<float>(*scale_blob.dptr<DType>());
  )
}

std::vector<std::string> ParamToVector(uint32_t num_args, const char *pName[], size_t nParams) {
  std::vector<std::string> ret;
  for (uint32_t i = 0; i < num_args; ++i) {
    const auto idx = std::to_string(i);
    for (size_t j = 0; j < nParams; ++j)
      ret.push_back(std::string(pName[i]) + idx);
  }

  return ret;
}

inline uint32_t num_weights(const nnvm::NodeAttrs& attrs) {
  return static_cast<uint32_t>(dmlc::get<MultiAdamWParam>(attrs.parsed).num_weights);
}

NNVM_REGISTER_OP(_multi_adamw_update)
.describe(R"code(Update function for AdamW optimizer.

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
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs) * 4 + 1;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs);
  })
.set_attr_parser(ParamParser<MultiAdamWParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MP_MultiAdamW_InferShape<MultiAdamWParam, 4>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, -1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const char *paramName[] = {"weight_", "grad_", "mean_", "var_", "rescale_grad_"};
    return ParamToVector(num_weights(attrs), paramName, sizeof(paramName)/sizeof(paramName[0]));
  })
// mutable: mean, var
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const auto iMax = num_weights(attrs);
    for (size_t i = 0; i < iMax; ++i) {
      ret.push_back(i * 4 + 2);
      ret.push_back(i * 4 + 3);
    }
    return ret;
  })

.set_attr<FCompute>("FCompute<cpu>", multiMPUpdate<cpu, false>)
.add_argument("data", "NDArray-or-Symbol[]", "data")
.add_arguments(MultiAdamWParam::__FIELDS__());


NNVM_REGISTER_OP(_multi_mp_adamw_update)
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
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs) * 5 + 1;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs);
  })
.set_attr_parser(ParamParser<MultiAdamWParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MP_MultiAdamW_InferShape<MultiAdamWParam, 5>)
.set_attr<nnvm::FInferType>("FInferType", MP_MultiAdamW_InferType<MultiAdamWParam, 5, 1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const char *paramName[] = {"weight_", "grad_", "mean_", "var_", "weight32_", "rescale_grad_"};
    return ParamToVector(num_weights(attrs), paramName, sizeof(paramName)/sizeof(paramName[0]));
  })
// mutable: mean, var, weights32
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const auto iMax = num_weights(attrs);
    for (size_t i = 0; i < iMax; ++i) {
      ret.push_back(i * 5 + 2);
      ret.push_back(i * 5 + 3);
      ret.push_back(i * 5 + 4);
    }
    return ret;
  })

.set_attr<FCompute>("FCompute<cpu>", multiMPUpdate<cpu, true>)
.add_argument("data", "NDArray-or-Symbol[]", "data")
.add_arguments(MultiAdamWParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
