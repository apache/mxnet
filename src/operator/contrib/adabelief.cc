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
 *  Copyright (c) 2021 by Contributors
 * \file adabelief.cc
 * \brief Optimizer operators
 * \author khaotik
 */
#include "./adabelief-inl.h"

namespace mxnet {
namespace op {
namespace adabelief {

DMLC_REGISTER_PARAMETER(AdaBeliefParam);
DMLC_REGISTER_PARAMETER(MultiAdaBeliefParam);

NNVM_REGISTER_OP(_mp_adabelief_update)
.describe(R"code(Update function for multi-precision AdaBelief optimizer.

AdaBelief is seen as a modification of Adam with a different variance 
estimator.

Adam update consists of the following steps, where g represents gradient and m, s
are 1st and 2nd order moment estimates (mean and variance).

.. math::

 g_t = \nabla J(W_{t-1}) + w * wd \\
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 s_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t - m_t)^2 + \epsilon\\
 W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon })

It updates the weights using::

 m = beta1*m + (1-beta1)*grad
 s = beta2*v + (1-beta2)*(grad**2)
 w -= eta * (learning_rate * m / (sqrt(s) + epsilon))

Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
the update is skipped.
)code" ADD_FILELINE)
.set_num_inputs(6)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdaBeliefParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MPUpdateInferShape<2, 1, 6>)
.set_attr<nnvm::FInferType>("FInferType", MPUpdateInferType<2, 1, 6>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3, 4};
  })
.set_attr<FCompute>("FCompute<cpu>", MPUpdate<cpu, MPAdaBeliefUpdate<cpu>>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_argument("rescale_grad", "NDArray-or-Symbol",
              "Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, "
              "the update is skipped.")
.add_arguments(AdaBeliefParam::__FIELDS__());

NNVM_REGISTER_OP(_adabelief_update)
.describe(R"code(Update function for AdaBelief optimizer.

AdaBelief is seen as a modification of Adam with a different variance 
estimator.

Adam update consists of the following steps, where g represents gradient and m, s
are 1st and 2nd order moment estimates (mean and variance).

.. math::

 g_t = \nabla J(W_{t-1}) + w * wd \\
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 s_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t - m_t)^2 + \epsilon\\
 W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon })

It updates the weights using::

 m = beta1*m + (1-beta1)*grad
 s = beta2*v + (1-beta2)*(grad**2)
 w -= eta * (learning_rate * m / (sqrt(s) + epsilon))

Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
the update is skipped.
))code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdaBeliefParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MPUpdateInferShape<4, 1, 5>)
.set_attr<nnvm::FInferType>("FInferType", MPUpdateInferType<4, 1, 5>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", MPUpdate<cpu, AdaBeliefUpdate<cpu>>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_argument("rescale_grad", "NDArray-or-Symbol",
              "Rescale gradient to rescale_grad * grad. If NaN, Inf, or 0, "
              "the update is skipped.")
.add_arguments(AdaBeliefParam::__FIELDS__());

template<>
void GetScaleFloat<cpu>(mshadow::Stream<cpu> *s, const TBlob &scale_blob, float *pScalef) {
  MSHADOW_REAL_TYPE_SWITCH(scale_blob.type_flag_, DType,
    *pScalef = static_cast<float>(*scale_blob.dptr<DType>());
  )
}

static std::vector<std::string>
ParamToVector(uint32_t num_args, const char *pName[], size_t nParams) {
  std::vector<std::string> ret;
  for (uint32_t i = 0; i < num_args; ++i) {
    const auto idx = std::to_string(i);
    for (size_t j = 0; j < nParams; ++j)
      ret.push_back(std::string(pName[i]) + idx);
  }

  return ret;
}

inline uint32_t num_weights(const nnvm::NodeAttrs& attrs) {
  return static_cast<uint32_t>(dmlc::get<MultiAdaBeliefParam>(attrs.parsed).num_weights);
}

NNVM_REGISTER_OP(_multi_adabelief_update)
.describe(R"code(Update function for AdaBelief optimizer.

AdaBelief is seen as a modification of Adam with a different variance 
estimator.

Adam update consists of the following steps, where g represents gradient and m, s
are 1st and 2nd order moment estimates (mean and variance).

.. math::

 g_t = \nabla J(W_{t-1}) + w * wd \\
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 s_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t - m_t)^2\\
 W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon })

It updates the weights using::

 m = beta1*m + (1-beta1)*grad
 s = beta2*v + (1-beta2)*(grad**2)
 w -= eta * (learning_rate * m / (sqrt(s) + epsilon))

Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
the update is skipped.
))code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs) * 4 + 1;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs);
  })
.set_attr_parser(ParamParser<MultiAdaBeliefParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MP_MultiAdaBelief_InferShape<MultiAdaBeliefParam, 4>)
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
.add_arguments(MultiAdaBeliefParam::__FIELDS__());


NNVM_REGISTER_OP(_multi_mp_adabelief_update)
.describe(R"code(Update function for multi-precision AdaBelief optimizer.

AdaBelief is seen as a modification of Adam with a different variance 
estimator.

Adam update consists of the following steps, where g represents gradient and m, s
are 1st and 2nd order moment estimates (mean and variance).

.. math::

 g_t = \nabla J(W_{t-1}) + w * wd \\
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 s_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t - m_t)^2 + \epsilon\\
 W_t = W_{t-1} - \eta_t (\alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon })

It updates the weights using::

 m = beta1*m + (1-beta1)*grad
 s = beta2*v + (1-beta2)*(grad**2)
 w -= eta * (learning_rate * m / (sqrt(s) + epsilon))

Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
the update is skipped.
))code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs) * 5 + 1;
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    return num_weights(attrs);
  })
.set_attr_parser(ParamParser<MultiAdaBeliefParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MP_MultiAdaBelief_InferShape<MultiAdaBeliefParam, 5>)
.set_attr<nnvm::FInferType>("FInferType", MP_MultiAdaBelief_InferType<MultiAdaBeliefParam, 5, 1>)
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
.add_arguments(MultiAdaBeliefParam::__FIELDS__());

}  // namespace adabelief
}  // namespace op
}  // namespace mxnet
