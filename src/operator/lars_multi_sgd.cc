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
 *  Copyright (c) 2019 by Contributors
 * \file lars_multi_sgd.cc
 * \brief Multi-sgd optimizers with lrs and wds as mxnet inputs
 * \author Clement Fuji Tsang
 */

#include <string>

#include "lars_multi_sgd-inl.h"
#include "elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LARSMultiSGDParam);
DMLC_REGISTER_PARAMETER(LARSMultiSGDMomParam);

NNVM_REGISTER_OP(lars_multi_sgd_update)
.describe(R"code(Update function for Stochastic Gradient Descent (SDG) optimizer.

It updates the weights using::

 weight = weight - learning_rate * (gradient + wd * weight)

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDParam& param = dmlc::get<LARSMultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 2 + 2);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDParam& param = dmlc::get<LARSMultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<LARSMultiSGDParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LARSMultiSGDShape<LARSMultiSGDParam, 2>)
.set_attr<nnvm::FInferType>("FInferType",
                            MP_LARSMultiSGD_InferType<LARSMultiSGDParam, 2, 0>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<LARSMultiSGDParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    ret.reserve(num_args * 2 + 2);
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
    }
    ret.emplace_back("lrs");
    ret.emplace_back("wds");
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", LARSMultiSGDUpdate<cpu, lars_type_identity, 2>)
.add_argument("data", "NDArray-or-Symbol[]", "Weights, gradients, learning rates and weight decays")
.add_arguments(LARSMultiSGDParam::__FIELDS__());

NNVM_REGISTER_OP(lars_multi_sgd_mom_update)
.describe(R"code(Momentum update function for Stochastic Gradient Descent (SGD) optimizer.

Momentum update has better convergence rates on neural networks. Mathematically it looks
like below:

.. math::

  v_1 = \alpha * \nabla J(W_0)\\
  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  W_t = W_{t-1} + v_t

It updates the weights using::

  v = momentum * v - learning_rate * gradient
  weight += v

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDMomParam& param = dmlc::get<LARSMultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 3 + 2);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDMomParam& param = dmlc::get<LARSMultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<LARSMultiSGDMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LARSMultiSGDShape<LARSMultiSGDMomParam, 3>)
.set_attr<nnvm::FInferType>("FInferType",
                            MP_LARSMultiSGD_InferType<LARSMultiSGDMomParam, 3, 0>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<LARSMultiSGDParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    ret.reserve(num_args * 3 + 2);
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
      ret.push_back(std::string("mom_") + std::to_string(i));
    }
    ret.emplace_back("lrs");
    ret.emplace_back("wds");
    return ret;
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const LARSMultiSGDMomParam& param = dmlc::get<LARSMultiSGDMomParam>(attrs.parsed);
    ret.reserve(param.num_weights);
    for (int i = 0; i < param.num_weights; ++i) {
      ret.push_back(i * 3 + 2);
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", LARSMultiSGDMomUpdate<cpu, lars_type_identity, 3>)
.add_argument("data", "NDArray-or-Symbol[]",
              "Weights, gradients, momentum, learning rates and weight decays")
.add_arguments(LARSMultiSGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(lars_multi_mp_sgd_update)
.describe(R"code(Update function for multi-precision Stochastic Gradient Descent (SDG) optimizer.

It updates the weights using::

 weight = weight - learning_rate * (gradient + wd * weight)

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDParam& param = dmlc::get<LARSMultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 3 + 2);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDParam& param = dmlc::get<LARSMultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<LARSMultiSGDParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LARSMultiSGDShape<LARSMultiSGDParam, 3>)
.set_attr<nnvm::FInferType>("FInferType",
                            MP_LARSMultiSGD_InferType<LARSMultiSGDParam, 3, 1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<LARSMultiSGDParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    ret.reserve(num_args * 3 + 2);
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
      ret.push_back(std::string("weight32_") + std::to_string(i));
    }
    ret.emplace_back("lrs");
    ret.emplace_back("wds");
    return ret;
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const LARSMultiSGDParam& param = dmlc::get<LARSMultiSGDParam>(attrs.parsed);
    ret.reserve(param.num_weights);
    for (int i = 0; i < param.num_weights; ++i) {
      ret.push_back(i * 3 + 2);
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", LARSMultiSGDUpdate<cpu, lars_single_precision, 3>)
.add_argument("data", "NDArray-or-Symbol[]", "Weights, gradients, learning rates and weight decays")
.add_arguments(LARSMultiSGDParam::__FIELDS__());

NNVM_REGISTER_OP(lars_multi_mp_sgd_mom_update)
.describe(R"code(Momentum update function for multi-precision Stochastic Gradient Descent (SGD) optimizer.

Momentum update has better convergence rates on neural networks. Mathematically it looks
like below:

.. math::

  v_1 = \alpha * \nabla J(W_0)\\
  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  W_t = W_{t-1} + v_t

It updates the weights using::

  v = momentum * v - learning_rate * gradient
  weight += v

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDMomParam& param = dmlc::get<LARSMultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 4 + 2);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const LARSMultiSGDMomParam& param = dmlc::get<LARSMultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<LARSMultiSGDMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LARSMultiSGDShape<LARSMultiSGDMomParam, 4>)
.set_attr<nnvm::FInferType>("FInferType",
                            MP_LARSMultiSGD_InferType<LARSMultiSGDMomParam, 4, 2>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<LARSMultiSGDMomParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    ret.reserve(num_args * 4 + 2);
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
      ret.push_back(std::string("mom_") + std::to_string(i));
      ret.push_back(std::string("weight32_") + std::to_string(i));
    }
    ret.emplace_back("lrs");
    ret.emplace_back("wds");
    return ret;
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const LARSMultiSGDMomParam& param = dmlc::get<LARSMultiSGDMomParam>(attrs.parsed);
    ret.reserve(param.num_weights * 2);
    for (int i = 0; i < param.num_weights; ++i) {
      ret.push_back(i * 4 + 2);
      ret.push_back(i * 4 + 3);
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", LARSMultiSGDMomUpdate<cpu, lars_single_precision, 4>)
.add_argument("data", "NDArray-or-Symbol[]",
              "Weights, gradients, momentums, learning rates and weight decays")
.add_arguments(LARSMultiSGDMomParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
