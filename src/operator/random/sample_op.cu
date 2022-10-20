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
 * \file sample_op.cu
 * \brief GPU Implementation of sample op
 */
#include "./sample_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_SAMPLE_GPU(name, ParamType)         \
  NNVM_REGISTER_OP(name)                                            \
      .set_attr<FCompute>("FCompute<gpu>", Sample_<gpu, ParamType>) \
      .set_attr<FComputeEx>("FComputeEx<gpu>", SampleEx_<gpu, ParamType>);

MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_uniform, SampleUniformParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_normal, SampleNormalParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_gamma, SampleGammaParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_exponential, SampleExponentialParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_poisson, SamplePoissonParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_binomial, SampleBinomialParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_negative_binomial, SampleNegBinomialParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_generalized_negative_binomial, SampleGenNegBinomialParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_randint, SampleRandIntParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_uniform_like, SampleUniformLikeParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_normal_like, SampleNormalLikeParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_gamma_like, SampleGammaLikeParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_exponential_like, SampleExponentialLikeParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_poisson_like, SamplePoissonLikeParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_binomial_like, SampleBinomialLikeParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_negative_binomial_like, SampleNegBinomialLikeParam)
MXNET_OPERATOR_REGISTER_SAMPLE_GPU(_random_generalized_negative_binomial_like,
                                   SampleGenNegBinomialLikeParam)

}  // namespace op
}  // namespace mxnet
