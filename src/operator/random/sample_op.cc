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
 * \file sample_op.cc
 * \brief CPU Implementation of sample op
 */

#include "./sample_op.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SampleUniformParam);
DMLC_REGISTER_PARAMETER(SampleNormalParam);
DMLC_REGISTER_PARAMETER(SampleGammaParam);
DMLC_REGISTER_PARAMETER(SampleExponentialParam);
DMLC_REGISTER_PARAMETER(SamplePoissonParam);
DMLC_REGISTER_PARAMETER(SampleNegBinomialParam);
DMLC_REGISTER_PARAMETER(SampleGenNegBinomialParam);
DMLC_REGISTER_PARAMETER(SampleRandIntParam);

DMLC_REGISTER_PARAMETER(SampleUniformLikeParam);
DMLC_REGISTER_PARAMETER(SampleNormalLikeParam);
DMLC_REGISTER_PARAMETER(SampleGammaLikeParam);
DMLC_REGISTER_PARAMETER(SampleExponentialLikeParam);
DMLC_REGISTER_PARAMETER(SamplePoissonLikeParam);
DMLC_REGISTER_PARAMETER(SampleNegBinomialLikeParam);
DMLC_REGISTER_PARAMETER(SampleGenNegBinomialLikeParam);

#define MXNET_OPERATOR_REGISTER_SAMPLE(name, ParamType)                                      \
  NNVM_REGISTER_OP(name)                                                                     \
  .set_num_inputs(0)                                                                         \
  .set_num_outputs(1)                                                                        \
  .set_attr_parser(ParamParser<ParamType>)                                                   \
  .set_attr<mxnet::FInferShape>("FInferShape", InitShape<ParamType>)                          \
  .set_attr<nnvm::FInferType>("FInferType", SampleOpType<ParamType>)                \
  .set_attr<FResourceRequest>("FResourceRequest", SampleResource)                            \
  .add_arguments(ParamType::__FIELDS__())                                                    \
  .set_attr<FInferStorageType>("FInferStorageType", InitStorageType<ParamType, true, false>) \
  .set_attr<FCompute>("FCompute<cpu>", Sample_<cpu, ParamType>)                              \
  .set_attr<FComputeEx>("FComputeEx<cpu>", SampleEx_<cpu, ParamType>)

#define MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(name, ParamType)                              \
  NNVM_REGISTER_OP(name)                                                                  \
  .set_num_inputs(1)                                                                      \
  .set_num_outputs(1)                                                                     \
  .set_attr_parser(ParamParser<ParamType>)                                                \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)                        \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)                           \
  .set_attr<FResourceRequest>("FResourceRequest", SampleResource)                         \
  .set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",                                         \
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })                   \
  .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                              \
  .add_arguments(ParamType::__FIELDS__())                                                 \
  .add_argument("data", "NDArray-or-Symbol", "The input")                                 \
  .set_attr<FInferStorageType>("FInferStorageType",                                       \
                               ElemwiseStorageType<1, 1, false, true, false>)             \
  .set_attr<FCompute>("FCompute<cpu>", Sample_<cpu, ParamType>)                           \
  .set_attr<FComputeEx>("FComputeEx<cpu>", SampleEx_<cpu, ParamType>)

// Add "uniform" alias for backward compatibility
MXNET_OPERATOR_REGISTER_SAMPLE(_random_uniform, SampleUniformParam)
.add_alias("uniform")
.add_alias("random_uniform")
.add_alias("_npi_random_uniform")
.describe(R"code(Draw random samples from a uniform distribution.

.. note:: The existing alias ``uniform`` is deprecated.

Samples are uniformly distributed over the half-open interval *[low, high)*
(includes *low*, but excludes *high*).

Example::

   uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
                                          [ 0.54488319,  0.84725171]]

)code" ADD_FILELINE);

// Add "normal" alias for backward compatibility
MXNET_OPERATOR_REGISTER_SAMPLE(_random_normal, SampleNormalParam)
.add_alias("normal")
.add_alias("random_normal")
.add_alias("_npi_random_normal")
.describe(R"code(Draw random samples from a normal (Gaussian) distribution.

.. note:: The existing alias ``normal`` is deprecated.

Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*
(standard deviation).

Example::

   normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
                                          [-1.23474145,  1.55807114]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE(_random_gamma, SampleGammaParam)
.add_alias("random_gamma")
.describe(R"code(Draw random samples from a gamma distribution.

Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).

Example::

   gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
                                            [ 3.91697288,  3.65933681]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE(_random_exponential, SampleExponentialParam)
.add_alias("random_exponential")
.describe(R"code(Draw random samples from an exponential distribution.

Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).

Example::

   exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
                                      [ 0.04146638,  0.31715935]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE(_random_poisson, SamplePoissonParam)
.add_alias("random_poisson")
.describe(R"code(Draw random samples from a Poisson distribution.

Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
Samples will always be returned as a floating point data type.

Example::

   poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
                                  [ 4.,  6.]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE(_random_negative_binomial, SampleNegBinomialParam)
.add_alias("random_negative_binomial")
.describe(R"code(Draw random samples from a negative binomial distribution.

Samples are distributed according to a negative binomial distribution parametrized by
*k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
Samples will always be returned as a floating point data type.

Example::

   negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
                                                 [ 2.,  5.]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE(_random_generalized_negative_binomial, SampleGenNegBinomialParam)
.add_alias("random_generalized_negative_binomial")
.describe(R"code(Draw random samples from a generalized negative binomial distribution.

Samples are distributed according to a generalized negative binomial distribution parametrized by
*mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
number of unsuccessful experiments (generalized to real numbers).
Samples will always be returned as a floating point data type.

Example::

   generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],
                                                                    [ 6.,  4.]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE(_random_randint, SampleRandIntParam)
.add_alias("random_randint")
.describe(R"code(Draw random samples from a discrete uniform distribution.

Samples are uniformly distributed over the half-open interval *[low, high)*
(includes *low*, but excludes *high*).

Example::

   randint(low=0, high=5, shape=(2,2)) = [[ 0,  2],
                                          [ 3,  1]]

)code" ADD_FILELINE);

// *_like operators

MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(_random_uniform_like, SampleUniformLikeParam)
.describe(R"code(Draw random samples from a uniform distribution according to the input array shape.

Samples are uniformly distributed over the half-open interval *[low, high)*
(includes *low*, but excludes *high*).

Example::

   uniform(low=0, high=1, data=ones(2,2)) = [[ 0.60276335,  0.85794562],
                                             [ 0.54488319,  0.84725171]]

)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(_random_normal_like, SampleNormalLikeParam)
.describe(R"code(Draw random samples from a normal (Gaussian) distribution according to the input array shape.

Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*
(standard deviation).

Example::

   normal(loc=0, scale=1, data=ones(2,2)) = [[ 1.89171135, -1.16881478],
                                             [-1.23474145,  1.55807114]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(_random_gamma_like, SampleGammaLikeParam)
.describe(R"code(Draw random samples from a gamma distribution according to the input array shape.

Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).

Example::

   gamma(alpha=9, beta=0.5, data=ones(2,2)) = [[ 7.10486984,  3.37695289],
                                               [ 3.91697288,  3.65933681]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(_random_exponential_like, SampleExponentialLikeParam)
.describe(R"code(Draw random samples from an exponential distribution according to the input array shape.

Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).

Example::

   exponential(lam=4, data=ones(2,2)) = [[ 0.0097189 ,  0.08999364],
                                         [ 0.04146638,  0.31715935]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(_random_poisson_like, SamplePoissonLikeParam)
.describe(R"code(Draw random samples from a Poisson distribution according to the input array shape.

Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
Samples will always be returned as a floating point data type.

Example::

   poisson(lam=4, data=ones(2,2)) = [[ 5.,  2.],
                                     [ 4.,  6.]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(_random_negative_binomial_like, SampleNegBinomialLikeParam)
.describe(R"code(Draw random samples from a negative binomial distribution according to the input array shape.

Samples are distributed according to a negative binomial distribution parametrized by
*k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
Samples will always be returned as a floating point data type.

Example::

   negative_binomial(k=3, p=0.4, data=ones(2,2)) = [[ 4.,  7.],
                                                    [ 2.,  5.]]
)code" ADD_FILELINE);

MXNET_OPERATOR_REGISTER_SAMPLE_LIKE(_random_generalized_negative_binomial_like,
                                    SampleGenNegBinomialLikeParam)
.describe(R"code(Draw random samples from a generalized negative binomial distribution according to the
input array shape.

Samples are distributed according to a generalized negative binomial distribution parametrized by
*mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
number of unsuccessful experiments (generalized to real numbers).
Samples will always be returned as a floating point data type.

Example::

   generalized_negative_binomial(mu=2.0, alpha=0.3, data=ones(2,2)) = [[ 2.,  1.],
                                                                       [ 6.,  4.]]
)code" ADD_FILELINE);

}  // namespace op
}  // namespace mxnet
