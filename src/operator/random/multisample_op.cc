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
 * \file multisample_op.cc
 * \brief CPU-implementation of multi-sampling operators
 */

#include "./multisample_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MultiSampleParam);

#define MXNET_OPERATOR_REGISTER_SAMPLING(distr,                                            \
                                         sampler,                                          \
                                         num_inputs,                                       \
                                         input_name_1,                                     \
                                         input_name_2,                                     \
                                         input_desc_1,                                     \
                                         input_desc_2,                                     \
                                         description)                                      \
  NNVM_REGISTER_OP(_sample_##distr)                                                        \
      .add_alias("sample_" #distr)                                                         \
      .describe(description() + std::string(ADD_FILELINE))                                 \
      .set_num_inputs(num_inputs)                                                          \
      .set_num_outputs(1)                                                                  \
      .set_attr_parser(ParamParser<MultiSampleParam>)                                      \
      .set_attr<nnvm::FListInputNames>(                                                    \
          "FListInputNames",                                                               \
          [](const NodeAttrs& attrs) {                                                     \
            std::vector<std::string> v = {input_name_1, input_name_2};                     \
            v.resize(num_inputs);                                                          \
            return v;                                                                      \
          })                                                                               \
      .set_attr<mxnet::FInferShape>("FInferShape", MultiSampleOpShape)                     \
      .set_attr<nnvm::FInferType>("FInferType", MultiSampleOpType)                         \
      .set_attr<FResourceRequest>("FResourceRequest",                                      \
                                  [](const NodeAttrs& attrs) {                             \
                                    return std::vector<ResourceRequest>{                   \
                                        ResourceRequest::kParallelRandom,                  \
                                        ResourceRequest::kTempSpace};                      \
                                  })                                                       \
      .set_attr<FCompute>("FCompute<cpu>", MultiSampleOpForward<cpu, sampler, num_inputs>) \
      .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)                           \
      .add_argument(input_name_1, "NDArray-or-Symbol", input_desc_1)                       \
      .add_arguments(MultiSampleParam::__FIELDS__())

#define MXNET_OPERATOR_REGISTER_SAMPLING1(distr, sampler, input_name, input_desc, description) \
  MXNET_OPERATOR_REGISTER_SAMPLING(                                                            \
      distr, sampler, 1, input_name, input_name, input_desc, input_desc, description)

#define MXNET_OPERATOR_REGISTER_SAMPLING2(                                                    \
    distr, sampler, input_name_1, input_name_2, input_desc_1, input_desc_2, description)      \
  MXNET_OPERATOR_REGISTER_SAMPLING(                                                           \
      distr, sampler, 2, input_name_1, input_name_2, input_desc_1, input_desc_2, description) \
      .add_argument(input_name_2, "NDArray-or-Symbol", input_desc_2)

inline std::string uniform_desc() {
  return std::string(R"code(Concurrent sampling from multiple
uniform distributions on the intervals given by *[low,high)*.

The parameters of the distributions are provided as input arrays.
Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input values at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input arrays.

Examples::

   low = [ 0.0, 2.5 ]
   high = [ 1.0, 3.7 ]

   // Draw a single sample for each distribution
   sample_uniform(low, high) = [ 0.40451524,  3.18687344]

   // Draw a vector containing two samples for each distribution
   sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],
                                           [ 3.18687344,  3.68352246]]
)code");
}

inline std::string normal_desc() {
  return std::string(R"code(Concurrent sampling from multiple
normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).

The parameters of the distributions are provided as input arrays.
Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input values at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input arrays.

Examples::

   mu = [ 0.0, 2.5 ]
   sigma = [ 1.0, 3.7 ]

   // Draw a single sample for each distribution
   sample_normal(mu, sigma) = [-0.56410581,  0.95934606]

   // Draw a vector containing two samples for each distribution
   sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],
                                          [ 0.95934606,  4.48287058]]
)code");
}

inline std::string gamma_desc() {
  return std::string(R"code(Concurrent sampling from multiple
gamma distributions with parameters *alpha* (shape) and *beta* (scale).

The parameters of the distributions are provided as input arrays.
Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input values at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input arrays.

Examples::

   alpha = [ 0.0, 2.5 ]
   beta = [ 1.0, 0.7 ]

   // Draw a single sample for each distribution
   sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]

   // Draw a vector containing two samples for each distribution
   sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],
                                           [ 2.25797319,  1.70734084]]
)code");
}

inline std::string exponential_desc() {
  return std::string(R"code(Concurrent sampling from multiple
exponential distributions with parameters lambda (rate).

The parameters of the distributions are provided as an input array.
Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input value at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input array.

Examples::

   lam = [ 1.0, 8.5 ]

   // Draw a single sample for each distribution
   sample_exponential(lam) = [ 0.51837951,  0.09994757]

   // Draw a vector containing two samples for each distribution
   sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],
                                         [ 0.09994757,  0.50447971]]
)code");
}

inline std::string poisson_desc() {
  return std::string(R"code(Concurrent sampling from multiple
Poisson distributions with parameters lambda (rate).

The parameters of the distributions are provided as an input array.
Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input value at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input array.

Samples will always be returned as a floating point data type.

Examples::

   lam = [ 1.0, 8.5 ]

   // Draw a single sample for each distribution
   sample_poisson(lam) = [  0.,  13.]

   // Draw a vector containing two samples for each distribution
   sample_poisson(lam, shape=(2)) = [[  0.,   4.],
                                     [ 13.,   8.]]
)code");
}

inline std::string binomial_desc() {
  return std::string(R"code(Concurrent sampling from multiple
binomial distributions with parameters *n* (number of trials) and *p* (success probability).

The parameters of the distributions are provided as input arrays.
Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input values at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input arrays.

Samples will always be returned as a floating point data type.

Examples::

   n = [ 20, 49 ]
   p = [ 0.4 , 0.77 ]

   // Draw a single sample for each distribution
   sample_binomial(n, p) = [ 5.,  36.]

   // Draw a vector containing two samples for each distribution
   sample_binomial(n, p, shape=(2)) = [[ 5.,  40.],
                                       [ 11.,  35.]]
)code");
}

inline std::string negative_binomial_desc() {
  return std::string(R"code(Concurrent sampling from multiple
negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).

The parameters of the distributions are provided as input arrays.
Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input values at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input arrays.

Samples will always be returned as a floating point data type.

Examples::

   k = [ 20, 49 ]
   p = [ 0.4 , 0.77 ]

   // Draw a single sample for each distribution
   sample_negative_binomial(k, p) = [ 15.,  16.]

   // Draw a vector containing two samples for each distribution
   sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],
                                                [ 16.,  12.]]
)code");
}

inline std::string generalized_negative_binomial_desc() {
  return std::string(R"code(Concurrent sampling from multiple
generalized negative binomial distributions with parameters *mu* (mean) and *alpha* (dispersion).

The parameters of the distributions are provided as input arrays.
Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
be the shape specified as the parameter of the operator, and *m* be the dimension
of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.

For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
will be an *m*-dimensional array that holds randomly drawn samples from the distribution
which is parameterized by the input values at index *i*. If the shape parameter of the
operator is not set, then one sample will be drawn per distribution and the output array
has the same shape as the input arrays.

Samples will always be returned as a floating point data type.

Examples::

   mu = [ 2.0, 2.5 ]
   alpha = [ 1.0, 0.1 ]

   // Draw a single sample for each distribution
   sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]

   // Draw a vector containing two samples for each distribution
   sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],
                                                                 [ 3.,  1.]]
)code");
}

MXNET_OPERATOR_REGISTER_SAMPLING2(uniform,
                                  UniformSampler<cpu>,
                                  "low",
                                  "high",
                                  "Lower bounds of the distributions.",
                                  "Upper bounds of the distributions.",
                                  uniform_desc);
MXNET_OPERATOR_REGISTER_SAMPLING2(normal,
                                  NormalSampler<cpu>,
                                  "mu",
                                  "sigma",
                                  "Means of the distributions.",
                                  "Standard deviations of the distributions.",
                                  normal_desc);
MXNET_OPERATOR_REGISTER_SAMPLING2(gamma,
                                  GammaSampler<cpu>,
                                  "alpha",
                                  "beta",
                                  "Alpha (shape) parameters of the distributions.",
                                  "Beta (scale) parameters of the distributions.",
                                  gamma_desc);
MXNET_OPERATOR_REGISTER_SAMPLING1(exponential,
                                  ExponentialSampler<cpu>,
                                  "lam",
                                  "Lambda (rate) parameters of the distributions.",
                                  exponential_desc);
MXNET_OPERATOR_REGISTER_SAMPLING1(poisson,
                                  PoissonSampler<cpu>,
                                  "lam",
                                  "Lambda (rate) parameters of the distributions.",
                                  poisson_desc)
    .add_alias("_npx_tensor_poisson");
MXNET_OPERATOR_REGISTER_SAMPLING2(binomial,
                                  BinomialSampler<cpu>,
                                  "n",
                                  "p",
                                  "Number of experiments.",
                                  "Success probabilities in each experiment.",
                                  binomial_desc);
MXNET_OPERATOR_REGISTER_SAMPLING2(negative_binomial,
                                  NegativeBinomialSampler<cpu>,
                                  "k",
                                  "p",
                                  "Limits of unsuccessful experiments.",
                                  "Failure probabilities in each experiment.",
                                  negative_binomial_desc);
MXNET_OPERATOR_REGISTER_SAMPLING2(generalized_negative_binomial,
                                  GeneralizedNegativeBinomialSampler<cpu>,
                                  "mu",
                                  "alpha",
                                  "Means of the distributions.",
                                  "Alpha (dispersion) parameters of the distributions.",
                                  generalized_negative_binomial_desc);

}  // namespace op
}  // namespace mxnet
