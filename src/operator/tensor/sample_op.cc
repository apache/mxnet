/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.cc
 * \brief CPU Implementation of sample op
 */
#include "./sample_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SampleUniformParam);
DMLC_REGISTER_PARAMETER(SampleNormalParam);
DMLC_REGISTER_PARAMETER(SampleGammaParam);
DMLC_REGISTER_PARAMETER(SampleExponentialParam);
DMLC_REGISTER_PARAMETER(SamplePoissonParam);
DMLC_REGISTER_PARAMETER(SampleNegBinomialParam);
DMLC_REGISTER_PARAMETER(SampleGenNegBinomialParam);

#define MXNET_OPERATOR_REGISTER_SAMPLE(name, ParamType)                 \
  NNVM_REGISTER_OP(name)                                                \
  .set_num_inputs(0)                                                    \
  .set_num_outputs(1)                                                   \
  .set_attr_parser(ParamParser<ParamType>)                              \
  .set_attr<nnvm::FInferShape>("FInferShape", InitShape<ParamType>)     \
  .set_attr<nnvm::FInferType>("FInferType", SampleOpType<ParamType>)    \
  .set_attr<FResourceRequest>("FResourceRequest", SampleResource)       \
  .add_arguments(ParamType::__FIELDS__())

// Add "uniform" alias for backward compatibility
MXNET_OPERATOR_REGISTER_SAMPLE(random_uniform, SampleUniformParam)
.add_alias("uniform")
.add_alias("_sample_uniform")
.describe(R"code(Draw samples from a uniform distribution.
Samples are uniformly distributed over the half-open interval [low, high)
(includes low, but excludes high). In other words, any value within the given
interval is equally likely to be drawn.
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleUniform_<cpu>);

// Add "normal" alias for backward compatibility
MXNET_OPERATOR_REGISTER_SAMPLE(random_normal, SampleNormalParam)
.add_alias("normal")
.add_alias("_sample_normal")
.describe("Draw samples from a normal (Gaussian) distribution.")
.set_attr<FCompute>("FCompute<cpu>", SampleNormal_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_gamma, SampleGammaParam)
.add_alias("_sample_gamma")
.describe("Draw samples from a gamma distribution.")
.set_attr<FCompute>("FCompute<cpu>", SampleGamma_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_exponential, SampleExponentialParam)
.add_alias("_sample_exponential")
.describe("Draw samples from an exponential distribution.")
.set_attr<FCompute>("FCompute<cpu>", SampleExponential_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_poisson, SamplePoissonParam)
.add_alias("_sample_poisson")
.describe("Draw samples from a Poisson distribution.")
.set_attr<FCompute>("FCompute<cpu>", SamplePoisson_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_negative_binomial, SampleNegBinomialParam)
.add_alias("_sample_negbinomial")
.describe("Draw samples from a negative binomial distribution.")
.set_attr<FCompute>("FCompute<cpu>", SampleNegBinomial_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_generalized_negative_binomial, SampleGenNegBinomialParam)
.add_alias("_sample_gennegbinomial")
.describe("Draw samples from a generalized negative binomial distribution.")
.set_attr<FCompute>("FCompute<cpu>", SampleGenNegBinomial_<cpu>);

}  // namespace op
}  // namespace mxnet
