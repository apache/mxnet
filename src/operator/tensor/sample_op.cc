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
(includes low, but excludes high)::

  nd.uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
                                            [ 0.54488319,  0.84725171]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleUniform_<cpu>);

// Add "normal" alias for backward compatibility
MXNET_OPERATOR_REGISTER_SAMPLE(random_normal, SampleNormalParam)
.add_alias("normal")
.add_alias("_sample_normal")
.describe(R"code(Draw random samples from a normal (Gaussian) distribution.

Examples::

  normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
                                         [-1.23474145,  1.55807114]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleNormal_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_gamma, SampleGammaParam)
.add_alias("_sample_gamma")
.describe("Sample a gamma distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleGamma_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_exponential, SampleExponentialParam)
.add_alias("_sample_exponential")
.describe("Sample an exponential distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleExponential_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_poisson, SamplePoissonParam)
.add_alias("_sample_poisson")
.describe("Sample a Poisson distribution")
.set_attr<FCompute>("FCompute<cpu>", SamplePoisson_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_negative_binomial, SampleNegBinomialParam)
.add_alias("_sample_negbinomial")
.describe("Sample a negative binomial distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleNegBinomial_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_generalized_negative_binomial, SampleGenNegBinomialParam)
.add_alias("_sample_gennegbinomial")
.describe("Sample a generalized negative binomial distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleGenNegBinomial_<cpu>);

}  // namespace op
}  // namespace mxnet
