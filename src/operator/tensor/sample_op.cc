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
.describe(R"code(Draw random samples from a uniform distribution.

.. note:: The existing alias ``uniform`` is deprecated.

Samples are uniformly distributed over the half-open interval *[low, high)*
(includes *low*, but excludes *high*).

Example::

   random_uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
                                                 [ 0.54488319,  0.84725171]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleUniform_<cpu>);

// Add "normal" alias for backward compatibility
MXNET_OPERATOR_REGISTER_SAMPLE(random_normal, SampleNormalParam)
.add_alias("normal")
.add_alias("_sample_normal")
.describe(R"code(Draw random samples from a normal (Gaussian) distribution.

.. note:: The existing alias ``normal`` is deprecated.

Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale* (standard deviation).

Example::

   random_normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
                                                 [-1.23474145,  1.55807114]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleNormal_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_gamma, SampleGammaParam)
.add_alias("_sample_gamma")
.describe(R"code(Draw random samples from a gamma distribution.

Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).

Example::

   random_gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
                                                   [ 3.91697288,  3.65933681]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleGamma_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_exponential, SampleExponentialParam)
.add_alias("_sample_exponential")
.describe(R"code(Draw random samples from an exponential distribution.

Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).

Example::

   random_exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
                                             [ 0.04146638,  0.31715935]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleExponential_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_poisson, SamplePoissonParam)
.add_alias("_sample_poisson")
.describe(R"code(Draw random samples from a Poisson distribution.

Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
Samples will always be returned as a floating point data type.

Example::

   random_poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
                                         [ 4.,  6.]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SamplePoisson_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_negative_binomial, SampleNegBinomialParam)
.add_alias("_sample_negbinomial")
.describe(R"code(Draw random samples from a negative binomial distribution.

Samples are distributed according to a negative binomial distribution parametrized by 
*k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
Samples will always be returned as a floating point data type.

Example::

   random_negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
                                                        [ 2.,  5.]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleNegBinomial_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(random_generalized_negative_binomial, SampleGenNegBinomialParam)
.add_alias("_sample_gennegbinomial")
.describe(R"code(Draw random samples from a generalized negative binomial distribution.

Samples are distributed according to a generalized negative binomial distribution parametrized by 
*mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the 
number of unsuccessful experiments (generalized to real numbers).
Samples will always be returned as a floating point data type.

Example::

   random_generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],
                                                                           [ 6.,  4.]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleGenNegBinomial_<cpu>);

}  // namespace op
}  // namespace mxnet
