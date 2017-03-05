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

MXNET_OPERATOR_REGISTER_SAMPLE(uniform, SampleUniformParam)
.add_alias("_sample_uniform")
.describe(R"code(Draw samples from a uniform distribution.

Samples are uniformly distributed over the half-open interval [low, high)
(includes low, but excludes high)::

  nd.uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
                                            [ 0.54488319,  0.84725171]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleUniform_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(normal, SampleNormalParam)
.add_alias("_sample_normal")
.describe(R"code(Draw random samples from a normal (Gaussian) distribution.

Examples::

  normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
                                         [-1.23474145,  1.55807114]]
)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", SampleNormal_<cpu>);

}  // namespace op
}  // namespace mxnet
