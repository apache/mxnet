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
.describe("Sample a uniform distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleUniform_<cpu>);

MXNET_OPERATOR_REGISTER_SAMPLE(normal, SampleNormalParam)
.add_alias("_sample_normal")
.describe("Sample a normal distribution")
.set_attr<FCompute>("FCompute<cpu>", SampleNormal_<cpu>);

}  // namespace op
}  // namespace mxnet
