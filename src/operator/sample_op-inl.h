/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op-inl.h
 * \brief Function defintion sampling operators.
 */
#ifndef MXNET_OPERATOR_SAMPLE_OP_INL_H_
#define MXNET_OPERATOR_SAMPLE_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

struct SampleUniformParam : public dmlc::Parameter<SampleUniformParam> {
  float low;
  float high;
  TShape shape;
  DMLC_DECLARE_PARAMETER(SampleUniformParam) {
    DMLC_DECLARE_FIELD(low).set_default(0.0f)
        .describe("The lower bound of distribution");
    DMLC_DECLARE_FIELD(high).set_default(1.0f)
        .describe("The upper bound of distribution");
    DMLC_DECLARE_FIELD(shape)
        .describe("The shape of the output");
  }
};

struct SampleNormalParam : public dmlc::Parameter<SampleNormalParam> {
  float loc;
  float scale;
  TShape shape;
  DMLC_DECLARE_PARAMETER(SampleNormalParam) {
    DMLC_DECLARE_FIELD(loc).set_default(0.0f)
        .describe("Mean of the distribution.");
    DMLC_DECLARE_FIELD(scale).set_default(1.0f)
        .describe("Standard deviation of the distribution.");
    DMLC_DECLARE_FIELD(shape)
        .describe("The shape of the output");
  }
};

template<typename xpu>
void SampleUniform_(const EnvArguments& env,
                    TBlob *ret,
                    OpReqType req,
                    RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, mshadow::kFloat32)
      << "only support float32 rnd so far";
  SampleUniformParam param;
  param.Init(env.kwargs);
  mshadow::Random<xpu, float> *prnd = env.resource[0].get_random<xpu, float>(s);
  mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
  prnd->SampleUniform(&tmp, float(param.low), float(param.high));  // NOLINT(*)
}

template<typename xpu>
void SampleNormal_(const EnvArguments& env,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, mshadow::kFloat32)
      << "only support float32 rnd so far";
  SampleNormalParam param;
  param.Init(env.kwargs);
  mshadow::Random<xpu, float> *prnd = env.resource[0].get_random<xpu, float>(s);
  mshadow::Tensor<xpu, 2, float> tmp = ret->FlatTo2D<xpu, float>(s);
  prnd->SampleGaussian(&tmp, float(param.loc), float(param.scale));  // NOLINT(*)
}

template<typename ParamType>
inline TShape SampleShape(const EnvArguments& env) {
  ParamType param;
  param.Init(env.kwargs);
  return param.shape;
}

// sample uniform
MXNET_REGISTER_SIMPLE_OP(_sample_uniform, XPU)
.set_symbol_op_name("uniform")
.set_enable_kwargs(true)
.set_resource_request(ResourceRequest::kRandom)
.set_function(XPU::kDevMask, SampleUniform_<XPU>)
.set_shape_function(SampleShape<SampleUniformParam>)
.describe("Sample a uniform distribution")
.add_arguments(SampleUniformParam::__FIELDS__());

// sample normal
MXNET_REGISTER_SIMPLE_OP(_sample_normal, XPU)
.set_symbol_op_name("normal")
.set_enable_kwargs(true)
.set_resource_request(ResourceRequest::kRandom)
.set_function(XPU::kDevMask, SampleNormal_<XPU>)
.set_shape_function(SampleShape<SampleNormalParam>)
.describe("Sample a normal distribution")
.add_arguments(SampleNormalParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SAMPLE_OP_INL_H_
