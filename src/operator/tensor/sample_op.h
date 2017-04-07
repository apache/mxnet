/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.h
 * \brief Elementary sampling operators
 */
#ifndef MXNET_OPERATOR_TENSOR_SAMPLE_OP_H_
#define MXNET_OPERATOR_TENSOR_SAMPLE_OP_H_

#include <mxnet/operator_util.h>
#include <mshadow/base.h>
#include <string>
#include <vector>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./init_op.h"

namespace mxnet {
namespace op {

struct SampleUniformParam : public dmlc::Parameter<SampleUniformParam> {
  float low;
  float high;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleUniformParam) {
    DMLC_DECLARE_FIELD(low).set_default(0.0f)
    .describe("The lower bound of distribution");
    DMLC_DECLARE_FIELD(high).set_default(1.0f)
    .describe("The upper bound of distribution");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output");
  }
};

struct SampleNormalParam : public dmlc::Parameter<SampleNormalParam> {
  float loc;
  float scale;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleNormalParam) {
    DMLC_DECLARE_FIELD(loc).set_default(0.0f)
    .describe("Mean of the distribution.");
    DMLC_DECLARE_FIELD(scale).set_default(1.0f)
    .describe("Standard deviation of the distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output");
  }
};

struct SampleGammaParam : public dmlc::Parameter<SampleGammaParam> {
  float alpha;
  float beta;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleGammaParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f)
    .describe("alpha parameter (shape parameter) of the gamma distribution.");
    DMLC_DECLARE_FIELD(beta).set_default(1.0f)
    .describe("beta parameter (scale parameter) of the gamma distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output");
  }
};

struct SampleExponentialParam : public dmlc::Parameter<SampleExponentialParam> {
  float lambda;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleExponentialParam) {
    DMLC_DECLARE_FIELD(lambda).set_default(1.0f)
    .describe("lambda parameter (rate) of the exponential distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output");
  }
};

struct SamplePoissonParam : public dmlc::Parameter<SamplePoissonParam> {
  float lambda;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SamplePoissonParam) {
    DMLC_DECLARE_FIELD(lambda).set_default(1.0f)
    .describe("lambda parameter (rate) of the Poisson distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output");
  }
};

struct SampleNegBinomialParam : public dmlc::Parameter<SampleNegBinomialParam> {
  int k;
  float p;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleNegBinomialParam) {
    DMLC_DECLARE_FIELD(k).set_default(1)
    .describe("limit of unsuccessful tries.");
    DMLC_DECLARE_FIELD(p).set_default(1.0f)
    .describe("success probability.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output");
  }
};

struct SampleGenNegBinomialParam : public dmlc::Parameter<SampleGenNegBinomialParam> {
  float mu;
  float alpha;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleGenNegBinomialParam) {
    DMLC_DECLARE_FIELD(mu).set_default(1.0f)
    .describe("mean of the negative binomial distribution.");
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f)
    .describe("alpha parameter of the negative binomial distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("The shape of the output");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              "Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output");
  }
};

template<typename xpu>
void SampleUniform_(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleUniformParam& param = nnvm::get<SampleUniformParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleUniform(&out, param.low, param.high);
  });
}

template<typename xpu>
void SampleNormal_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleNormalParam& param = nnvm::get<SampleNormalParam>(attrs.parsed);
  CHECK_GT(param.scale, 0) << "scale parameter in gaussian has to be positive";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleGaussian(&out, param.loc, param.scale);  // NOLINT(*)
  });
}

template<typename xpu>
void SampleGamma_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleGammaParam& param = nnvm::get<SampleGammaParam>(attrs.parsed);
  CHECK_GT(param.alpha, 0) << "alpha parameter in gamma distribution has to be positive";
  CHECK_GT(param.beta, 0) << "beta parameter in gamma distribution has to be positive";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleGamma(&out, param.alpha, param.beta);  // NOLINT(*)
  });
}

template<typename xpu>
void SampleExponential_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleExponentialParam& param = nnvm::get<SampleExponentialParam>(attrs.parsed);
  CHECK_GT(param.lambda, 0) << "lambda parameter in exponential distribution has to be positive";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleExponential(&out, param.lambda);  // NOLINT(*)
  });
}

template<typename xpu>
void SamplePoisson_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SamplePoissonParam& param = nnvm::get<SamplePoissonParam>(attrs.parsed);
  CHECK_GE(param.lambda, 0) << "lambda parameter in poisson distribution has to be non-negative";
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SamplePoisson(&out, param.lambda);  // NOLINT(*)
  });
}

template<typename xpu>
void SampleNegBinomial_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleNegBinomialParam& param = nnvm::get<SampleNegBinomialParam>(attrs.parsed);
  CHECK_GE(param.k, 0) << "k parameter in negative binomial distribution has to be non-negative";
  CHECK_GE(param.p, 0) << "p parameter in negative binomial distribution has to be non-negative";
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleNegativeBinomial(&out, param.k, param.p);  // NOLINT(*)
  });
}

template<typename xpu>
void SampleGenNegBinomial_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleGenNegBinomialParam& param = nnvm::get<SampleGenNegBinomialParam>(attrs.parsed);
  CHECK_GE(param.mu, 0)
    << "mu parameter in generalized negative binomial distribution has to be non-negative";
  CHECK_GE(param.alpha, 0)
    << "alpha parameter in generalized negative binomial distribution has to be non-negative";
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleGeneralizedNegativeBinomial(&out, param.mu, param.alpha);  // NOLINT(*)
  });
}

#if MSHADOW_USE_CUDA
// GPU versions of uniform and normal distribution.
template<>
void SampleUniform_<gpu>(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  typedef gpu xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleUniformParam& param = nnvm::get<SampleUniformParam>(attrs.parsed);
  mshadow::Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  if (outputs[0].type_flag_ != mshadow::kFloat32) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      // Not float32: use workspace and copy to output
      mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, float> workspace =
        ctx.requested[ResourceRequest::kTempSpace].get_space_typed<xpu, 1, float>
        (mshadow::Shape1(out.shape_.Size()), s);
      prnd->SampleUniform(&workspace, param.low, param.high);
      out = reshape(tcast<DType>(workspace), mshadow::Shape2(out.shape_[0], out.shape_[1]));
    });
  } else {
    // float32: write directly into output
    mshadow::Tensor<xpu, 2, float> out = outputs[0].FlatTo2D<xpu, float>(s);
    prnd->SampleUniform(&out, param.low, param.high);
  }
}

template<>
void SampleNormal_<gpu>(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  typedef gpu xpu;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const SampleNormalParam& param = nnvm::get<SampleNormalParam>(attrs.parsed);
  mshadow::Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  if (outputs[0].type_flag_ != mshadow::kFloat32) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      // Not float32: use workspace and copy to output
      mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, float> workspace =
        ctx.requested[ResourceRequest::kTempSpace].get_space_typed<xpu, 1, float>
        (mshadow::Shape1(out.shape_.Size()), s);
      prnd->SampleGaussian(&workspace, param.loc, param.scale);
      out = reshape(tcast<DType>(workspace), mshadow::Shape2(out.shape_[0], out.shape_[1]));
    });
  } else {
    // float32: write directly into output
    mshadow::Tensor<xpu, 2, float> out = outputs[0].FlatTo2D<xpu, float>(s);
    prnd->SampleGaussian(&out, param.loc, param.scale);
  }
}
#endif

inline std::vector<ResourceRequest> SampleResource(const NodeAttrs& attrs) {
  return { ResourceRequest::kRandom };
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_SAMPLE_OP_H_
