/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.h
 * \brief Elementary sampling operators
 */
#ifndef MXNET_OPERATOR_RANDOM_SAMPLE_OP_H_
#define MXNET_OPERATOR_RANDOM_SAMPLE_OP_H_

#include <mxnet/operator_util.h>
#include <mshadow/base.h>
#include <string>
#include <vector>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

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
    .describe("Lower bound of the distribution.");
    DMLC_DECLARE_FIELD(high).set_default(1.0f)
    .describe("Upper bound of the distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
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
    .describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
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
    .describe("Alpha parameter (shape) of the gamma distribution.");
    DMLC_DECLARE_FIELD(beta).set_default(1.0f)
    .describe("Beta parameter (scale) of the gamma distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleExponentialParam : public dmlc::Parameter<SampleExponentialParam> {
  float lam;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SampleExponentialParam) {
    DMLC_DECLARE_FIELD(lam).set_default(1.0f)
    .describe("Lambda parameter (rate) of the exponential distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SamplePoissonParam : public dmlc::Parameter<SamplePoissonParam> {
  float lam;
  TShape shape;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(SamplePoissonParam) {
    DMLC_DECLARE_FIELD(lam).set_default(1.0f)
    .describe("Lambda parameter (rate) of the Poisson distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
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
    .describe("Limit of unsuccessful experiments.");
    DMLC_DECLARE_FIELD(p).set_default(1.0f)
    .describe("Failure probability in each experiment.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
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
    .describe("Mean of the negative binomial distribution.");
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f)
    .describe("Alpha (dispersion) parameter of the negative binomial distribution.");
    DMLC_DECLARE_FIELD(shape)
    .set_default(TShape())
    .describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx)
    .set_default("")
    .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
              " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
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
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
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
  CHECK_GT(param.lam, 0) << "lambda parameter in exponential distribution has to be positive";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleExponential(&out, param.lam);  // NOLINT(*)
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
  CHECK_GE(param.lam, 0) << "lambda parameter in poisson distribution has to be non-negative";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SamplePoisson(&out, param.lam);  // NOLINT(*)
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
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
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
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Random<xpu, DType> *prnd = ctx.requested[0].get_random<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    prnd->SampleGeneralizedNegativeBinomial(&out, param.mu, param.alpha);  // NOLINT(*)
  });
}

template<typename ParamType>
inline bool SampleOpType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_type,
                         std::vector<int> *out_type) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_type->size(), 0);
  CHECK_EQ(out_type->size(), 1);
  int dtype = -1;
  int dtype_out = (*out_type)[0];
  if (dtype_out != -1) {
    // Output type can be inferred, use it and make sure it
    dtype = dtype_out;
    if (param.dtype != -1) {
      // dtype given in args, check that it matches the output type
      CHECK_EQ(dtype_out, param.dtype) << "Output type does not match requested type: "
      << dtype_out << " vs " << param.dtype;
    }
  } else {
    // Output type can't be inferred
    if (param.dtype != -1) {
      // Use dtype given in args
      dtype = param.dtype;
    } else {
      // Use default
      dtype = mshadow::kFloat32;
    }
  }
  bool dtype_ok = (dtype == mshadow::kFloat16) || (dtype == mshadow::kFloat32) ||
  (dtype == mshadow::kFloat64);
  CHECK_EQ(dtype_ok, true) << "Output type must be float16, float32, or float64: dtype is "
  << dtype_out << " vs " << mshadow::kFloat16 << " or " << mshadow::kFloat32 << " or "
  << mshadow::kFloat64;
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return true;
}

inline std::vector<ResourceRequest> SampleResource(const NodeAttrs& attrs) {
  return { ResourceRequest::kRandom, ResourceRequest::kTempSpace };
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RANDOM_SAMPLE_OP_H_
