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
 * \file sample_op.h
 * \brief Elementary sampling operators
 */
#ifndef MXNET_OPERATOR_RANDOM_SAMPLE_OP_H_
#define MXNET_OPERATOR_RANDOM_SAMPLE_OP_H_

#include <mxnet/operator_util.h>
#include <mshadow/base.h>
#include <string>
#include <vector>
#include "../../common/utils.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"
#include "./sampler.h"

namespace mxnet {
namespace op {

struct SampleOpParam {
  mxnet::TShape shape;
  std::string ctx;
  int dtype;
};

struct UniformParam {
  float low;
  float high;
};

struct NormalParam {
  float loc;
  float scale;
};

struct GammaParam {
  float alpha;
  float beta;
};

struct ExponentialParam {
  float lam;
};

struct PoissonParam {
  float lam;
};

struct BinomialParam {
  int n;
  float p;
};

struct NegBinomialParam {
  int k;
  float p;
};

struct GenNegBinomialParam {
  float mu;
  float alpha;
};

struct RandIntParam {
  int64_t low;
  int64_t high;
};

struct SampleUniformParam : public dmlc::Parameter<SampleUniformParam>,
                            UniformParam,
                            SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleUniformParam) {
    DMLC_DECLARE_FIELD(low).set_default(0.0f).describe("Lower bound of the distribution.");
    DMLC_DECLARE_FIELD(high).set_default(1.0f).describe("Upper bound of the distribution.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleNormalParam : public dmlc::Parameter<SampleNormalParam>, NormalParam, SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleNormalParam) {
    DMLC_DECLARE_FIELD(loc).set_default(0.0f).describe("Mean of the distribution.");
    DMLC_DECLARE_FIELD(scale).set_default(1.0f).describe("Standard deviation of the distribution.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleGammaParam : public dmlc::Parameter<SampleGammaParam>, GammaParam, SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleGammaParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f).describe(
        "Alpha parameter (shape) of the gamma distribution.");
    DMLC_DECLARE_FIELD(beta).set_default(1.0f).describe(
        "Beta parameter (scale) of the gamma distribution.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleExponentialParam : public dmlc::Parameter<SampleExponentialParam>,
                                ExponentialParam,
                                SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleExponentialParam) {
    DMLC_DECLARE_FIELD(lam).set_default(1.0f).describe(
        "Lambda parameter (rate) of the exponential distribution.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SamplePoissonParam : public dmlc::Parameter<SamplePoissonParam>,
                            PoissonParam,
                            SampleOpParam {
  DMLC_DECLARE_PARAMETER(SamplePoissonParam) {
    DMLC_DECLARE_FIELD(lam).set_default(1.0f).describe(
        "Lambda parameter (rate) of the Poisson distribution.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleBinomialParam : public dmlc::Parameter<SampleBinomialParam>,
                             BinomialParam,
                             SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleBinomialParam) {
    DMLC_DECLARE_FIELD(n).set_default(1).describe("number of experiments.");
    DMLC_DECLARE_FIELD(p).set_default(1.0f).describe("success probability in each experiment.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleNegBinomialParam : public dmlc::Parameter<SampleNegBinomialParam>,
                                NegBinomialParam,
                                SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleNegBinomialParam) {
    DMLC_DECLARE_FIELD(k).set_default(1).describe("Limit of unsuccessful experiments.");
    DMLC_DECLARE_FIELD(p).set_default(1.0f).describe("Failure probability in each experiment.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleGenNegBinomialParam : public dmlc::Parameter<SampleGenNegBinomialParam>,
                                   GenNegBinomialParam,
                                   SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleGenNegBinomialParam) {
    DMLC_DECLARE_FIELD(mu).set_default(1.0f).describe(
        "Mean of the negative binomial distribution.");
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f).describe(
        "Alpha (dispersion) parameter of the negative binomial distribution.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bfloat16", mshadow::kBfloat16)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

struct SampleRandIntParam : public dmlc::Parameter<SampleRandIntParam>,
                            RandIntParam,
                            SampleOpParam {
  DMLC_DECLARE_PARAMETER(SampleRandIntParam) {
    DMLC_DECLARE_FIELD(low).describe("Lower bound of the distribution.");
    DMLC_DECLARE_FIELD(high).describe("Upper bound of the distribution.");
    DMLC_DECLARE_FIELD(shape).set_default(mxnet::TShape()).describe("Shape of the output.");
    DMLC_DECLARE_FIELD(ctx).set_default("").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("None", -1)
        .add_enum("int32", mshadow::kInt32)
        .add_enum("int64", mshadow::kInt64)
        .set_default(-1)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to int32 if not defined (dtype=None).");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream low_s, high_s, dtype_s, shape_s;
    low_s << low;
    high_s << high;
    dtype_s << dtype;
    shape_s << shape;
    (*dict)["low"]   = low_s.str();
    (*dict)["high"]  = high_s.str();
    (*dict)["dtype"] = MXNetTypeWithBool2String(dtype);
    (*dict)["shape"] = shape_s.str();
  }
};

struct SampleUniformLikeParam : public dmlc::Parameter<SampleUniformLikeParam>, UniformParam {
  DMLC_DECLARE_PARAMETER(SampleUniformLikeParam) {
    DMLC_DECLARE_FIELD(low).set_default(0.0f).describe("Lower bound of the distribution.");
    DMLC_DECLARE_FIELD(high).set_default(1.0f).describe("Upper bound of the distribution.");
  }
};

struct SampleNormalLikeParam : public dmlc::Parameter<SampleNormalLikeParam>, NormalParam {
  DMLC_DECLARE_PARAMETER(SampleNormalLikeParam) {
    DMLC_DECLARE_FIELD(loc).set_default(0.0f).describe("Mean of the distribution.");
    DMLC_DECLARE_FIELD(scale).set_default(1.0f).describe("Standard deviation of the distribution.");
  }
};

struct SampleGammaLikeParam : public dmlc::Parameter<SampleGammaLikeParam>, GammaParam {
  DMLC_DECLARE_PARAMETER(SampleGammaLikeParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f).describe(
        "Alpha parameter (shape) of the gamma distribution.");
    DMLC_DECLARE_FIELD(beta).set_default(1.0f).describe(
        "Beta parameter (scale) of the gamma distribution.");
  }
};

struct SampleExponentialLikeParam : public dmlc::Parameter<SampleExponentialLikeParam>,
                                    ExponentialParam {
  DMLC_DECLARE_PARAMETER(SampleExponentialLikeParam) {
    DMLC_DECLARE_FIELD(lam).set_default(1.0f).describe(
        "Lambda parameter (rate) of the exponential distribution.");
  }
};

struct SamplePoissonLikeParam : public dmlc::Parameter<SamplePoissonLikeParam>, PoissonParam {
  DMLC_DECLARE_PARAMETER(SamplePoissonLikeParam) {
    DMLC_DECLARE_FIELD(lam).set_default(1.0f).describe(
        "Lambda parameter (rate) of the Poisson distribution.");
  }
};

struct SampleBinomialLikeParam : public dmlc::Parameter<SampleBinomialLikeParam>, BinomialParam {
  DMLC_DECLARE_PARAMETER(SampleBinomialLikeParam) {
    DMLC_DECLARE_FIELD(n).set_default(1).describe("Number of experiments.");
    DMLC_DECLARE_FIELD(p).set_default(1.0f).describe("success probability in each experiment.");
  }
};

struct SampleNegBinomialLikeParam : public dmlc::Parameter<SampleNegBinomialLikeParam>,
                                    NegBinomialParam {
  DMLC_DECLARE_PARAMETER(SampleNegBinomialLikeParam) {
    DMLC_DECLARE_FIELD(k).set_default(1).describe("Limit of unsuccessful experiments.");
    DMLC_DECLARE_FIELD(p).set_default(1.0f).describe("Failure probability in each experiment.");
  }
};

struct SampleGenNegBinomialLikeParam : public dmlc::Parameter<SampleGenNegBinomialLikeParam>,
                                       GenNegBinomialParam {
  DMLC_DECLARE_PARAMETER(SampleGenNegBinomialLikeParam) {
    DMLC_DECLARE_FIELD(mu).set_default(1.0f).describe(
        "Mean of the negative binomial distribution.");
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f).describe(
        "Alpha (dispersion) parameter of the negative binomial distribution.");
  }
};

using FSampleCompute = std::function<
    void(const nnvm::NodeAttrs& attrs, const OpContext& ctx, const OpReqType& req, TBlob* outputs)>;

using mxnet::TBlob;
using namespace mxnet::common::random;

// Allocates a single chunk of workspace memory and partitions it into three
// workspace tensors that hold the seeds as well as the distribution parameters.
template <typename xpu, typename DType>
MSHADOW_FORCE_INLINE void GetSamplingTempData(DType p1,
                                              DType p2,
                                              const OpContext& ctx,
                                              Tensor<xpu, 1, DType>* parm1,
                                              Tensor<xpu, 1, DType>* parm2) {
  Stream<xpu>* s = ctx.get_stream<xpu>();
  // Combined memory requirement for the workspace data.
  const index_t nInt((2 * sizeof(DType) + sizeof(unsigned) - 1) / sizeof(unsigned));
  Tensor<xpu, 1, unsigned> wspace =
      ctx.requested[1].get_space_typed<xpu, 1, unsigned>(Shape1(nInt), s);
  // Partition workspace into two chunks and initialize them.
  DType* pspace = static_cast<DType*>(static_cast<void*>(wspace.dptr_));
  *parm1        = Tensor<xpu, 1, DType>(pspace, Shape1(1), s);
  Copy(*parm1, Tensor<cpu, 1, DType>(&p1, Shape1(1)), s);
  *parm2 = Tensor<xpu, 1, DType>(pspace + 1, Shape1(1), s);
  Copy(*parm2, Tensor<cpu, 1, DType>(&p2, Shape1(1)), s);
}

template <typename xpu, typename ParamType>
static inline void uniform_op(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const OpReqType& req,
                              TBlob* outputs) {
  Stream<xpu>* s            = ctx.get_stream<xpu>();
  const UniformParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GE(param.high, param.low) << "low must be less or equal to high in uniform distribution";
  Tensor<xpu, 1, float> low, high;
  GetSamplingTempData<xpu, float>(param.low, param.high, ctx, &low, &high);
  UniformSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(low, high, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void normal_op(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const OpReqType& req,
                             TBlob* outputs) {
  Stream<xpu>* s           = ctx.get_stream<xpu>();
  const NormalParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GT(param.scale, 0) << "scale parameter in gaussian has to be positive";
  Tensor<xpu, 1, float> loc, scale;
  GetSamplingTempData<xpu, float>(param.loc, param.scale, ctx, &loc, &scale);
  NormalSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(loc, scale, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void gamma_op(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const OpReqType& req,
                            TBlob* outputs) {
  Stream<xpu>* s          = ctx.get_stream<xpu>();
  const GammaParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GT(param.alpha, 0) << "alpha parameter in gamma distribution has to be positive";
  CHECK_GT(param.beta, 0) << "beta parameter in gamma distribution has to be positive";
  Tensor<xpu, 1, float> alpha, beta;
  GetSamplingTempData<xpu, float>(param.alpha, param.beta, ctx, &alpha, &beta);
  GammaSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(alpha, beta, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void exponential_op(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const OpReqType& req,
                                  TBlob* outputs) {
  Stream<xpu>* s                = ctx.get_stream<xpu>();
  const ExponentialParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GT(param.lam, 0) << "lambda parameter in exponential distribution has to be positive";
  Tensor<xpu, 1, float> lam, dummy;
  GetSamplingTempData<xpu, float>(param.lam, 0, ctx, &lam, &dummy);
  ExponentialSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(lam, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void poisson_op(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const OpReqType& req,
                              TBlob* outputs) {
  Stream<xpu>* s            = ctx.get_stream<xpu>();
  const PoissonParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GE(param.lam, 0) << "lambda parameter in poisson distribution has to be non-negative";
  Tensor<xpu, 1, float> lam, dummy;
  GetSamplingTempData<xpu, float>(param.lam, 0, ctx, &lam, &dummy);
  PoissonSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(lam, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void binomial_op(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const OpReqType& req,
                               TBlob* outputs) {
  Stream<xpu>* s             = ctx.get_stream<xpu>();
  const BinomialParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GE(param.n, 0) << "n parameter in binomial distribution has to be non-negative";
  CHECK_GE(param.p, 0) << "p parameter in binomial distribution has to be non-negative";
  Tensor<xpu, 1, float> n, p;
  GetSamplingTempData<xpu, float>(param.n, param.p, ctx, &n, &p);
  BinomialSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(n, p, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void neg_binomial_op(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const OpReqType& req,
                                   TBlob* outputs) {
  Stream<xpu>* s                = ctx.get_stream<xpu>();
  const NegBinomialParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GE(param.k, 0) << "k parameter in negative binomial distribution has to be non-negative";
  CHECK_GE(param.p, 0) << "p parameter in negative binomial distribution has to be non-negative";
  Tensor<xpu, 1, float> k, p;
  GetSamplingTempData<xpu, float>(param.k, param.p, ctx, &k, &p);
  NegativeBinomialSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(k, p, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void gen_neg_binomial_op(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const OpReqType& req,
                                       TBlob* outputs) {
  Stream<xpu>* s                   = ctx.get_stream<xpu>();
  const GenNegBinomialParam& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_GE(param.mu, 0)
      << "mu parameter in generalized negative binomial distribution has to be non-negative";
  CHECK_GE(param.alpha, 0)
      << "alpha parameter in generalized negative binomial distribution has to be non-negative";
  Tensor<xpu, 1, float> mu, alpha;
  GetSamplingTempData<xpu, float>(param.mu, param.alpha, ctx, &mu, &alpha);
  GeneralizedNegativeBinomialSampler<xpu> sampler;
  MSHADOW_REAL_TYPE_SWITCH_EX(outputs[0].type_flag_, OType, _, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(mu, alpha, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
static inline void rand_int_op(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const OpReqType& req,
                               TBlob* outputs) {
  Stream<xpu>* s                  = ctx.get_stream<xpu>();
  const SampleRandIntParam& param = nnvm::get<SampleRandIntParam>(attrs.parsed);
  CHECK_GE(param.high, param.low) << "low must be less or equal to high in uniform distribution";
  Tensor<xpu, 1, int64_t> low, high;
  GetSamplingTempData<xpu, int64_t>(param.low, param.high, ctx, &low, &high);
  RandIntSampler<xpu> sampler;
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    RandGenerator<xpu, OType>* pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
    Tensor<xpu, 1, OType> out       = outputs->FlatTo1D<xpu, OType>(s);
    sampler.Sample(low, high, out, pgen, s);
  });
}

template <typename xpu, typename ParamType>
struct SampleMaster;

template <typename xpu>
struct SampleMaster<xpu, SampleUniformParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    uniform_op<xpu, SampleUniformParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleUniformLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    uniform_op<xpu, SampleUniformLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleNormalParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    normal_op<xpu, SampleNormalParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleNormalLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    normal_op<xpu, SampleNormalLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleGammaParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    gamma_op<xpu, SampleGammaParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleGammaLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    gamma_op<xpu, SampleGammaLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleExponentialParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    exponential_op<xpu, SampleExponentialParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleExponentialLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    exponential_op<xpu, SampleExponentialLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SamplePoissonParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    poisson_op<xpu, SamplePoissonParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SamplePoissonLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    poisson_op<xpu, SamplePoissonLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleBinomialParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    binomial_op<xpu, SampleBinomialParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleBinomialLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    binomial_op<xpu, SampleBinomialLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleNegBinomialParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    neg_binomial_op<xpu, SampleNegBinomialParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleNegBinomialLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    neg_binomial_op<xpu, SampleNegBinomialLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleGenNegBinomialParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    gen_neg_binomial_op<xpu, SampleGenNegBinomialParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleGenNegBinomialLikeParam> {
  static inline void op(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const OpReqType& req,
                        TBlob* outputs) {
    gen_neg_binomial_op<xpu, SampleGenNegBinomialLikeParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu>
struct SampleMaster<xpu, SampleRandIntParam> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    rand_int_op<xpu, SampleRandIntParam>(attrs, ctx, req, outputs);
  }
};

template <typename xpu, typename ParamType>
void SampleComputeEx_(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs,
                      SampleMaster<xpu, ParamType> sample_master) {
  using namespace mxnet::op;
  NDArray output          = outputs[0];
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  if (output.storage_type() == kRowSparseStorage) {
    // indices
    nnvm::dim_t nnr = output.shape()[0];
    output.CheckAndAlloc({mshadow::Shape1(nnr)});
    MSHADOW_IDX_TYPE_SWITCH(output.aux_type(rowsparse::kIdx), IType, {
      IType* idx = output.aux_data(rowsparse::kIdx).dptr<IType>();
      mxnet_op::Kernel<PopulateFullIdxRspKernel, xpu>::Launch(s, nnr, idx);
    });
    // data
    TBlob out_blob = output.data();
    sample_master.op(attrs, ctx, req[0], &out_blob);
  } else {
    LOG(FATAL) << "Unexpected storage type for SampleComputeEx_: " << output.storage_type();
  }
}

template <typename xpu, typename ParamType>
void Sample_(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob>& outputs) {
  TBlob out = outputs[0];
  SampleMaster<xpu, ParamType>::op(attrs, ctx, req[0], &out);
}

template <typename xpu, typename ParamType>
void SampleEx_(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs) {
  SampleMaster<xpu, ParamType> sample_master;
  SampleComputeEx_<xpu, ParamType>(attrs, ctx, inputs, req, outputs, sample_master);
}

template <typename ParamType>
inline bool SampleOpType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_type,
                         std::vector<int>* out_type) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_type->size(), 0);
  CHECK_EQ(out_type->size(), 1);
  int dtype     = -1;
  int dtype_out = (*out_type)[0];
  if (dtype_out != -1) {
    // Output type can be inferred, use it and make sure it matches
    dtype = dtype_out;
    if (param.dtype != -1) {
      // dtype given in args, check that it matches the output type
      CHECK_EQ(dtype_out, param.dtype)
          << "Output type does not match requested type: " << dtype_out << " vs " << param.dtype;
    }
  } else {
    // Output type can't be inferred
    if (param.dtype != -1) {
      // Use dtype given in args
      dtype = param.dtype;
    } else {
      // Use default
      dtype = mxnet::common::GetDefaultDtype();
    }
  }
  bool dtype_ok = dtype == mshadow::kBfloat16 || dtype == mshadow::kFloat16 ||
                  dtype == mshadow::kFloat32 || dtype == mshadow::kFloat64;
  CHECK(dtype_ok) << "Output type must be bfloat16, float16, float32, float64: dtype is "
                  << dtype_out << " vs " << mshadow::kBfloat16 << " or " << mshadow::kFloat16
                  << " or " << mshadow::kFloat32 << " or " << mshadow::kFloat64;
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return true;
}

template <>
inline bool SampleOpType<SampleRandIntParam>(const nnvm::NodeAttrs& attrs,
                                             std::vector<int>* in_type,
                                             std::vector<int>* out_type) {
  const SampleRandIntParam& param = nnvm::get<SampleRandIntParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), 0);
  CHECK_EQ(out_type->size(), 1);
  int dtype     = -1;
  int dtype_out = (*out_type)[0];
  if (dtype_out != -1) {
    // Output type can be inferred, use it and make sure it matches
    dtype = dtype_out;
    if (param.dtype != -1) {
      // dtype given in args, check that it matches the output type
      CHECK_EQ(dtype_out, param.dtype)
          << "Output type does not match requested type: " << dtype_out << " vs " << param.dtype;
    }
  } else {
    // Output type can't be inferred
    if (param.dtype != -1) {
      // Use dtype given in args
      dtype = param.dtype;
    } else {
      // Use default
      dtype = mshadow::kInt32;
    }
  }
  bool dtype_ok = (dtype == mshadow::kInt32) || (dtype == mshadow::kInt64);
  CHECK(dtype_ok) << "Output type must be int32, int64: dtype is " << dtype_out << " vs "
                  << mshadow::kInt32 << " or " << mshadow::kInt64;
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return true;
}

inline std::vector<ResourceRequest> SampleResource(const NodeAttrs& attrs) {
  return {ResourceRequest::kParallelRandom, ResourceRequest::kTempSpace};
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RANDOM_SAMPLE_OP_H_
