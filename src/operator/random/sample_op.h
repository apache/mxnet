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
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"
#include "./sampler.h"

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
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
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
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
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
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
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
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
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
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
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
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
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
    .add_enum("float32", kFloat32)
    .add_enum("float64", kFloat64)
    .add_enum("float16", kFloat16)
    .set_default(-1)
    .describe("DType of the output in case this can't be inferred. "
              "Defaults to float32 if not defined (dtype=None).");
  }
};

using FSampleCompute = std::function<void (const nnvm::NodeAttrs& attrs,
                                           const OpContext& ctx,
                                           const OpReqType& req,
                                           TBlob* outputs)>;

using mxnet::TBlob;
using namespace mxnet::common::random;

// Allocates a single chunk of workspace memory and partitions it into three
// workspace tensors that hold the seeds as well as the distribution parameters.
template<typename xpu, typename DType>
MSHADOW_FORCE_INLINE void GetSamplingTempData(DType p1, DType p2, const OpContext& ctx,
                                              Tensor<xpu, 1, DType>* parm1,
                                              Tensor<xpu, 1, DType>* parm2) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Combined memory requirement for the workspace data.
  const index_t nInt((2 * sizeof(DType) + sizeof(unsigned) - 1) / sizeof(unsigned));
  Tensor<xpu, 1, unsigned> wspace
    = ctx.requested[1].get_space_typed<xpu, 1, unsigned>(Shape1(nInt), s);
  // Partition workspace into two chunks and initialize them.
  DType *pspace = static_cast<DType*>(static_cast<void*>(wspace.dptr_));
  *parm1 = Tensor<xpu, 1, DType>(pspace, Shape1(1), s);
  Copy(*parm1, Tensor<cpu, 1, DType>(&p1, Shape1(1)), s);
  *parm2 = Tensor<xpu, 1, DType>(pspace+1, Shape1(1), s);
  Copy(*parm2, Tensor<cpu, 1, DType>(&p2, Shape1(1)), s);
}

template<typename xpu, typename Sampler>
struct SampleMaster;

template<typename xpu>
struct SampleMaster<xpu, UniformSampler<xpu>> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const SampleUniformParam& param = nnvm::get<SampleUniformParam>(attrs.parsed);
    CHECK_GE(param.high, param.low) << "low must be less or equal to high in uniform distribution";
    Tensor<xpu, 1, float> low, high;
    GetSamplingTempData<xpu, float>(param.low, param.high, ctx,
                                    &low, &high);
    UniformSampler<xpu> sampler;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      Tensor<xpu, 1, OType> out = outputs->FlatTo1D<xpu, OType>(s);
      sampler.Sample(low, high, out, pgen, s);
    });
  }
};

template<typename xpu>
struct SampleMaster<xpu, NormalSampler<xpu>> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const SampleNormalParam& param = nnvm::get<SampleNormalParam>(attrs.parsed);
    CHECK_GT(param.scale, 0) << "scale parameter in gaussian has to be positive";
    Tensor<xpu, 1, float> loc, scale;
    GetSamplingTempData<xpu, float>(param.loc, param.scale, ctx, &loc, &scale);
    NormalSampler<xpu> sampler;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      Tensor<xpu, 1, OType> out = outputs->FlatTo1D<xpu, OType>(s);
      sampler.Sample(loc, scale, out, pgen, s);
    });
  }
};

template<typename xpu>
struct SampleMaster<xpu, GammaSampler<xpu>> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const SampleGammaParam& param = nnvm::get<SampleGammaParam>(attrs.parsed);
    CHECK_GT(param.alpha, 0) << "alpha parameter in gamma distribution has to be positive";
    CHECK_GT(param.beta, 0) << "beta parameter in gamma distribution has to be positive";
    Tensor<xpu, 1, float> alpha, beta;
    GetSamplingTempData<xpu, float>(param.alpha, param.beta, ctx, &alpha, &beta);
    GammaSampler<xpu> sampler;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      Tensor<xpu, 1, OType> out = outputs->FlatTo1D<xpu, OType>(s);
      sampler.Sample(alpha, beta, out, pgen, s);
    });
  }
};

template<typename xpu>
struct SampleMaster<xpu, ExponentialSampler<xpu>> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const SampleExponentialParam& param = nnvm::get<SampleExponentialParam>(attrs.parsed);
    CHECK_GT(param.lam, 0) << "lambda parameter in exponential distribution has to be positive";
    Tensor<xpu, 1, float> lam, dummy;
    GetSamplingTempData<xpu, float>(param.lam, 0, ctx, &lam, &dummy);
    ExponentialSampler<xpu> sampler;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      Tensor<xpu, 1, OType> out = outputs->FlatTo1D<xpu, OType>(s);
      sampler.Sample(lam, out, pgen, s);
    });
  }
};

template<typename xpu>
struct SampleMaster<xpu, PoissonSampler<xpu>> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const SamplePoissonParam& param = nnvm::get<SamplePoissonParam>(attrs.parsed);
    CHECK_GE(param.lam, 0) << "lambda parameter in poisson distribution has to be non-negative";
    Tensor<xpu, 1, float> lam, dummy;
    GetSamplingTempData<xpu, float>(param.lam, 0, ctx, &lam, &dummy);
    PoissonSampler<xpu> sampler;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      Tensor<xpu, 1, OType> out = outputs->FlatTo1D<xpu, OType>(s);
      sampler.Sample(lam, out, pgen, s);
    });
  }
};

template<typename xpu>
struct SampleMaster<xpu, NegativeBinomialSampler<xpu>> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const SampleNegBinomialParam& param = nnvm::get<SampleNegBinomialParam>(attrs.parsed);
    CHECK_GE(param.k, 0) << "k parameter in negative binomial distribution has to be non-negative";
    CHECK_GE(param.p, 0) << "p parameter in negative binomial distribution has to be non-negative";
    Tensor<xpu, 1, float> k, p;
    GetSamplingTempData<xpu, float>(param.k, param.p, ctx, &k, &p);
    NegativeBinomialSampler<xpu> sampler;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      Tensor<xpu, 1, OType> out = outputs->FlatTo1D<xpu, OType>(s);
      sampler.Sample(k, p, out, pgen, s);
    });
  }
};

template<typename xpu>
struct SampleMaster<xpu, GeneralizedNegativeBinomialSampler<xpu>> {
  static void op(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const OpReqType& req,
                 TBlob* outputs) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const SampleGenNegBinomialParam& param = nnvm::get<SampleGenNegBinomialParam>(attrs.parsed);
    CHECK_GE(param.mu, 0)
      << "mu parameter in generalized negative binomial distribution has to be non-negative";
    CHECK_GE(param.alpha, 0)
      << "alpha parameter in generalized negative binomial distribution has to be non-negative";
    Tensor<xpu, 1, float> mu, alpha;
    GetSamplingTempData<xpu, float>(param.mu, param.alpha, ctx, &mu, &alpha);
    GeneralizedNegativeBinomialSampler<xpu> sampler;
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      RandGenerator<xpu, OType> *pgen = ctx.requested[0].get_parallel_random<xpu, OType>();
      Tensor<xpu, 1, OType> out = outputs->FlatTo1D<xpu, OType>(s);
      sampler.Sample(mu, alpha, out, pgen, s);
    });
  }
};

template<typename xpu, typename Sampler>
void SampleComputeEx_(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs,
                      SampleMaster<xpu, Sampler> sample_master) {
  using namespace mxnet::op;
  NDArray output = outputs[0];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
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
    LOG(FATAL) << "Unexpected storage type for SampleComputeEx_: "
               << output.storage_type();
  }
}

template<typename xpu, typename Sampler>
void Sample_(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob>& outputs) {
  TBlob out = outputs[0];
  SampleMaster<xpu, Sampler>::op(attrs, ctx, req[0], &out);
}

template<typename xpu, typename Sampler>
void SampleEx_(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs) {
  SampleMaster<xpu, Sampler> sample_master;
  SampleComputeEx_<xpu, Sampler>(attrs, ctx, inputs, req, outputs, sample_master);
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
      dtype = kFloat32;
    }
  }
  bool dtype_ok = (dtype == kFloat16) || (dtype == kFloat32) ||
  (dtype == kFloat64);
  CHECK_EQ(dtype_ok, true) << "Output type must be float16, float32, or float64: dtype is "
  << dtype_out << " vs " << kFloat16 << " or " << kFloat32 << " or "
  << kFloat64;
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  return true;
}

inline std::vector<ResourceRequest> SampleResource(const NodeAttrs& attrs) {
  return { ResourceRequest::kParallelRandom, ResourceRequest::kTempSpace };
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RANDOM_SAMPLE_OP_H_
