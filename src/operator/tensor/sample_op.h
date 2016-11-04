/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.h
 * \brief Function defintion sampling operators.
 */
#ifndef MXNET_OPERATOR_TENSOR_SAMPLE_OP_H_
#define MXNET_OPERATOR_TENSOR_SAMPLE_OP_H_

#include <mxnet/operator_util.h>
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
  }
};

struct SampleNormalParam : public dmlc::Parameter<SampleNormalParam> {
  float loc;
  float scale;
  TShape shape;
  std::string ctx;
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
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
      << "only support float32 rnd so far";
  const SampleUniformParam& param = nnvm::get<SampleUniformParam>(attrs.parsed);
  mshadow::Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  mshadow::Tensor<xpu, 2, float> out = outputs[0].FlatTo2D<xpu, float>(s);
  prnd->SampleUniform(&out, param.low, param.high);
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
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
      << "only support float32 rnd so far";
  const SampleNormalParam& param = nnvm::get<SampleNormalParam>(attrs.parsed);
  mshadow::Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  mshadow::Tensor<xpu, 2, float> out = outputs[0].FlatTo2D<xpu, float>(s);
  prnd->SampleGaussian(&out, param.loc, param.scale);  // NOLINT(*)
}

inline std::vector<ResourceRequest> SampleResource(const NodeAttrs& attrs) {
  return { ResourceRequest::kRandom };
}

#define MXNET_OPERATOR_REGISTER_SAMPLE(name, ParamType)                 \
  NNVM_REGISTER_OP(name)                                                \
  .set_num_inputs(0)                                                    \
  .set_num_outputs(1)                                                   \
  .set_attr_parser(ParamParser<ParamType>)                              \
  .set_attr<nnvm::FInferShape>("FInferShape", InitShape<ParamType>)     \
  .set_attr<nnvm::FInferType>("FInferType", InitType)                   \
  .set_attr<FResourceRequest>("FResourceRequest", SampleResource)       \
  .add_arguments(ParamType::__FIELDS__())
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_SAMPLE_OP_H_
