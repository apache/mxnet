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
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output. If output given, set to type of output."
              "If output not given and type not defined (dtype=None), set to float32.");
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
    .add_enum("None", -1)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .set_default(-1)
    .describe("DType of the output. If output given, set to type of output."
              "If output not given and type not defined (dtype=None), set to float32.");
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

#define MXNET_OPERATOR_REGISTER_SAMPLE(name, ParamType)                 \
  NNVM_REGISTER_OP(name)                                                \
  .set_num_inputs(0)                                                    \
  .set_num_outputs(1)                                                   \
  .set_attr_parser(ParamParser<ParamType>)                              \
  .set_attr<nnvm::FInferShape>("FInferShape", InitShape<ParamType>)     \
  .set_attr<nnvm::FInferType>("FInferType", SampleOpType<ParamType>)    \
  .set_attr<FResourceRequest>("FResourceRequest", SampleResource)       \
  .add_arguments(ParamType::__FIELDS__())
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_SAMPLE_OP_H_
