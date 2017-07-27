/*!
 *  Copyright (c) 2016 by Contributors
 * \file sample_op.cu
 * \brief GPU Implementation of sample op
 */
#include "./sample_op.h"

namespace mxnet {
namespace op {

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
        ctx.requested[1].get_space_typed<xpu, 1, float>
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
        ctx.requested[1].get_space_typed<xpu, 1, float>
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

NNVM_REGISTER_OP(random_uniform)
.set_attr<FCompute>("FCompute<gpu>", SampleUniform_<gpu>);

NNVM_REGISTER_OP(random_normal)
.set_attr<FCompute>("FCompute<gpu>", SampleNormal_<gpu>);

}  // namespace op
}  // namespace mxnet
