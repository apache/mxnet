/*!
 *  Copyright (c) 2016 by Contributors
 * \file optimizer_op-inl.h
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#ifndef MXNET_OPERATOR_OPTIMIZER_OP_INL_H_
#define MXNET_OPERATOR_OPTIMIZER_OP_INL_H_
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./elemwise_op_common.h"

namespace mxnet {
namespace op {
struct SGDParam : public dmlc::Parameter<SGDParam> {
  float lr;
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(SGDParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("learning_rate");
    DMLC_DECLARE_FIELD(momentum)
    .set_default(0.0f)
    .describe("momentum");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("weight decay");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("rescale gradient as grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("If greater than 0, clip gradient to "
              "grad = max(min(grad, -clip_gradient), clip_gradient). "
              "Otherwise turned off.");
  }
};

template<typename xpu>
inline void SGDUpdate(const nnvm::NodeAttrs& attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const SGDParam& param = nnvm::get<SGDParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    if (param.clip_gradient >= 0.0f) {
      Assign(out, req[0],
             scalar<DType>(1.f-param.lr*param.wd)*weight
               - scalar<DType>(param.lr)
                 * F<clip>(scalar<DType>(param.rescale_grad)*grad,
                           DType(param.clip_gradient)));
    } else {
      Assign(out, req[0],
             scalar<DType>(1.f-param.lr*param.wd)*weight
               - scalar<DType>(param.lr*param.rescale_grad)*grad);
    }
  });
}

template<typename xpu>
inline void SGDMomUpdate(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  SGDParam param = nnvm::get<SGDParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mom = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    if (param.clip_gradient >= 0.0f) {
      mom = scalar<DType>(param.momentum)*mom
              - scalar<DType>(param.lr*param.wd)*weight
              - scalar<DType>(param.lr)
                * F<clip>(scalar<DType>(param.rescale_grad)*grad,
                          DType(param.clip_gradient));
    } else {
      mom = scalar<DType>(param.momentum)*mom
              - scalar<DType>(param.lr*param.wd)*weight
              - scalar<DType>(param.lr*param.rescale_grad)*grad;
    }
    Assign(out, req[0], weight + mom);
  });
}

struct AdamParam : public dmlc::Parameter<AdamParam> {
  float lr;
  float beta1;
  float beta2;
  float epsilon;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(AdamParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("learning_rate");
    DMLC_DECLARE_FIELD(beta1)
    .set_default(0.9f)
    .describe("beta1");
    DMLC_DECLARE_FIELD(beta2)
    .set_default(0.999f)
    .describe("beta2");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-8f)
    .describe("epsilon");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("weight decay");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("rescale gradient as grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("If greater than 0, clip gradient to "
              "grad = max(min(grad, -clip_gradient), clip_gradient). "
              "Otherwise turned off.");
  }
};

template<typename xpu>
inline void AdamUpdate(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const AdamParam& param = nnvm::get<AdamParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mean = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> var = inputs[3].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    if (param.clip_gradient >= 0.0f) {
      mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) *
          F<clip>(scalar<DType>(param.rescale_grad)*grad,
                  DType(param.clip_gradient));
      var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2)*F<square>(
          F<clip>(scalar<DType>(param.rescale_grad)*grad,
                  DType(param.clip_gradient)));
    } else {
      mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) *
          scalar<DType>(param.rescale_grad) * grad;
      var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2) *
          F<square>(scalar<DType>(param.rescale_grad)*grad);
    }
    Assign(out, req[0],
           scalar<DType>(1.f-param.lr*param.wd)*weight
             - scalar<DType>(param.lr)*mean/(F<square_root>(var)+scalar<DType>(param.epsilon)));
  });
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_OPTIMIZER_OP_INL_H_
