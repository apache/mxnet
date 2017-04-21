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
#include "mxnet_op.h"

namespace mxnet {
namespace op {
struct SGDParam : public dmlc::Parameter<SGDParam> {
  float lr;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(SGDParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};

struct SGDKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* weight_data,
    const DType* grad_data, const DType param_clip_gradient,
    const DType param_lr, const DType param_wd, const DType param_rescale_grad,
    const OpReqType req) {
    if (param_clip_gradient >= 0.0f) {
      KERNEL_ASSIGN(out_data[i], req,
             (1.f-param_lr*param_wd)*weight_data[i]
               - (param_lr)
                 * mshadow_op::clip::Map(param_rescale_grad*grad_data[i], param_clip_gradient));
    } else {
      KERNEL_ASSIGN(out_data[i], req,
             (1.f-param_lr*param_wd)*weight_data[i]
               - (param_lr*param_rescale_grad)*grad_data[i]);
    }
  }
};

template<typename xpu>
inline void SGDUpdate(const nnvm::NodeAttrs& attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const SGDParam& param = nnvm::get<SGDParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<SGDKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient),
      static_cast<DType>(param.lr), static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), req[0]);
  });
}

struct SGDMomParam : public dmlc::Parameter<SGDMomParam> {
  float lr;
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(SGDMomParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(momentum)
    .set_default(0.0f)
    .describe("The decay rate of momentum estimates at each epoch.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};

struct SGDMomKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, DType* mom_data, const DType* weight_data,
    const DType* grad_data, const DType param_clip_gradient, const DType param_momentum,
    const DType param_lr, const DType param_wd, const DType param_rescale_grad,
    const OpReqType req) {
    if (param_clip_gradient >= 0.0f) {
      mom_data[i] = param_momentum*mom_data[i]
              - param_lr*param_wd*weight_data[i]
              - param_lr
              *mshadow_op::clip::Map(param_rescale_grad*grad_data[i], param_clip_gradient);
    } else {
      mom_data[i] = param_momentum*mom_data[i]
                - param_lr*param_wd*weight_data[i]
                - param_lr*param_rescale_grad*grad_data[i];
    }
    KERNEL_ASSIGN(out_data[i], req, weight_data[i] + mom_data[i]);
  }
};

template<typename xpu>
inline void SGDMomUpdate(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  SGDMomParam param = nnvm::get<SGDMomParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mom = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<SGDMomKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, mom.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient), static_cast<DType>(param.momentum),
      static_cast<DType>(param.lr), static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), req[0]);
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
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(beta1)
    .set_default(0.9f)
    .describe("The decay rate for the 1st moment estimates.");
    DMLC_DECLARE_FIELD(beta2)
    .set_default(0.999f)
    .describe("The decay rate for the 2nd moment estimates.");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
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

    grad = scalar<DType>(param.rescale_grad) * grad +
      scalar<DType>(param.wd) * weight;

    if (param.clip_gradient >= 0.0f) {
      mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) *
          F<clip>(grad, DType(param.clip_gradient));
      var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2)*F<square>(
          F<clip>(grad, DType(param.clip_gradient)));
    } else {
      mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) * grad;
      var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2) * F<square>(grad);
    }
    Assign(out, req[0],
           weight -
           scalar<DType>(param.lr) * mean /
           (F<square_root>(var) + scalar<DType>(param.epsilon)));
  });
}

// This RMSProp code follows the version in
// http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45)
// by Alex Graves, 2013.
struct RMSPropAlexParam : public dmlc::Parameter<RMSPropAlexParam> {
  float lr;
  float gamma1;
  float gamma2;
  float epsilon;
  float wd;
  float rescale_grad;
  float clip_gradient;
  float clip_weights;
  DMLC_DECLARE_PARAMETER(RMSPropAlexParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(gamma1).set_default(0.95f)
    .describe("Decay rate.");
    DMLC_DECLARE_FIELD(gamma2).set_default(0.9f)
    .describe("Decay rate.");
    DMLC_DECLARE_FIELD(epsilon).set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wd).set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(clip_weights)
    .set_default(-1.0f)
    .describe("Clip weights to the range of [-clip_weights, clip_weights] "
              "If clip_weights <= 0, weight clipping is turned off. "
              "weights = max(min(weights, clip_weights), -clip_weights).");
  }
};

template <typename xpu>
inline void RMSPropAlexUpdate(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const RMSPropAlexParam &param = nnvm::get<RMSPropAlexParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> state_n = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> state_g = inputs[3].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> delta = inputs[4].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

    grad = scalar<DType>(param.rescale_grad) * grad +
           scalar<DType>(param.wd) * weight;

    if (param.clip_gradient >= 0.0f) {
      state_n = scalar<DType>(1.f - param.gamma1) *
                    F<clip>(grad, DType(param.clip_gradient)) *
                    F<clip>(grad, DType(param.clip_gradient)) +
                scalar<DType>(param.gamma1) * state_n;
      state_g = scalar<DType>(1.f - param.gamma1) *
                    F<clip>(grad, DType(param.clip_gradient)) +
                scalar<DType>(param.gamma1) * state_g;
      delta = scalar<DType>(param.gamma2) * delta -
              scalar<DType>(param.lr) *
                  (F<clip>(grad, DType(param.clip_gradient)) /
                   (F<square_root>(state_n - state_g * state_g) +
                    scalar<DType>(param.epsilon)));
    } else {
      state_n = scalar<DType>(1.f - param.gamma1) * (grad * grad) +
                scalar<DType>(param.gamma1) * state_n;
      state_g = scalar<DType>(1.f - param.gamma1) * grad +
                scalar<DType>(param.gamma1) * state_g;
      delta = scalar<DType>(param.gamma2) * delta -
              scalar<DType>(param.lr) *
                  (grad / (F<square_root>(state_n - state_g * state_g) +
                           scalar<DType>(param.epsilon)));
    }

    if (param.clip_weights >= 0.0f) {
      Assign(out, req[0], F<clip>(weight + delta, DType(param.clip_weights)));
    } else {
      Assign(out, req[0], weight + delta);
    }
  });
}

// This RMSProp code follows the version in
// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
// by Tieleman & Hinton, 2012
struct RMSPropParam : public dmlc::Parameter<RMSPropParam> {
  float lr;
  float gamma1;
  float epsilon;
  float wd;
  float rescale_grad;
  float clip_gradient;
  float clip_weights;
  DMLC_DECLARE_PARAMETER(RMSPropParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(gamma1).set_default(0.95f)
    .describe("The dacay rate of momentum estimates.");
    DMLC_DECLARE_FIELD(epsilon).set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wd).set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(clip_weights)
    .set_default(-1.0f)
    .describe("Clip weights to the range of [-clip_weights, clip_weights] "
              "If clip_weights <= 0, weight clipping is turned off. "
              "weights = max(min(weights, clip_weights), -clip_weights).");
  }
};

template <typename xpu>
inline void RMSPropUpdate(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const RMSPropParam &param = nnvm::get<RMSPropParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> state_n = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

    grad = scalar<DType>(param.rescale_grad) * grad +
           scalar<DType>(param.wd) * weight;

    if (param.clip_gradient >= 0.0f) {
      state_n = scalar<DType>(1.f - param.gamma1) *
                    F<clip>(grad, DType(param.clip_gradient)) *
                    F<clip>(grad, DType(param.clip_gradient)) +
                scalar<DType>(param.gamma1) * state_n;
      if (param.clip_weights >= 0.0f) {
        Assign(out, req[0],
               F<clip>(weight -
                           scalar<DType>(param.lr) *
                               (F<clip>(grad, DType(param.clip_gradient)) /
                                (F<square_root>(state_n) +
                                 scalar<DType>(param.epsilon))),
                       DType(param.clip_weights)));
      } else {
        Assign(out, req[0], weight -
                                scalar<DType>(param.lr) *
                                    (F<clip>(grad, DType(param.clip_gradient)) /
                                     (F<square_root>(state_n) +
                                      scalar<DType>(param.epsilon))));
      }
    } else {
      state_n = scalar<DType>(1.f - param.gamma1) * (grad * grad) +
                scalar<DType>(param.gamma1) * state_n;
      if (param.clip_weights >= 0.0f) {
        Assign(out, req[0],
               F<clip>(weight -
                           scalar<DType>(param.lr) *
                               (grad / (F<square_root>(state_n) +
                                        scalar<DType>(param.epsilon))),
                       DType(param.clip_weights)));
      } else {
        Assign(out, req[0], weight -
                                scalar<DType>(param.lr) *
                                    (grad / (F<square_root>(state_n) +
                                             scalar<DType>(param.epsilon))));
      }
    }
  });
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_OPTIMIZER_OP_INL_H_
