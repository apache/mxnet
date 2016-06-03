/*!
 * Copyright (c) 2015 by Contributors
 * \file smooth_l1_unary-inl.h
 * \brief Smooth L1 loss
*/
#ifndef MXNET_OPERATOR_SMOOTH_L1_UNARY_INL_H_
#define MXNET_OPERATOR_SMOOTH_L1_UNARY_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

namespace mshadow_op {
/* Smooth L1 Loss is a loss specific for R-CNN franchise training
 * Smooth L1 Loss function
 * f(x) = 0.5 * (sigma * x) ^ 2,     x < 1 / sigma^2
 *      = |x| - 0.5 / sigma / sigma, otherwise
 * When sigma = 1, it is equivalent to Huber Loss evaluated at
 * delta = 1.
 * smooth_l1_loss = w_out * f(w_in * x)
 * with w_in, w_out provided by input_data.
 */
struct smooth_l1_loss {
  // a is x, b is sigma2
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    if (a > 1.0f / b) {
      return a - 0.5f / b;
    } else if (a < -1.0f / b) {
      return -a - 0.5f / b;
    } else {
      return 0.5f * a * a * b;
    }
  }
};  // struct smooth_l1_loss

/* The derivative of smooth l1 loss is
 * f'(x) = sigma^2 * x, x < 1 / sigma^2
 *       = sign(x),     otherwise
 */
struct smooth_l1_gradient {
  // a is x, b is sigma2
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    if (a > 1.0f / b) {
      return 1.0f;
    } else if (a < -1.0f / b) {
      return -1.0f;
    } else {
      return b * a;
    }
  }
};  // struct smooth_l1_derivative
}  // namespace mshadow_op

template<typename xpu>
void SmoothL1Forward_(const TBlob& src,
                      const EnvArguments& env,
                      TBlob *ret,
                      OpReqType req,
                      RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Unary function only support input/output with the same type";
  real_t sigma2 = env.scalar * env.scalar;
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> in = src.FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<mshadow_op::smooth_l1_loss>(in, ScalarExp<DType>(sigma2)));
  });
}

template<typename xpu>
void SmoothL1BackwardUseIn_(const OutputGrad& out_grad,
                            const Input0& in_data0,
                            const EnvArguments& env,
                            TBlob *in_grad,
                            OpReqType req,
                            RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, in_data0.data.type_flag_)
    << "Unary function only support input/output with the same type";
  real_t sigma2 = env.scalar * env.scalar;
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> src = in_data0.data.FlatTo2D<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> ograd = out_grad.data.FlatTo2D<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    ograd * F<mshadow_op::smooth_l1_gradient>(src, ScalarExp<DType>(sigma2)));
  });
}

MXNET_REGISTER_SIMPLE_OP(smooth_l1, XPU)
.set_function(XPU::kDevMask, SmoothL1Forward_<XPU>, kNoInplace)
.set_gradient(XPU::kDevMask, SmoothL1BackwardUseIn_<XPU>, kInplaceOutIn)
.set_enable_scalar(true)
.describe("Calculate Smooth L1 Loss(lhs, scalar)");

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SMOOTH_L1_UNARY_INL_H_
