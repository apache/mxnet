/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_UNARY_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_UNARY_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

template<typename xpu, typename OP>
void UnaryForward_(const TBlob& src,
                   const EnvArguments& env,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = ret->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req, F<OP>(src.FlatTo1D<xpu, DType>(s)));
  });
}

// backward function that takes input value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseIn_(const OutputGrad& out_grad,
                         const Input0& in_data0,
                         const EnvArguments& env,
                         TBlob *in_grad,
                         OpReqType req,
                         RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, in_data0.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> igrad = in_grad->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (F<OP>(in_data0.data.FlatTo1D<xpu, DType>(s)) *
                     out_grad.data.FlatTo1D<xpu, DType>(s)));
  });
}

// backward function that takes output value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseOut_(const OutputGrad& out_grad,
                          const OutputValue& out_value,
                          const EnvArguments& env,
                          TBlob *in_grad,
                          OpReqType req,
                          RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, out_value.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> igrad = in_grad->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (F<OP>(out_value.data.FlatTo1D<xpu, DType>(s)) *
                     out_grad.data.FlatTo1D<xpu, DType>(s)));
    });
}

MXNET_REGISTER_SIMPLE_OP(abs, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::abs>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::sign>, kInplaceOutIn)
.describe("Take absolute value of the src");
// sign
MXNET_REGISTER_SIMPLE_OP(sign, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::sign>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::sign_grad>, kInplaceOutIn)
.describe("Take sign value of the src");
// round
MXNET_REGISTER_SIMPLE_OP(round, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::round>, kInplaceInOut)
.describe("Take round value of the src");
// ceil
MXNET_REGISTER_SIMPLE_OP(ceil, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::ceil>, kInplaceInOut)
.describe("Take ceil value of the src");
// floor
MXNET_REGISTER_SIMPLE_OP(floor, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::floor>, kInplaceInOut)
.describe("Take floor value of the src");
// square
MXNET_REGISTER_SIMPLE_OP(square, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::square>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::square_grad>, kInplaceOutIn)
.describe("Take square of the src");
// sqrt
MXNET_REGISTER_SIMPLE_OP(sqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::square_root>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, mshadow_op::square_root_grad>, kInplaceOutIn)
.describe("Take sqrt of the src");
// rsqrt
MXNET_REGISTER_SIMPLE_OP(rsqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::reciprocal_square_root>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              UnaryBackwardUseIn_<XPU, mshadow_op::reciprocal_square_root_grad>, kInplaceOutIn)
.describe("Take rsqrt of the src");
// exp
MXNET_REGISTER_SIMPLE_OP(exp, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::exp>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, mshadow_op::identity>, kInplaceOutIn)
.describe("Take exp of the src");
// log
MXNET_REGISTER_SIMPLE_OP(log, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::log>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::log_grad>, kInplaceOutIn)
.describe("Take log of the src");
// cos
MXNET_REGISTER_SIMPLE_OP(cos, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::cos>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::cos_grad>, kInplaceOutIn)
.describe("Take cos of the src");
// sin
MXNET_REGISTER_SIMPLE_OP(sin, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, mshadow_op::sin>, kInplaceInOut)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, mshadow_op::sin_grad>, kInplaceOutIn)
.describe("Take sin of the src");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_UNARY_OP_INL_H_
