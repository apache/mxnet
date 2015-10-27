/*!
 *  Copyright (c) 2015 by Contributors
 * \file unary-function-inl.h
 * \brief the real execution functions of ndarray operations
 */
#ifndef MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
#define MXNET_NDARRAY_UNARY_FUNCTION_INL_H_

#include "../common/tblob_op_registry.h"
#include "../operator/mshadow_op.h"
#include "../operator/operator_common.h"
#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace ndarray {

using namespace common; // NOLINT(*)

template<typename xpu, typename OP>
void UnaryForward_(const TBlob &src,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2> out = ret->FlatTo2D<xpu, real_t>(s);
  Assign(out, req, F<OP>(src.FlatTo2D<xpu, real_t>(s)));
}

// backward function that takes input value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseIn_(const arg::OutGrad& out_grad,
                         const arg::Input0& in_data0,
                         TBlob *in_grad,
                         OpReqType req,
                         RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2> igrad = in_grad->FlatTo2D<xpu, real_t>(s);
  Assign(igrad, req,
         F<OP>(in_data0.data.FlatTo2D<xpu, real_t>(s)) *
         out_grad.data.FlatTo2D<xpu, real_t>());
}

// backward function that takes output value of the op
template<typename xpu, typename OP>
void UnaryBackwardUseOut_(const arg::OutGrad& out_grad,
                          const arg::OutValue& out_value,
                          TBlob *in_grad,
                          OpReqType req,
                          RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2> igrad = in_grad->FlatTo2D<xpu, real_t>(s);
  Assign(igrad, req,
         F<OP>(out_value.data.FlatTo2D<xpu, real_t>(s)) *
         out_grad.data.FlatTo2D<xpu, real_t>());
}

// return a shape of scalar
inline TShape ScalarShape(const TShape& ishape) {
  mshadow::index_t shape[] = {1};
  return TShape(shape, shape + 1);
}

template<typename xpu>
void L2Norm(const TBlob &src,
            TBlob *ret,
            OpReqType req,
            RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 1> out = ret->get<xpu, 1, real_t>(s);
  mshadow::Tensor<xpu, 1> in =
      src.get_with_shape<xpu, 1, real_t>(mshadow::Shape1(src.shape_.Size()));
  mshadow::VectorDot(out, in, in);
  out = mshadow::expr::F<mxnet::op::mshadow_op::square_root>(out);
}

// Register all unary operations here
// Square
struct square_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 2.0f * a;
  }
};
// The true means inplace can be enabled.
MXNET_REGISTER_TBLOB_FUN(square, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::square>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, square_grad>, true)
.describe("Take square of the src");

// square root
struct square_root_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 0.5f / a;
  }
};
MXNET_REGISTER_TBLOB_FUN(sqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::square_root>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, square_root_grad>, true)
.describe("Take square root of the src");

// exp
MXNET_REGISTER_TBLOB_FUN(exp, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::exp>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, op::mshadow_op::exp_grad>, true)
.describe("Take exp of the src");

//log
MXNET_REGISTER_TBLOB_FUN(log, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::log>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, op::mshadow_op::log_grad>, true)
.describe("Take log of the src");

// L2 norm
MXNET_REGISTER_TBLOB_FUN(norm, XPU)
.set_function(XPU::kDevMask, L2Norm<XPU>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take L2 norm of the src."
          "The result will be ndarray of shape (1,) on the same device.");
}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
