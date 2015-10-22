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


// Square root
struct square_root_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 0.5f / a;
  }
};
MXNET_REGISTER_TBLOB_FUN(sqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::square_root>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, square_root_grad>, true)
.describe("Take square root of the src");
}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
