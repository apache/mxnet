/*!
 *  Copyright (c) 2015 by Contributors
 * \file unary-function-inl.h
 * \brief the real execution functions of ndarray operations
 */
#ifndef MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
#define MXNET_NDARRAY_UNARY_FUNCTION_INL_H_

#include <vector>
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
  CHECK_EQ(ret->type_flag_, src.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    Assign(out, req, F<OP>(src.FlatTo2D<xpu, DType>(s)));
  });
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
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, in_data0.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    Assign(igrad, req,
           (F<OP>(in_data0.data.FlatTo2D<xpu, DType>(s)) *
           out_grad.data.FlatTo2D<xpu, DType>()));
  });
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
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  CHECK_EQ(in_grad->type_flag_, out_value.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    Assign(igrad, req,
           (F<OP>(out_value.data.FlatTo2D<xpu, DType>(s)) *
           out_grad.data.FlatTo2D<xpu, DType>()));
  });
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
      src.get_with_shape<xpu, 1, real_t>(mshadow::Shape1(src.shape_.Size()), s);
  mshadow::VectorDot(out, in, in);
  out = mshadow::expr::F<mxnet::op::mshadow_op::square_root>(out);
}

template<typename xpu, typename Reducer>
void Reduce(const TBlob &src,
            TBlob *ret,
            OpReqType req,
            RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 1> out = ret->get<xpu, 1, real_t>(s);
  mshadow::Tensor<xpu, 2> in =
      src.get_with_shape<xpu, 2, real_t>(mshadow::Shape2(1, src.shape_.Size()), s);
  out = mshadow::expr::reduce_except_dim<0, Reducer>(in);
}

template<typename xpu, typename Reducer, bool get_mask>
void ReduceChannel(const TBlob &src,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> out = ret->get_with_shape<xpu, 2, real_t>(
    Shape2(src.shape_[0], src.Size()/src.shape_[0]/src.shape_[1]),
    s);
  Tensor<xpu, 3> in = src.get_with_shape<xpu, 3, real_t>(
    Shape3(src.shape_[0], src.shape_[1], src.Size()/src.shape_[0]/src.shape_[1]),
    s);
  out = reduce_with_axis<Reducer, get_mask>(in, 1);
}

// return a shape of ReduceChannel output
inline TShape ReduceChannelShape(const TShape& ishape) {
  std::vector<mshadow::index_t> shape;
  shape.push_back(ishape[0]);
  for (index_t i = 2; i < ishape.ndim(); ++i) {
    shape.push_back(ishape[i]);
  }
  return TShape(shape.begin(), shape.end());
}

// Register all unary operations here
// The true means inplace can be enabled.
// abs
MXNET_REGISTER_TBLOB_FUN(abs, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::abs>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::sign>, true)
.describe("Take absolute value of the src");
// sign
MXNET_REGISTER_TBLOB_FUN(sign, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::sign>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::sign_grad>, true)
.describe("Take sign value of the src");
// round
MXNET_REGISTER_TBLOB_FUN(round, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::round>, true)
.describe("Take round value of the src");
// ceil
MXNET_REGISTER_TBLOB_FUN(ceil, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::ceil>, true)
.describe("Take ceil value of the src");
// floor
MXNET_REGISTER_TBLOB_FUN(floor, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::floor>, true)
.describe("Take floor value of the src");
// square
MXNET_REGISTER_TBLOB_FUN(square, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::square>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::square_grad>, true)
.describe("Take square of the src");
// sqrt
MXNET_REGISTER_TBLOB_FUN(sqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::square_root>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, op::mshadow_op::square_root_grad>, true)
.describe("Take sqrt of the src");
// rsqrt
MXNET_REGISTER_TBLOB_FUN(rsqrt, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::reciprocal_square_root>, true)
.set_gradient(XPU::kDevMask,
              UnaryBackwardUseIn_<XPU, op::mshadow_op::reciprocal_square_root_grad>, true)
.describe("Take rsqrt of the src");
// exp
MXNET_REGISTER_TBLOB_FUN(exp, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::exp>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseOut_<XPU, op::mshadow_op::identity>, true)
.describe("Take exp of the src");
// log
MXNET_REGISTER_TBLOB_FUN(log, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::log>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::log_grad>, true)
.describe("Take log of the src");
// cos
MXNET_REGISTER_TBLOB_FUN(cos, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::cos>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::cos_grad>, true)
.describe("Take cos of the src");
// sin
MXNET_REGISTER_TBLOB_FUN(sin, XPU)
.set_function(XPU::kDevMask, UnaryForward_<XPU, op::mshadow_op::sin>, true)
.set_gradient(XPU::kDevMask, UnaryBackwardUseIn_<XPU, op::mshadow_op::sin_grad>, true)
.describe("Take sin of the src");
// L2 norm
MXNET_REGISTER_TBLOB_FUN(norm, XPU)
.set_function(XPU::kDevMask, L2Norm<XPU>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take L2 norm of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Max
MXNET_REGISTER_TBLOB_FUN(max, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::maximum>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take max of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Min
MXNET_REGISTER_TBLOB_FUN(min, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::minimum>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take min of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Sum
MXNET_REGISTER_TBLOB_FUN(sum, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::sum>, false, false)
.set_shape_infer(ScalarShape)
.describe("Take sum of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// argmax channel
MXNET_REGISTER_TBLOB_FUN(argmax_channel, XPU)
.set_function(XPU::kDevMask, ReduceChannel<XPU, mshadow::red::maximum, true>, false, false)
.set_shape_infer(ReduceChannelShape)
.describe("Take argmax indices of each channel of the src."
          "The result will be ndarray of shape (num_channel,) on the same device.");
}  // namespace ndarray
}  // namespace mxnet
#endif  // MXNET_NDARRAY_UNARY_FUNCTION_INL_H_
