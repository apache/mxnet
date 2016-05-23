/*!
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_op-inl.h
 * \brief Function defintion of broadcast/reduce operators.
 */
#ifndef MXNET_OPERATOR_BROADCAST_REDUCE_OP_INL_H_
#define MXNET_OPERATOR_BROADCAST_REDUCE_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {
// return a shape of scalar
inline TShape ScalarShape(const TShape& ishape,
                          const EnvArguments& env) {
  mshadow::index_t shape[] = {1};
  return TShape(shape, shape + 1);
}

template<typename xpu>
void L2Norm(const TBlob &src,
            const EnvArguments& env,
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
            const EnvArguments& env,
            TBlob *ret,
            OpReqType req,
            RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 1> out = ret->get<xpu, 1, real_t>(s);
  mshadow::Tensor<xpu, 2> in =
      src.get_with_shape<xpu, 2, real_t>(mshadow::Shape2(1, src.shape_.Size()), s);
  out = mshadow::expr::reduce_except_dim<0, Reducer>(in);
}

// backward function that takes input value of the op
template<typename xpu>
void SumBackward_(const OutputGrad& scale,
                  const EnvArguments& env,
                  TBlob *in_grad,
                  OpReqType req,
                  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, scale.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mscale = scale.data.get<xpu, 1, DType>(s);
      mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(igrad, req,
                      broadcast_scalar(mscale, igrad.shape_));
  });
}

template <typename xpu, typename Reducer>
void ReduceMid(TBlob const& src,
               const EnvArguments& env,
               TBlob* ret,
               OpReqType,
               RunContext ctx) {
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2> out = ret->get<xpu, 2, real_t>(s);
  mshadow::Tensor<xpu, 3> in = src.get<xpu, 3, real_t>(s);
  out = mshadow::expr::reduce_with_axis<Reducer, false>(in, 1);
}

// backward function that takes input value of the op
template<typename xpu>
void SumMidBackward_(const OutputGrad& out_grad,
  const EnvArguments& env,
  TBlob *in_grad,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> ograd = out_grad.data.get<xpu, 2, DType>(s);
    mshadow::Tensor<xpu, 3, DType> igrad = in_grad->get<xpu, 3, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
      broadcast_with_axis(ograd, 0, igrad.shape_[1]));
  });
}

inline TShape ReduceMidShape(const TShape& ishape,
                             const EnvArguments& env)  {
  CHECK_EQ(ishape.ndim(), 3) << "Input shape must be 3 dimensional.";
  std::vector<mshadow::index_t> shape;
  shape.push_back(ishape[0]);
  shape.push_back(ishape[2]);
  return TShape(shape.begin(), shape.end());
}

template<typename xpu, typename Reducer, bool get_mask>
void ReduceChannel(const TBlob &src,
                   const EnvArguments& env,
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
inline TShape ReduceChannelShape(const TShape& ishape,
                                 const EnvArguments& env) {
  std::vector<mshadow::index_t> shape;
  shape.push_back(ishape[0]);
  for (index_t i = 2; i < ishape.ndim(); ++i) {
    shape.push_back(ishape[i]);
  }
  return TShape(shape.begin(), shape.end());
}




// L2 norm
MXNET_REGISTER_SIMPLE_OP(norm, XPU)
.set_function(XPU::kDevMask, L2Norm<XPU>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ScalarShape)
.describe("Take L2 norm of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Max
MXNET_REGISTER_SIMPLE_OP(max, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::maximum>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ScalarShape)
.describe("Take max of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Min
MXNET_REGISTER_SIMPLE_OP(min, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::minimum>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ScalarShape)
.describe("Take min of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Sum
MXNET_REGISTER_SIMPLE_OP(sum, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::sum>, kNoInplace, kRegisterSymbolic)
.set_shape_function(ScalarShape)
.set_gradient(XPU::kDevMask, SumBackward_<XPU>, kNoInplace)
.describe("Take sum of the src."
          "The result will be ndarray of shape (1,) on the same device.");

// sum_mid
MXNET_REGISTER_SIMPLE_OP(sum_mid_internal, XPU)
.set_function(XPU::kDevMask, ReduceMid<XPU, mshadow::red::sum>, kNoInplace, kRegisterSymbolic)
.set_shape_function(ReduceMidShape)
.set_gradient(XPU::kDevMask, SumMidBackward_<XPU>, kNoInplace)
.describe("Take sum on medium dimension of the 3D src.");

// argmax channel
MXNET_REGISTER_SIMPLE_OP(argmax_channel, XPU)
.set_function(XPU::kDevMask, ReduceChannel<XPU, mshadow::red::maximum, true>,
              kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceChannelShape)
.describe("Take argmax indices of each channel of the src."
          "The result will be ndarray of shape (num_channel,) on the same device.");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BROADCAST_REDUCE_OP_INL_H_
