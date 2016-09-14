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
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = ret->get<xpu, 1, DType>(s);
    mshadow::Tensor<xpu, 1, DType> in =
      src.get_with_shape<xpu, 1, DType>(mshadow::Shape1(src.shape_.Size()), s);
    mshadow::VectorDot(out, in, in);
    ASSIGN_DISPATCH(out, req, mshadow::expr::F<mxnet::op::mshadow_op::square_root>(out));
  });
}

template<typename xpu, typename Reducer>
void ReduceChannel(const TBlob &src,
                   const EnvArguments& env,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
    Tensor<xpu, 2, DType> out = ret->get_with_shape<xpu, 2, DType>(
    Shape2(src.shape_[0], src.Size() / src.shape_[0] / src.shape_[1]),
    s);
    Tensor<xpu, 3, DType> in = src.get_with_shape<xpu, 3, DType>(
      Shape3(src.shape_[0], src.shape_[1], src.Size() / src.shape_[0] / src.shape_[1]),
      s);
    CHECK(req != kAddTo) << "AddTo is not supported";
    ASSIGN_DISPATCH(out, req, (reduce_with_axis<Reducer, true>(in, 1)));
  });
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

// argmax channel
MXNET_REGISTER_SIMPLE_OP(argmax_channel, XPU)
.set_function(XPU::kDevMask, ReduceChannel<XPU, mshadow::red::maximum>,
              kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceChannelShape)
.describe("Take argmax indices of each channel of the src."
          "The result will be ndarray of shape (num_channel,) on the same device.");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BROADCAST_REDUCE_OP_INL_H_
