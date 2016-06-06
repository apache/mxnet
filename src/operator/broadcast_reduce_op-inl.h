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

struct ReduceAxisParam : public dmlc::Parameter<ReduceAxisParam> {
  bool keepdims;
  int axis;
  DMLC_DECLARE_PARAMETER(ReduceAxisParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1).set_lower_bound(-1)
      .describe("The axis to perform the reduction. axis=-1 means to reduce all dimensions");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("Same as Numpy. If keepdims is set to true, "
      "the axis which is reduced is left in the result as dimension with size one.");
  }
};

struct BroadcastAxisParam : public dmlc::Parameter<BroadcastAxisParam> {
  int axis;
  int size;
  DMLC_DECLARE_PARAMETER(BroadcastAxisParam) {
    DMLC_DECLARE_FIELD(axis).set_default(0).set_lower_bound(0)
      .describe("The target axis of broadcasting.");
    DMLC_DECLARE_FIELD(size).set_default(0).set_lower_bound(1)
      .describe("Size of the broadcasting axis.");
  }
};

inline TShape ReduceAxisShape(const TShape& ishape,
  const EnvArguments& env) {
  ReduceAxisParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < ishape.ndim() || -1 == param.axis) <<
    "axis must be smaller than the source ndim or equal to -1! Received axis=" <<
    param.axis << ", src_ndim=" << ishape.ndim();
  if (param.axis == -1 || (1 == ishape.ndim())) {
    if (param.keepdims) {
      return TShape(ishape.ndim());
    } else {
      return TShape(1);
    }
  }
  std::vector<mshadow::index_t> shape;
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (i == param.axis) {
      if (param.keepdims) {
        shape.push_back(1);
      }
    } else {
      shape.push_back(ishape[i]);
    }
  }
  return TShape(shape.begin(), shape.end());
}

inline TShape BroadcastAxisShape(const TShape& ishape,
  const EnvArguments& env) {
  BroadcastAxisParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < ishape.ndim()) <<
    "axis must be smaller than the source ndim" << param.axis << ", src_ndim=" << ishape.ndim();
  CHECK_EQ(ishape[param.axis], 1) <<
    "Size of the broadcasting axis in the source must be 1, axis=" << param.axis
    << ", size=" << ishape[param.axis];
  std::vector<mshadow::index_t> shape;
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (i != param.axis) {
      shape.push_back(ishape[i]);
    } else {
      shape.push_back(param.size);
    }
  }
  return TShape(shape.begin(), shape.end());
}

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

// Reduce the given axis
template<typename xpu, typename Reducer, bool get_mask>
void ReduceAxisImpl_(const TBlob &src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx,
  int axis,
  bool keepdims) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (-1 == axis) {
    // Reduce all dimensions if axis == -1
    mshadow::Tensor<xpu, 2> in =
      src.get_with_shape<xpu, 2, real_t>(mshadow::Shape2(1, src.shape_.Size()), s);
    mshadow::Tensor<xpu, 1> out =
      ret->get_with_shape<xpu, 1, real_t>(mshadow::Shape1(ret->shape_.Size()), s);
    out = mshadow::expr::reduce_except_dim<0, Reducer>(in);
    return;
  }
  int trailing = 1;
  int leading = 1;
  for (int i = 0; i < src.shape_.ndim(); ++i) {
    if (i < axis) {
      leading *= src.shape_[i];
    } else if (i > axis) {
      trailing *= src.shape_[i];
    }
  }
  mshadow::Tensor<xpu, 3> in =
    src.get_with_shape<xpu, 3, real_t>(mshadow::Shape3(leading, src.shape_[axis], trailing), s);
  mshadow::Tensor<xpu, 2> out =
    ret->get_with_shape<xpu, 2, real_t>(mshadow::Shape2(leading, trailing), s);
  out = mshadow::expr::reduce_with_axis<Reducer, get_mask>(in, 1);
}

// Broadcast the given axis to the given broadcasting size
template<typename xpu>
void BroadcastAxisImpl_(const TBlob &src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx,
  int axis,
  int bsize,
  bool keepdims) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (axis == -1) {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> in =
        src.get_with_shape<xpu, 1, DType>(mshadow::Shape1(src.shape_.Size()), s);
      mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req,
        broadcast_scalar(in, out.shape_));
    });
    return;
  }
  int trailing = 1;
  int leading = 1;
  for (int i = 0; i < ret->shape_.ndim(); ++i) {
    if (i < axis) {
      leading *= ret->shape_[i];
    } else if (i > axis) {
      trailing *= ret->shape_[i];
    }
  }
  mshadow::Tensor<xpu, 2> in =
    src.get_with_shape<xpu, 2, real_t>(mshadow::Shape2(leading, trailing), s);
  mshadow::Tensor<xpu, 3> out =
    ret->get_with_shape<xpu, 3, real_t>(mshadow::Shape3(leading, bsize, trailing), s);
  out = mshadow::expr::broadcast_with_axis(in, 0, bsize);
}

// Forward pass of reduce over the given axis
template<typename xpu, typename Reducer, bool get_mask>
void ReduceAxis(const TBlob &src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx) {
  using namespace mshadow::expr;
  ReduceAxisParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < src.shape_.ndim() || -1 == param.axis) <<
    "axis must be smaller than the source ndim or equals to -1!"
    " Received axis=" << param.axis << ", src_ndim=" << src.shape_.ndim();
  ReduceAxisImpl_<xpu, Reducer, get_mask>(src, env, ret, req, ctx, param.axis, param.keepdims);
}

// Backward pass of reduce over the given axis
template<typename xpu>
void SumAxisGrad_(const OutputGrad& out_grad,
  const EnvArguments& env,
  TBlob *in_grad,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  ReduceAxisParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < in_grad->shape_.ndim() || param.axis == -1) <<
    "axis must be smaller than the input grad ndim or equals to -1."
    " Received axis=" << param.axis << ", igrad_ndim=" << in_grad->shape_.ndim();
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  if (-1 == param.axis) {
    BroadcastAxisImpl_<xpu>(out_grad.data, env, in_grad, req, ctx, param.axis, 0, param.keepdims);
  } else {
    BroadcastAxisImpl_<xpu>(out_grad.data, env, in_grad, req, ctx, param.axis,
      in_grad->shape_[param.axis], param.keepdims);
  }
}

// Forward pass of broadcast over the given axis
template<typename xpu>
void BroadcastAxis(const TBlob &src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx) {
  using namespace mshadow::expr;
  BroadcastAxisParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < src.shape_.ndim()) <<
    "axis must be smaller than the source ndim" << param.axis <<
    ", src_ndim=" << src.shape_.ndim();
  CHECK_EQ(src.shape_[param.axis], 1) <<
    "Size of the broadcasting axis in the source must be 1, "
    "axis=" << param.axis << ", size=" << src.shape_[param.axis];
  BroadcastAxisImpl_<xpu>(src, env, ret, req, ctx, param.axis, param.size, true);
}

// Backward pass of broadcast over the given axis
template<typename xpu>
void BroadcastAxisGrad_(const OutputGrad& out_grad,
  const EnvArguments& env,
  TBlob *in_grad,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  BroadcastAxisParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < in_grad->shape_.ndim()) <<
    "axis must be smaller than the source ndim" << param.axis <<
    ", src_ndim=" << in_grad->shape_.ndim();
  CHECK_EQ(in_grad->shape_[param.axis], 1) <<
    "Size of the broadcasting axis in the source must be 1, "
    "axis=" << param.axis << ", size=" << in_grad->shape_[param.axis];
  CHECK_EQ(in_grad->type_flag_, out_grad.data.type_flag_)
    << "Unary function only support input/output with the same type";
  ReduceAxisImpl_<xpu, mshadow::red::sum, false>(out_grad.data, env, in_grad, req, ctx,
                                                 param.axis, true);
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
.describe("(Deprecated! Use max_axis instead.) Take max of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Min
MXNET_REGISTER_SIMPLE_OP(min, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::minimum>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ScalarShape)
.describe("(Deprecated! Use min_axis instead.) Take min of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// Sum
MXNET_REGISTER_SIMPLE_OP(sum, XPU)
.set_function(XPU::kDevMask, Reduce<XPU, mshadow::red::sum>, kNoInplace, kRegisterSymbolic)
.set_shape_function(ScalarShape)
.set_gradient(XPU::kDevMask, SumBackward_<XPU>, kNoInplace)
.describe("(Deprecated! Use sum_axis instead.) Take sum of the src."
          "The result will be ndarray of shape (1,) on the same device.");
// max_axis
MXNET_REGISTER_SIMPLE_OP(max_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::maximum, false>,
              kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.describe("Take max of the src in the given axis. axis=-1 means to reduce all the dimensions."
"The keepdims option has the same meaning as Numpy.");

// min_axis
MXNET_REGISTER_SIMPLE_OP(min_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::minimum, false>,
              kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.describe("Take min of the src in the given axis. axis=-1 means to reduce all the dimensions."
"The keepdims option has the same meaning as Numpy.");

// sum_axis
MXNET_REGISTER_SIMPLE_OP(sum_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::sum, false>,
              kNoInplace, kRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.set_gradient(XPU::kDevMask, SumAxisGrad_<XPU>, kNoInplace)
.describe("Take sum of the src in the given axis. axis=-1 means to reduce all the dimensions."
"The keepdims option has the same meaning as Numpy.");

// argmax channel
MXNET_REGISTER_SIMPLE_OP(argmax_channel, XPU)
.set_function(XPU::kDevMask, ReduceChannel<XPU, mshadow::red::maximum, true>,
              kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceChannelShape)
.describe("Take argmax indices of each channel of the src."
          "The result will be ndarray of shape (num_channel,) on the same device.");

// broadcast_axis
MXNET_REGISTER_SIMPLE_OP(broadcast_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, BroadcastAxis<XPU>,
              kNoInplace, kRegisterSymbolic)
.set_shape_function(BroadcastAxisShape)
.set_gradient(XPU::kDevMask, BroadcastAxisGrad_<XPU>, kNoInplace)
.describe("Broadcast data in the given axis to the given size. "
"The original size of the broadcasting axis must be 1.");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BROADCAST_REDUCE_OP_INL_H_
