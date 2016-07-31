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
#include "./broadcast_reduce_op_common.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

struct ReduceAxisParam : public dmlc::Parameter<ReduceAxisParam> {
  bool keepdims;
  TShape axis;
  DMLC_DECLARE_PARAMETER(ReduceAxisParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
      .describe("Same as Numpy. The axes to perform the reduction."
                "If left empty, a global reduction will be performed.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("Same as Numpy. If keepdims is set to true, "
      "the axis which is reduced is left in the result as dimension with size one.");
  }
};

struct BroadcastAxisParam : public dmlc::Parameter<BroadcastAxisParam> {
  TShape axis;
  TShape size;
  DMLC_DECLARE_PARAMETER(BroadcastAxisParam) {
    DMLC_DECLARE_FIELD(axis).set_default(TShape())
      .describe("The axes to perform the broadcasting.");
    DMLC_DECLARE_FIELD(size).set_default(TShape())
      .describe("Target sizes of the broadcasting axes.");
  }
};

struct BroadcastToParam : public dmlc::Parameter<BroadcastToParam> {
  TShape shape;
  DMLC_DECLARE_PARAMETER(BroadcastToParam) {
    DMLC_DECLARE_FIELD(shape).set_default(TShape())
      .describe("The shape of the desired array."
                " We can set the dim to zero if it's same as the original."
                " E.g `A = broadcast_to(B, shape=(10, 0, 0))` "
                "has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.");
  }
};

inline TShape ReduceAxisShape(const TShape& ishape,
  const EnvArguments& env) {
  ReduceAxisParam param;
  param.Init(env.kwargs);
  std::vector<index_t> axes = ParseAxes_(param.axis, ishape.ndim());
  if (axes.size() == 0) {
    for (index_t i = 0; i < ishape.ndim(); ++i) {
      axes.push_back(i);
    }
  }
  std::vector<mshadow::index_t> shape;
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (!std::binary_search(axes.begin(), axes.end(), i)) {
      shape.push_back(ishape[i]);
    } else if (param.keepdims) {
      shape.push_back(1);
    }
  }
  // We need to treat the global reduction case specially to avoid an empty output TShape.
  if (shape.size() == 0) {
    shape.push_back(1);
  }
  return TShape(shape.begin(), shape.end());
}

inline TShape BroadcastAxisShape(const TShape& ishape,
  const EnvArguments& env) {
  BroadcastAxisParam param;
  param.Init(env.kwargs);
  CHECK_EQ(param.axis.ndim(), param.size.ndim());
  TShape ret = ishape;
  for (index_t i = 0; i < param.axis.ndim(); i++) {
    CHECK_EQ(ishape[param.axis[i]], 1) <<
      "Size of the broadcasting axis in the source must be 1, axis=" << param.axis
      << ", size=" << param.size;
    ret[param.axis[i]] = param.size[i];
  }
  return ret;
}

inline TShape BroadcastToShape(const TShape& ishape,
  const EnvArguments& env) {
  BroadcastToParam param;
  param.Init(env.kwargs);
  CHECK_EQ(param.shape.ndim(), ishape.ndim());
  TShape ret = ishape;
  for (index_t i = 0; i < param.shape.ndim(); i++) {
    if (param.shape[i] > 0 && (param.shape[i] != ishape[i])) {
      CHECK_EQ(ishape[i], 1) <<
        "Size of the broadcasting axis in the source must be 1, src_shape=" << ishape
        << ", broadcast_to=" << param.shape;
      ret[i] = param.shape[i];
    }
  }
  return ret;
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

// Reduce the given axis
template<typename xpu, typename Reducer>
void ReduceAxisImpl_(const TBlob &src,
                     const EnvArguments& env,
                     TBlob *ret,
                     OpReqType req,
                     RunContext ctx,
                     TShape axes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  // If the axes is empty, we just need to give an identity mapping.
  if (axes.ndim() == 0) {
    MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
      Tensor<xpu, 2, DType> in = src.FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req, F<mshadow_op::identity>(in));
    });
    return;
  }
  bool is_contiguous_axes;
  index_t reducing_size;
  CheckContiguousAxes_(&is_contiguous_axes, &reducing_size, axes, src.shape_);
  if (is_contiguous_axes) {
    MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
      Tensor<xpu, 3, DType> in = src.FlatTo3D<xpu, DType>(axes[0], axes[axes.ndim() - 1], s);
      Tensor<xpu, 1, DType> out =
        ret->get_with_shape<xpu, 1, DType>(mshadow::Shape1(ret->Size()), s);
      ReduceAxesAssign<Reducer>(out, req, TShape(1), in);
    });
  } else {
    Shape<MXNET_SPECIAL_MAX_NDIM> padded_shape_;
    for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; ++i) {
      padded_shape_[i] = (i < src.ndim()) ? src.shape_[i] : 1;
    }
    MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> in =
        src.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(padded_shape_, s);
      Tensor<xpu, 1, DType> out =
        ret->get_with_shape<xpu, 1, DType>(mshadow::Shape1(ret->Size()), s);
      ReduceAxesAssign<Reducer>(out, req, axes, in);
    });
  }
}

// Broadcast the given axis to the given broadcasting size
template<typename xpu>
void BroadcastAxisImpl_(const TBlob &src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx,
  const TShape &axes,
  const TShape &bsizes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  // If the axes is empty, we just need to give an identity mapping.
  if (axes.ndim() == 0) {
    MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
      Tensor<xpu, 2, DType> in = src.FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req, F<mshadow_op::identity>(in));
    });
    return;
  }
  bool is_contiguous_axes;
  index_t broadcasting_size;
  CheckContiguousAxes_(&is_contiguous_axes, &broadcasting_size, axes, ret->shape_);
  if (is_contiguous_axes) {
    MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
      Tensor<xpu, 3, DType> out = ret->FlatTo3D<xpu, DType>(axes[0], axes[axes.ndim() - 1], s);
      Tensor<xpu, 3, DType> in =
        src.get_with_shape<xpu, 3, DType>(Shape3(out.shape_[0], 1, out.shape_[2]), s);
      ASSIGN_DISPATCH(out, req, broadcast_keepdim(in, 1, broadcasting_size));
    });
  } else {
    CHECK(ret->ndim() <= MXNET_SPECIAL_MAX_NDIM) << "non-contiguous axis supports ndim up to "
                                                 << MXNET_SPECIAL_MAX_NDIM;
    Shape<MXNET_SPECIAL_MAX_NDIM> padded_src_shape_;
    Shape<MXNET_SPECIAL_MAX_NDIM> padded_ret_shape_;
    for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; ++i) {
      padded_ret_shape_[i] = (i < ret->ndim()) ? ret->shape_[i] : 1;
    }
    padded_src_shape_ = padded_ret_shape_;
    for (index_t i = 0; i < axes.ndim(); ++i) {
      padded_src_shape_[axes[i]] = 1;
    }
    MSHADOW_REAL_TYPE_SWITCH(src.type_flag_, DType, {
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> in =
        src.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(padded_src_shape_, s);
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> out =
        ret->get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(padded_ret_shape_, s);
      ASSIGN_DISPATCH(out, req, broadcast_multi_axes(in, axes, bsizes));
    });
  }
}

// Forward pass of reduce over the given axis
template<typename xpu, typename Reducer>
void ReduceAxis(const TBlob &src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx) {
  using namespace mshadow::expr;
  ReduceAxisParam param;
  param.Init(env.kwargs);
  std::vector<index_t> axes = ParseAxes_(param.axis, src.ndim());
  if (axes.size() == 0) {
    for (index_t i = 0; i < src.ndim(); i++) {
      axes.push_back(i);
    }
  }
  ReduceAxisImpl_<xpu, Reducer>(src, env, ret, req, ctx,
                                TShape(axes.begin(), axes.end()));
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
  std::vector<index_t> axes = ParseAxes_(param.axis, in_grad->ndim());
  if (axes.size() == 0) {
    for (index_t i = 0; i < in_grad->ndim(); i++) {
      axes.push_back(i);
    }
  }
  std::vector<size_t> bsizes;
  for (std::vector<index_t>::iterator it = axes.begin(); it != axes.end(); ++it) {
    bsizes.push_back(in_grad->shape_[*it]);
  }
  BroadcastAxisImpl_<xpu>(out_grad.data, env, in_grad, req, ctx,
                          TShape(axes.begin(), axes.end()), TShape(bsizes.begin(), bsizes.end()));
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
  std::vector<index_t> axes = ParseAxes_(param.axis, src.ndim());
  std::vector<size_t> bsizes;
  for (std::vector<index_t>::iterator it = axes.begin(); it != axes.end(); ++it) {
    bsizes.push_back(ret->shape_[*it]);
  }
  BroadcastAxisImpl_<xpu>(src, env, ret, req, ctx,
                          TShape(axes.begin(), axes.end()), TShape(bsizes.begin(), bsizes.end()));
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
  BroadcastAxisParam param;
  param.Init(env.kwargs);
  std::vector<index_t> axes = ParseAxes_(param.axis, in_grad->ndim());
  ReduceAxisImpl_<xpu, mshadow::red::sum>(out_grad.data, env, in_grad, req, ctx,
                                          TShape(axes.begin(), axes.end()));
}

// Forward pass of broadcast_to
template<typename xpu>
void BroadcastTo(const TBlob &src,
  const EnvArguments& env,
  TBlob *ret,
  OpReqType req,
  RunContext ctx) {
  using namespace mshadow::expr;
  std::vector<index_t> axes;
  std::vector<size_t> bsizes;
  for (index_t i = 0; i < src.shape_.ndim(); ++i) {
    if (src.shape_[i] != ret->shape_[i]) {
      axes.push_back(i);
      bsizes.push_back(ret->shape_[i]);
    }
  }
  BroadcastAxisImpl_<xpu>(src, env, ret, req, ctx,
                          TShape(axes.begin(), axes.end()), TShape(bsizes.begin(), bsizes.end()));
}

// Backward pass of broadcast_to
template<typename xpu>
void BroadcastToGrad_(const OutputGrad& out_grad,
  const EnvArguments& env,
  TBlob *in_grad,
  OpReqType req,
  RunContext ctx) {
  using namespace mxnet::op;
  using namespace mshadow::expr;
  std::vector<index_t> axes;
  for (index_t i = 0; i < in_grad->shape_.ndim(); ++i) {
    if (out_grad.data.shape_[i] != in_grad->shape_[i]) {
      axes.push_back(i);
    }
  }
  ReduceAxisImpl_<xpu, mshadow::red::sum>(out_grad.data, env, in_grad, req, ctx,
                                          TShape(axes.begin(), axes.end()));
}


// L2 norm
MXNET_REGISTER_SIMPLE_OP(norm, XPU)
.set_function(XPU::kDevMask, L2Norm<XPU>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ScalarShape)
.describe("Take L2 norm of the src."
          "The result will be ndarray of shape (1,) on the same device.");

// Max
MXNET_REGISTER_SIMPLE_OP(max, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::maximum>,
kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.describe("Take max of the src in the given axis and returns a NDArray. Follows numpy semantics.")
.add_arguments(ReduceAxisParam::__FIELDS__());

// Min
MXNET_REGISTER_SIMPLE_OP(min, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::minimum>,
kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.describe("Take min of the src in the given axis and returns a NDArray. Follows numpy semantics.")
.add_arguments(ReduceAxisParam::__FIELDS__());

// Sum
MXNET_REGISTER_SIMPLE_OP(sum, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::sum>,
kNoInplace, kRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.set_gradient(XPU::kDevMask, SumAxisGrad_<XPU>, kNoInplace)
.describe("Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.")
.add_arguments(ReduceAxisParam::__FIELDS__());

// max_axis
MXNET_REGISTER_SIMPLE_OP(max_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::maximum>,
              kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.describe("(Depreciated! Use max instead!)"
          " Take max of the src in the given axis and returns a NDArray. Follows numpy semantics.")
.add_arguments(ReduceAxisParam::__FIELDS__());

// min_axis
MXNET_REGISTER_SIMPLE_OP(min_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::minimum>,
              kNoInplace, kNotRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.describe("(Depreciated! Use min instead!)"
          " Take min of the src in the given axis and returns a NDArray. Follows numpy semantics.")
.add_arguments(ReduceAxisParam::__FIELDS__());

// sum_axis
MXNET_REGISTER_SIMPLE_OP(sum_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReduceAxis<XPU, mshadow::red::sum>,
              kNoInplace, kRegisterSymbolic)
.set_shape_function(ReduceAxisShape)
.set_gradient(XPU::kDevMask, SumAxisGrad_<XPU>, kNoInplace)
.describe("(Depreciated! Use sum instead!)"
          " Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.")
.add_arguments(ReduceAxisParam::__FIELDS__());

// argmax channel
MXNET_REGISTER_SIMPLE_OP(argmax_channel, XPU)
.set_function(XPU::kDevMask, ReduceChannel<XPU, mshadow::red::maximum>,
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
          "The original size of the broadcasting axis must be 1.")
.add_arguments(BroadcastAxisParam::__FIELDS__());

// broadcast_to
MXNET_REGISTER_SIMPLE_OP(broadcast_to, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, BroadcastTo<XPU>,
kNoInplace, kRegisterSymbolic)
.set_shape_function(BroadcastToShape)
.set_gradient(XPU::kDevMask, BroadcastToGrad_<XPU>, kNoInplace)
.describe("Broadcast data to the target shape. "
          "The original size of the broadcasting axis must be 1.")
.add_arguments(BroadcastToParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BROADCAST_REDUCE_OP_INL_H_
