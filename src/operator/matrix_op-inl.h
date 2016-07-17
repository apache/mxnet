/*!
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_op-inl.h
 * \brief Function defintion of matrix related operators
 */
#ifndef MXNET_OPERATOR_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_MATRIX_OP_INL_H_

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

struct TransposeParam : public dmlc::Parameter<TransposeParam> {
  TShape axes;
  DMLC_DECLARE_PARAMETER(TransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Target axis order. By default the axes will be inverted.");
  }
};


template<typename xpu>
void TransposeImpl(const TBlob &src,
              TBlob *ret,
              RunContext ctx,
              const TShape &axes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    switch (axes.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = src.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = ret->get<xpu, 1, DType>(s);
      Copy(out, in, s);
      break;
     }
     case 2: {
      mshadow::Tensor<xpu, 2, DType> in = src.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      if (axes[0] == 1 && axes[1] == 0) {
        out = in.T();
      } else {
        Copy(out, in, s);
      }
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = ret->get<xpu, 3, DType>(s);
      out = transpose(in, axes.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = ret->get<xpu, 4, DType>(s);
      out = transpose(in, axes.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = ret->get<xpu, 5, DType>(s);
      out = transpose(in, axes.get<5>());
      break;
     }
     default:
      LOG(FATAL) << "Transpose support at most 5 dimensions";
      break;
    }
  });
}

// matrix transpose
template<typename xpu>
void Transpose(const TBlob &src,
               const EnvArguments& env,
               TBlob *ret,
               OpReqType req,
               RunContext ctx) {
  TransposeParam param;
  param.Init(env.kwargs);
  if (param.axes.ndim() == 0) {
    param.axes = TShape(src.shape_.ndim());
    for (index_t i = 0; i < param.axes.ndim(); ++i) {
      param.axes[i] = param.axes.ndim() - 1 - i;
    }
  }
  TransposeImpl<xpu>(src, ret, ctx, param.axes);
}

template<typename xpu>
void TransposeGrad(const OutputGrad& src,
                   const EnvArguments& env,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  TransposeParam param;
  param.Init(env.kwargs);
  TShape axes = TShape(src.data.shape_.ndim());
  if (param.axes.ndim() == 0) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
  } else {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      axes[param.axes[i]] = i;
    }
  }
  TransposeImpl<xpu>(src.data, ret, ctx, axes);
}

inline TShape TransposeShape(const TShape& shp,
                             const EnvArguments& env) {
  TransposeParam param;
  param.Init(env.kwargs);
  CHECK(shp.ndim() <= 5) << "Transpose support at most 5 dimensions";
  TShape ret(shp.ndim());
  if (param.axes.ndim() == 0) {
    for (index_t i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
  } else {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (index_t i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < shp.ndim());
      ret[i] = shp[param.axes[i]];
    }
  }
  return ret;
}


struct ExpandDimParam : public dmlc::Parameter<ExpandDimParam> {
  index_t axis;
  DMLC_DECLARE_PARAMETER(ExpandDimParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("Position (amongst axes) where new axis is to be inserted.");
  }
};


inline TShape ExpandDimShape(const TShape& shp,
                             const EnvArguments& env) {
  ExpandDimParam param;
  param.Init(env.kwargs);
  CHECK_LE(param.axis, shp.ndim())
      << "axis must be smaller equal to the dimension of the array";
  std::vector<index_t> idx(shp.data(), shp.data() + shp.ndim());
  idx.insert(idx.begin() + param.axis, 1);
  return TShape(idx.begin(), idx.end());
}


template<typename xpu>
void ReshapeImpl(const TBlob &src,
                 const EnvArguments& env,
                 TBlob *ret,
                 OpReqType req,
                 RunContext ctx) {
  if (req == kNullOp) return;
  if (req == kWriteInplace) {
    CHECK(ret->CheckContiguous() && src.CheckContiguous());
  }
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      using namespace mshadow::expr;
      mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mout = src.get_with_shape<xpu, 2, DType>(out.shape_, s);
      ASSIGN_DISPATCH(out, req, F<mshadow::op::identity>(mout));
    });
}

template<typename xpu>
void ReshapeGrad_(const OutputGrad& out_grad,
                  const EnvArguments& env,
                  TBlob *in_grad,
                  OpReqType req,
                  RunContext ctx) {
  ReshapeImpl<xpu>(
      out_grad.data, env, in_grad, req, ctx);
}


template<typename xpu>
void DotForward_(const TBlob& lhs,
                 const TBlob& rhs,
                 const EnvArguments& env,
                 TBlob *ret,
                 OpReqType req,
                 RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, mshadow::kFloat32)
      << "dot only support 32 bit float so far";

  if (lhs.shape_.ndim() == 2 && rhs.shape_.ndim() == 2) {
    mshadow::Tensor<xpu, 2, real_t> out = ret->FlatTo2D<xpu, real_t>(s);
    ASSIGN_DISPATCH(out, req,
                    dot(lhs.get<xpu, 2, real_t>(s),
                        rhs.get<xpu, 2, real_t>(s)));
  } else if (lhs.shape_.ndim() == 1 && rhs.shape_.ndim() == 1) {
    CHECK_NE(req, kAddTo) << "AddTo not yet suported";
    mshadow::Tensor<xpu, 1, real_t> out = ret->get<xpu, 1, real_t>(s);
    mshadow::VectorDot(out,
                       lhs.get<xpu, 1, real_t>(s),
                       rhs.get<xpu, 1, real_t>(s));
  } else {
    LOG(FATAL) << "not reached";
  }
}

template<typename xpu>
void DotBackward_(const OutputGrad& out_grad,
                  const Input0& lhs,
                  const Input1& rhs,
                  const EnvArguments& env,
                  TBlob* lhs_grad,
                  TBlob* rhs_grad,
                  OpReqType req_lhs_grad,
                  OpReqType req_rhs_grad,
                  RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req_rhs_grad, kWriteInplace);
  CHECK_NE(req_lhs_grad, kWriteInplace);

  if (lhs.data.shape_.ndim() == 2 && rhs.data.shape_.ndim() == 2) {
    mshadow::Tensor<xpu, 2, real_t> mout_grad = out_grad.data.get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mlhs_data = lhs.data.get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mrhs_data = rhs.data.get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mlhs_grad = lhs_grad->get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mrhs_grad = rhs_grad->get<xpu, 2, real_t>(s);
    ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, dot(mlhs_data.T(), mout_grad));
    ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, dot(mout_grad, mrhs_data.T()));
  } else if (lhs.data.shape_.ndim() == 1 && rhs.data.shape_.ndim() == 1) {
    mshadow::Tensor<xpu, 1, real_t> mout_grad = out_grad.data.get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mlhs_data = lhs.data.get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mrhs_data = rhs.data.get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mlhs_grad = lhs_grad->get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mrhs_grad = rhs_grad->get<xpu, 1, real_t>(s);
    ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                    broadcast_scalar(mout_grad, mlhs_data.shape_) * mlhs_data);
    ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                    broadcast_scalar(mout_grad, mlhs_data.shape_) * mrhs_data);
  } else {
    LOG(FATAL) << "not reached";
  }
}

inline TShape DotShape(const TShape& lshape,
                       const TShape& rshape,
                       const EnvArguments& env) {
  if (lshape.ndim() == 2 && rshape.ndim() == 2) {
    CHECK_EQ(lshape[1], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    size_t target_shape[] = {lshape[0], rshape[1]};
    return TShape(target_shape, target_shape + 2);
  } else if (lshape.ndim() == 1 && rshape.ndim() == 1) {
    CHECK_EQ(lshape[0], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    size_t target_shape[] = {1};
    return TShape(target_shape, target_shape + 1);
  } else {
    LOG(FATAL) << "dot currently only support 2D 2D array or 1D 1D array"
               << lshape << " v.s. " << rshape;
    return TShape();
  }
}

template<typename xpu>
void BatchDotForward_(const TBlob& lhs,
                        const TBlob& rhs,
                        const EnvArguments& env,
                        TBlob *ret,
                        OpReqType req,
                        RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, mshadow::kFloat32)
      << "dot only support 32 bit float so far";

  if (lhs.shape_.ndim() == 3 && rhs.shape_.ndim() == 3) {
    mshadow::Tensor<xpu, 3, real_t> out = ret->get<xpu, 3, real_t>(s);
    ASSIGN_DISPATCH(out, req, (batch_dot<false, false>(lhs.get<xpu, 3, real_t>(s),
                                                       rhs.get<xpu, 3, real_t>(s))));
  } else {
    LOG(FATAL) << "not reached";
  }
}

template<typename xpu>
void BatchDotBackward_(const OutputGrad& out_grad,
                         const Input0& lhs,
                         const Input1& rhs,
                         const EnvArguments& env,
                         TBlob* lhs_grad,
                         TBlob* rhs_grad,
                         OpReqType req_lhs_grad,
                         OpReqType req_rhs_grad,
                         RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req_rhs_grad, kWriteInplace);
  CHECK_NE(req_lhs_grad, kWriteInplace);

  if (lhs.data.shape_.ndim() == 3 && rhs.data.shape_.ndim() == 3) {
    mshadow::Tensor<xpu, 3, real_t> mout_grad = out_grad.data.get<xpu, 3, real_t>(s);
    mshadow::Tensor<xpu, 3, real_t> mlhs_data = lhs.data.get<xpu, 3, real_t>(s);
    mshadow::Tensor<xpu, 3, real_t> mrhs_data = rhs.data.get<xpu, 3, real_t>(s);
    mshadow::Tensor<xpu, 3, real_t> mlhs_grad = lhs_grad->get<xpu, 3, real_t>(s);
    mshadow::Tensor<xpu, 3, real_t> mrhs_grad = rhs_grad->get<xpu, 3, real_t>(s);
    ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, (batch_dot<true, false>(mlhs_data, mout_grad)));
    ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, (batch_dot<false, true>(mout_grad, mrhs_data)));
  } else {
    LOG(FATAL) << "not reached";
  }
}

inline TShape BatchDotShape(const TShape& lshape,
                              const TShape& rshape,
                              const EnvArguments& env) {
  if (lshape.ndim() == 3 && rshape.ndim() == 3) {
    CHECK(lshape[0] == rshape[0] && lshape[2] == rshape[1])
      << "batch_dot shape error: " << lshape << " X " << rshape;
    size_t target_shape[] = {lshape[0], lshape[1], rshape[2]};
    return TShape(target_shape, target_shape + 3);
  } else {
    LOG(FATAL) << "batch_dot currently only support 3D dot 3D array"
               << lshape << " v.s. " << rshape;
    return TShape();
  }
}


struct SimpleCropParam : public dmlc::Parameter<SimpleCropParam> {
  TShape begin, end;
  DMLC_DECLARE_PARAMETER(SimpleCropParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting coordinates");
    DMLC_DECLARE_FIELD(end)
    .describe("ending coordinates");
  }
};

// matrix crop for multi dimensional cropping: see also slice
template<typename xpu>
void Crop(const TBlob &src,
          const EnvArguments& env,
          TBlob *ret,
          OpReqType req,
          RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  SimpleCropParam param;
  param.Init(env.kwargs);
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    switch (src.shape_.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = src.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = ret->get<xpu, 1, DType>(s);
      out = slice(in, param.begin.get<1>(), param.end.get<1>());
      break;
     }
     case 2: {
      Tensor<xpu, 2, DType> in = src.get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = ret->get<xpu, 2, DType>(s);
      out = slice(in, param.begin.get<2>(), param.end.get<2>());
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = ret->get<xpu, 3, DType>(s);
      out = slice(in, param.begin.get<3>(), param.end.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = ret->get<xpu, 4, DType>(s);
      out = slice(in, param.begin.get<4>(), param.end.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = ret->get<xpu, 5, DType>(s);
      out = slice(in, param.begin.get<5>(), param.end.get<5>());
      break;
     }
     default:
      LOG(FATAL) << "crop supports at most 5 dimensions";
      break;
    }
  });
}

inline TShape CropShape(const TShape& shp,
                        const EnvArguments& env) {
  SimpleCropParam param;
  param.Init(env.kwargs);
  CHECK_EQ(shp.ndim(), param.begin.ndim());
  CHECK_EQ(shp.ndim(), param.end.ndim());
  TShape ret(shp.ndim());
  for (index_t i = 0; i < shp.ndim(); ++i) {
    CHECK(param.begin[i] <= shp[i] && param.end[i] <= shp[i]);
    ret[i] = param.end[i] - param.begin[i];
  }
  return ret;
}


struct SliceParam : public dmlc::Parameter<SliceParam> {
  int axis;
  int begin;
  int end;
  DMLC_DECLARE_PARAMETER(SliceParam) {
    DMLC_DECLARE_FIELD(axis).set_lower_bound(0)
      .describe("The axis to be sliced");
    DMLC_DECLARE_FIELD(begin).set_lower_bound(0)
      .describe("The beginning index to be sliced");
    DMLC_DECLARE_FIELD(end).set_lower_bound(0)
      .describe("The end index to be sliced");
  }
};

inline TShape SliceShape(const TShape& ishape,
                         const EnvArguments& env) {
  SliceParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < static_cast<int>(ishape.ndim())) <<
    "axis must be smaller than the source ndim! Recieved axis=" <<
      param.axis << ", src_ndim=" << ishape.ndim();
  int axis_size = static_cast<int>(ishape[param.axis]);
  CHECK_LE(param.end, axis_size);
  CHECK_LT(param.begin, param.end);

  std::vector<mshadow::index_t> shape;
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (static_cast<int>(i) == param.axis) {
      shape.push_back(static_cast<index_t>(param.end - param.begin));
    } else {
      shape.push_back(ishape[i]);
    }
  }
  return TShape(shape.begin(), shape.end());
}


template<typename xpu>
void Slice(const TBlob &src,
           const EnvArguments& env,
           TBlob *ret,
           OpReqType req,
           RunContext ctx) {
  using namespace mshadow::expr;
  SliceParam param;
  param.Init(env.kwargs);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int ndim = static_cast<int>(ret->shape_.ndim());

  if (param.axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> in =
            src.FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> out =
            ret->FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(out, req, slice<1>(in, param.begin, param.end));
      });
  } else {
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> in =
            src.FlatTo3D<xpu, DType>(param.axis, s);
        mshadow::Tensor<xpu, 3, DType> out =
            ret->FlatTo3D<xpu, DType>(param.axis, s);
        ASSIGN_DISPATCH(out, req, slice<1>(in, param.begin, param.end));
      });
  }
}

// Backward pass of broadcast over the given axis
template<typename xpu>
void SliceGrad_(const OutputGrad& out_grad,
                const EnvArguments& env,
                TBlob *in_grad,
                OpReqType req,
                RunContext ctx) {
  using namespace mshadow::op;
  using namespace mshadow::expr;
  SliceParam param;
  param.Init(env.kwargs);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int ndim = static_cast<int>(in_grad->shape_.ndim());

  if (param.axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> ograd =
            out_grad.data.FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> igrad =
            in_grad->FlatTo2D<xpu, DType>(s);
        if (req == kAddTo) {
          slice<1>(igrad, param.begin, param.end) += F<identity>(ograd);
        } else if (req == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, param.begin, param.end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req, kNullOp);
        }
      });
  } else {
    MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> ograd =
            out_grad.data.FlatTo3D<xpu, DType>(param.axis, s);
        mshadow::Tensor<xpu, 3, DType> igrad =
            in_grad->FlatTo3D<xpu, DType>(param.axis, s);
        if (req == kAddTo) {
          slice<1>(igrad, param.begin, param.end) += F<identity>(ograd);
        } else if (req == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, param.begin, param.end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req, kNullOp);
        }
      });
  }
}

struct FlipParam : public dmlc::Parameter<FlipParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(FlipParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("The dimension to flip");
  }
};

// matrix crop
template<typename xpu>
void Flip(const TBlob &src,
          const EnvArguments& env,
          TBlob *ret,
          OpReqType req,
          RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  FlipParam param;
  param.Init(env.kwargs);
  CHECK_EQ(src.type_flag_, ret->type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    switch (src.shape_.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = src.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = ret->get<xpu, 1, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 2: {
      Tensor<xpu, 2, DType> in = src.get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = ret->get<xpu, 2, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = ret->get<xpu, 3, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = ret->get<xpu, 4, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = ret->get<xpu, 5, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     default:
      LOG(FATAL) << "flip supports at most 5 dimensions";
      break;
    }
  });
}

inline TShape FlipShape(const TShape& shp,
                        const EnvArguments& env) {
  FlipParam param;
  param.Init(env.kwargs);
  CHECK(param.axis < static_cast<int>(shp.ndim()) && param.axis >= static_cast<int>(0));
  return shp;
}


// transpose
MXNET_REGISTER_SIMPLE_OP(transpose, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, Transpose<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(TransposeShape)
.set_gradient(XPU::kDevMask, TransposeGrad<XPU>, kNoInplace)
.describe("Transpose the input matrix and return a new one")
.add_arguments(TransposeParam::__FIELDS__());

// expand_dim
MXNET_REGISTER_SIMPLE_OP(expand_dims, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, ReshapeImpl<XPU>, kInplaceInOut)
.set_shape_function(ExpandDimShape)
.set_gradient(XPU::kDevMask, ReshapeGrad_<XPU>, kInplaceOutIn)
.describe("Expand the shape of array by inserting a new axis.")
.add_arguments(ExpandDimParam::__FIELDS__());

// crop
MXNET_REGISTER_SIMPLE_OP(crop, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, Crop<XPU>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(CropShape)
.describe("Crop the input matrix and return a new one")
.add_arguments(SimpleCropParam::__FIELDS__());

// slice_axis
MXNET_REGISTER_SIMPLE_OP(slice_axis, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, Slice<XPU>,
              kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, SliceGrad_<XPU>, kNoInplace)
.set_shape_function(SliceShape)
.describe("Slice the input along certain axis and return a sliced array.")
.add_arguments(SliceParam::__FIELDS__());

// flip
MXNET_REGISTER_SIMPLE_OP(flip, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, Flip<XPU>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(FlipShape)
.describe("Flip the input matrix along axis and return a new one")
.add_arguments(FlipParam::__FIELDS__());

// dot
MXNET_REGISTER_SIMPLE_OP(dot, XPU)
.set_function(XPU::kDevMask, DotForward_<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(DotShape)
.set_gradient(XPU::kDevMask, DotBackward_<XPU>, kNoInplace)
.describe("Calculate dot product of two matrices or two vectors");

// batched_dot
MXNET_REGISTER_SIMPLE_OP(batch_dot, XPU)
.set_function(XPU::kDevMask, BatchDotForward_<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(BatchDotShape)
.set_gradient(XPU::kDevMask, BatchDotBackward_<XPU>, kNoInplace)
.describe("Calculate batched dot product of two matrices."
          " (batch, M, K) batch_dot (batch, K, N) --> (batch, M, N)");
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MATRIX_OP_INL_H_
