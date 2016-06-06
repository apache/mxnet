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


struct SimpleCropParam : public dmlc::Parameter<SimpleCropParam> {
  TShape begin, end;
  DMLC_DECLARE_PARAMETER(SimpleCropParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting coordinates");
    DMLC_DECLARE_FIELD(end)
    .describe("ending coordinates");
  }
};

// matrix crop
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
.describe("Transpose the input matrix and return a new one");

// crop
MXNET_REGISTER_SIMPLE_OP(crop, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, Crop<XPU>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(CropShape)
.describe("Crop the input matrix and return a new one");

// flip
MXNET_REGISTER_SIMPLE_OP(flip, XPU)
.set_enable_kwargs(true)
.set_function(XPU::kDevMask, Flip<XPU>, kNoInplace, kNotRegisterSymbolic)
.set_shape_function(FlipShape)
.describe("Flip the input matrix along axis and return a new one");


// dot
MXNET_REGISTER_SIMPLE_OP(dot, XPU)
.set_function(XPU::kDevMask, DotForward_<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(DotShape)
.set_gradient(XPU::kDevMask, DotBackward_<XPU>, kNoInplace)
.describe("Calculate dot product of two matrices or two vectors");
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MATRIX_OP_INL_H_
