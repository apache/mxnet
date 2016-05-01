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
// matrix transpose
template<typename xpu>
void Transpose(const TBlob &src,
               const EnvArguments& env,
               TBlob *ret,
               OpReqType req,
               RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2> out = ret->FlatTo2D<xpu, real_t>(s);
  mshadow::Tensor<xpu, 2> in = src.FlatTo2D<xpu, real_t>(s);
  out = in.T();
}

template<typename xpu>
void TransposeGrad(const OutputGrad& src,
                   const EnvArguments& env,
                   TBlob *ret,
                   OpReqType req,
                   RunContext ctx) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  mshadow::Tensor<xpu, 2> out = ret->FlatTo2D<xpu, real_t>(s);
  mshadow::Tensor<xpu, 2> in = src.data.FlatTo2D<xpu, real_t>(s);
  out = in.T();
}

inline TShape TransposeShape(const TShape& shp,
                             const EnvArguments& env) {
  CHECK(shp.ndim() == 2)
      << "Transpose only accept two dimensional input";
  std::vector<mshadow::index_t> ret;
  ret.push_back(shp[1]);
  ret.push_back(shp[0]);
  return TShape(ret.begin(), ret.end());
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



// transpose
MXNET_REGISTER_SIMPLE_OP(transpose, XPU)
.set_function(XPU::kDevMask, Transpose<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(TransposeShape)
.set_gradient(XPU::kDevMask, TransposeGrad<XPU>, kNoInplace)
.describe("Transpose the input matrix and return a new one");

// dot
MXNET_REGISTER_SIMPLE_OP(dot, XPU)
.set_function(XPU::kDevMask, DotForward_<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(DotShape)
.set_gradient(XPU::kDevMask, DotBackward_<XPU>, kNoInplace)
.describe("Calculate dot product of two matrices or two vectors");
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MATRIX_OP_INL_H_
