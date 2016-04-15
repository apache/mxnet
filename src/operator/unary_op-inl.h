/*!
 * Copyright (c) 2015 by Contributors
 * \file unary_op-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_UNARY_OP_INL_H_
#define MXNET_OPERATOR_UNARY_OP_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
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
      << "transpose only accept two dimensional input";
  std::vector<mshadow::index_t> ret;
  ret.push_back(shp[1]);
  ret.push_back(shp[0]);
  return TShape(ret.begin(), ret.end());
}

// -------------- operator

struct PadParam : public dmlc::Parameter<PadParam> {
  TShape pad_shape;
  DMLC_DECLARE_PARAMETER(PadParam) {
    int shape[] = {0, 0};
    DMLC_DECLARE_FIELD(pad_shape).set_default(TShape(shape, shape + 2))
    .describe("pad size: (y, x)");
  }
};

template<typename xpu>
void Pad(const TBlob &src,
         const EnvArguments &env,
         TBlob *ret,
         OpReqType req,
         RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  PadParam param;
  param.Init(env.kwargs);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4> in = src.get<xpu, 4, real_t>(s);
  Tensor<xpu, 4> out = ret->get<xpu, 4, real_t>(s);
  out = pad(in, param.pad_shape[0], param.pad_shape[1]);
}

template<typename xpu>
void PadGrad(const OutputGrad& src,
             const EnvArguments& env,
             TBlob *ret,
             OpReqType req,
             RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  PadParam param;
  param.Init(env.kwargs);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 4, real_t> grad = src.data.get<xpu, 4, real_t>(s);
  Tensor<xpu, 4, real_t> out = ret->get<xpu, 4, real_t>(s);
  out = crop(grad, Shape2(out.size(2), out.size(3)),
             param.pad_shape[0], param.pad_shape[1]);
}


inline TShape PadShape(const TShape& shp,
                       const EnvArguments& env) {
  PadParam param;
  param.Init(env.kwargs);
  std::vector<mshadow::index_t> ret;
  CHECK_GE(shp.ndim(), 2);
  mshadow::index_t srcdim = shp.ndim();
  for (index_t i = 0; i < srcdim; ++i) {
    ret.push_back(shp[i]);
  }
  ret[srcdim - 2] += 2 * param.pad_shape[0];
  ret[srcdim - 1] += 2 * param.pad_shape[1];
  return TShape(ret.begin(), ret.end());
}



// transpose
MXNET_REGISTER_SIMPLE_OP(transpose, XPU)
.set_function(XPU::kDevMask, Transpose<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(TransposeShape)
.set_gradient(XPU::kDevMask, TransposeGrad<XPU>, kNoInplace)
.describe("Transpose the input matrix and return a new one");

// pad
MXNET_REGISTER_SIMPLE_OP(pad, XPU)
.set_function(XPU::kDevMask, Pad<XPU>, kNoInplace, kRegisterSymbolic)
.set_shape_function(PadShape)
.set_gradient(XPU::kDevMask, PadGrad<XPU>, kNoInplace)
.set_enable_kwargs(true)
.add_arguments(PadParam::__FIELDS__())
.describe("Pad 4d input and return a new one");

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_UNARY_OP_INL_H_
