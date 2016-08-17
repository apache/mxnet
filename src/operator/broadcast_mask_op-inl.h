/*!
 * Copyright (c) 2015 by Contributors
 * \file broadcast_mask_op-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_BROADCAST_MASK_OP_INL_H_
#define MXNET_OPERATOR_BROADCAST_MASK_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./operator_common.h"


#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

inline TShape ElementwiseMaskShape_(const TShape& lhs,
                                    const TShape& rhs,
                                    const EnvArguments& env) {
  CHECK(lhs.ndim() > 1 && rhs.ndim() == 1) <<
    "source tensor should be 2D or more and mask should be 1D";
  CHECK_EQ(lhs[0], rhs[0]) << "The first dimention of inputs should be same";
  return TShape(lhs);
}

template<typename xpu>
void ElementwiseMaskForward_(const TBlob& lhs,
                             const TBlob& rhs,
                             const EnvArguments& env,
                             TBlob *ret,
                             OpReqType req,
                             RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK(lhs.shape_.ndim() > 1 && rhs.shape_.ndim() == 1 &&
        lhs.shape_[0] == rhs.shape_[0]) <<
    "the first ndim of lhs and rhs must be equal, lhs should be 2D or more and rhs shoube be 1D"
    " shape of lhs=" << lhs.shape_ << " shape of rhs=" << rhs.shape_;
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
      // TODO(bing): swap because requirement of inplace, change mshadow later
      mask(rhs.get<xpu, 1, DType>(s), lhs.FlatTo2D<xpu, DType>(s)));
  });
  return;
}

template<typename xpu>
void ElementwiseMaskBackward_(const OutputGrad& out_grad,
                              const Input0& lhs,
                              const Input1& rhs,
                              const EnvArguments& env,
                              TBlob* lhs_grad,
                              TBlob* rhs_grad,
                              OpReqType req_lhs_grad,
                              OpReqType req_rhs_grad,
                              RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
    mshadow::Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
    mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.get<xpu, 1, DType>(s);
    ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
      // TODO(bing): swap because requirement of inplace, change mshadow later
      mask(mrhs_data, mout_grad));
  });
  return;
}


MXNET_REGISTER_SIMPLE_OP(element_mask, XPU)
.set_shape_function(ElementwiseMaskShape_)
.set_function(XPU::kDevMask, ElementwiseMaskForward_<XPU>, kInplaceLhsOut, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, ElementwiseMaskBackward_<XPU>, kInplaceOutLhs)
.describe("rhs elmentwise mask lhs with broadcast");

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_BROADCAST_MASK_OP_INL_H_

