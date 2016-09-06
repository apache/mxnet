/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op-inl.h
 * \brief Function defintion of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_

#include <mxnet/operator_util.h>
#include "./mshadow_op.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {

template<typename xpu, typename OP>
void BinaryForward_(const TBlob& lhs,
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
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> out = ret->FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<OP>(lhs.FlatTo1D<xpu, DType>(s),
                          rhs.FlatTo1D<xpu, DType>(s)));
  });
}


template<typename xpu>
void PlusBackward_(const OutputGrad& out_grad,
                   const EnvArguments& env,
                   TBlob* lhs_grad,
                   TBlob* rhs_grad,
                   OpReqType req_lhs_grad,
                   OpReqType req_rhs_grad,
                   RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<mshadow_op::identity>(mout_grad));
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<mshadow_op::identity>(mout_grad));
    });
}

template<typename xpu>
void MinusBackward_(const OutputGrad& out_grad,
                    const EnvArguments& env,
                    TBlob* lhs_grad,
                    TBlob* rhs_grad,
                    OpReqType req_lhs_grad,
                    OpReqType req_rhs_grad,
                    RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<mshadow_op::identity>(mout_grad));
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<mshadow_op::negation>(mout_grad));
    });
}

template<typename xpu>
void MulBackward_(const OutputGrad& out_grad,
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
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, mlhs_data * mout_grad);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mrhs_data * mout_grad);
    });
}

template<typename xpu>
void DivBackward_(const OutputGrad& out_grad,
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
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      F<mshadow_op::negation>(mout_grad * mlhs_data)/
                      F<mshadow_op::square>(mrhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mout_grad /  mrhs_data);
    });
}

template<typename xpu>
void PowerBackward_(const OutputGrad& out_grad,
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
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      F<mshadow_op::log>(mlhs_data) *
                      F<mshadow_op::power>(mlhs_data, mrhs_data) * mout_grad);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                      mrhs_data *
                      F<mshadow_op::power>(mlhs_data, mrhs_data - scalar<DType>(1)) *
                      mout_grad);
    });
}

template<typename xpu>
void MaximumBackward_(const OutputGrad& out_grad,
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
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      mout_grad * F<mshadow_op::maximum_grad>(mrhs_data, mlhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                      mout_grad * F<mshadow_op::maximum_grad>(mlhs_data, mrhs_data));
    });
}

template<typename xpu>
void MinimumBackward_(const OutputGrad& out_grad,
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
  MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 1, DType> mout_grad = out_grad.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_data = lhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_data = rhs.data.FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad = lhs_grad->FlatTo1D<xpu, DType>(s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad = rhs_grad->FlatTo1D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
                      mout_grad * F<mshadow_op::minimum_grad>(mrhs_data, mlhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
                      mout_grad * F<mshadow_op::minimum_grad>(mlhs_data, mrhs_data));
    });
}


MXNET_REGISTER_SIMPLE_OP(_plus, XPU)
.set_symbol_op_name("_Plus")
.set_function(XPU::kDevMask, BinaryForward_<XPU, mshadow::op::plus>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, PlusBackward_<XPU>, kInplaceOutLhs)
.describe("Add lhs and rhs");

MXNET_REGISTER_SIMPLE_OP(_minus, XPU)
.set_symbol_op_name("_Minus")
.set_function(XPU::kDevMask, BinaryForward_<XPU, mshadow::op::minus>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, MinusBackward_<XPU>, kInplaceOutLhs)
.describe("Minus lhs and rhs");

MXNET_REGISTER_SIMPLE_OP(_mul, XPU)
.set_symbol_op_name("_Mul")
.set_function(XPU::kDevMask, BinaryForward_<XPU, mshadow::op::mul>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, MulBackward_<XPU>, kInplaceOutLhs)
.describe("Multiply lhs and rhs");

MXNET_REGISTER_SIMPLE_OP(_div, XPU)
.set_symbol_op_name("_Div")
.set_function(XPU::kDevMask, BinaryForward_<XPU, mshadow::op::div>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, DivBackward_<XPU>, kInplaceOutLhs)
.describe("Multiply lhs by rhs");

MXNET_REGISTER_SIMPLE_OP(_power, XPU)
.set_symbol_op_name("_Power")
.set_function(XPU::kDevMask, BinaryForward_<XPU, mshadow_op::power>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, PowerBackward_<XPU>, kInplaceOutLhs)
.describe("Elementwise power(lhs, rhs)");

MXNET_REGISTER_SIMPLE_OP(_maximum, XPU)
.set_symbol_op_name("_Maximum")
.set_function(XPU::kDevMask, BinaryForward_<XPU, mshadow_op::maximum>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, MaximumBackward_<XPU>, kInplaceOutLhs)
.describe("Elementwise max of lhs by rhs");

MXNET_REGISTER_SIMPLE_OP(_minimum, XPU)
.set_symbol_op_name("_Minimum")
.set_function(XPU::kDevMask, BinaryForward_<XPU, mshadow_op::minimum>, kInplaceLhsOut)
.set_gradient(XPU::kDevMask, MinimumBackward_<XPU>, kInplaceOutLhs)
.describe("Elementwise min of lhs by rhs");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_OP_INL_H_
