/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_binary_scalar_op-inl.h
 * \brief Function defintion of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_SCALAR_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_SCALAR_OP_INL_H_

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
void BinaryScalarLForward_(const TBlob& lhs,
                           const EnvArguments& env,
                           TBlob *ret,
                           OpReqType req,
                           RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<OP>(lhs.FlatTo2D<xpu, DType>(s),
                          scalar<DType>(env.scalar)));
  });
}

template<typename xpu, typename OP>
void BinaryScalarRForward_(const TBlob& rhs,
                           const EnvArguments& env,
                           TBlob *ret,
                           OpReqType req,
                           RunContext ctx) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<OP>(scalar<DType>(env.scalar),
                          rhs.FlatTo2D<xpu, DType>(s)));
  });
}

template<typename xpu, typename BackwardOp>
void BinaryScalarBackwardT0_(const OutputGrad& out_grad,
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
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    F<BackwardOp>(out_grad.data.FlatTo2D<xpu, DType>()));
    });
}

template<typename xpu, typename BackwardOp>
void BinaryScalarBackwardT1_(const OutputGrad& out_grad,
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
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    F<BackwardOp>(out_grad.data.FlatTo2D<xpu, DType>(),
                                  scalar<DType>(env.scalar)));
  });
}

template<typename xpu, typename BackwardOp>
void BinaryScalarBackwardT2_(const OutputGrad& out_grad,
                             const Input0& lhs,
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
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (F<BackwardOp>(lhs.data.FlatTo2D<xpu, DType>(),
                                   scalar<DType>(env.scalar)) *
                     out_grad.data.FlatTo2D<xpu, DType>()));
    });
}

template<typename xpu>
void DivRBackward_(const OutputGrad& out_grad,
                   const Input0& in_data,
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
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (scalar<DType>(-env.scalar) /
                     F<mshadow_op::square>(in_data.data.FlatTo2D<xpu, DType>()) *
                     out_grad.data.FlatTo2D<xpu, DType>()));
  });
}


template<typename xpu>
void PowerLBackward_(const OutputGrad& out_grad,
                     const Input0& lhs,
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
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (F<mshadow_op::power>(lhs.data.FlatTo2D<xpu, DType>(),
                                          scalar<DType>(env.scalar - 1.0f)) *
                     scalar<DType>(env.scalar) *
                     out_grad.data.FlatTo2D<xpu, DType>()));
  });
}

template<typename xpu>
void PowerRBackward_(const OutputGrad& out_grad,
                     const OutputValue& out_data,
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
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->FlatTo2D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    (scalar<DType>(logf(env.scalar)) *
                     out_data.data.FlatTo2D<xpu, DType>() *
                     out_grad.data.FlatTo2D<xpu, DType>()));
  });
}


MXNET_REGISTER_SIMPLE_OP(_plus_scalar, XPU)
.set_symbol_op_name("_PlusScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarLForward_<XPU, mshadow::op::plus>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              BinaryScalarBackwardT0_<XPU, mshadow_op::identity>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_minus_scalar, XPU)
.set_symbol_op_name("_MinusScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarLForward_<XPU, mshadow::op::minus>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              BinaryScalarBackwardT0_<XPU, mshadow_op::identity>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_rminus_scalar, XPU)
.set_symbol_op_name("_RMinusScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarRForward_<XPU, mshadow::op::minus>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              BinaryScalarBackwardT0_<XPU, mshadow_op::negation>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_mul_scalar, XPU)
.set_symbol_op_name("_MulScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarLForward_<XPU, mshadow::op::mul>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              BinaryScalarBackwardT1_<XPU, mshadow::op::mul>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_div_scalar, XPU)
.set_symbol_op_name("_DivScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarLForward_<XPU, mshadow::op::div>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              BinaryScalarBackwardT1_<XPU, mshadow::op::div>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_rdiv_scalar, XPU)
.set_symbol_op_name("_RDivScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarRForward_<XPU, mshadow::op::div>, kInplaceInOut)
.set_gradient(XPU::kDevMask, DivRBackward_<XPU>, kInplaceOutIn);


MXNET_REGISTER_SIMPLE_OP(_maximum_scalar, XPU)
.set_symbol_op_name("_MaximumScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarLForward_<XPU, mshadow_op::maximum>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              BinaryScalarBackwardT2_<XPU, mshadow_op::maximum_grad>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_minimum_scalar, XPU)
.set_symbol_op_name("_MinimumScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarLForward_<XPU, mshadow_op::minimum>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              BinaryScalarBackwardT2_<XPU, mshadow_op::minimum_grad>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_power_scalar, XPU)
.set_symbol_op_name("_PowerScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarLForward_<XPU, mshadow_op::power>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              PowerLBackward_<XPU>, kInplaceOutIn);

MXNET_REGISTER_SIMPLE_OP(_rpower_scalar, XPU)
.set_symbol_op_name("_RPowerScalar")
.set_enable_scalar(true, kArrayBeforeScalar)
.set_function(XPU::kDevMask,
              BinaryScalarRForward_<XPU, mshadow_op::power>, kInplaceInOut)
.set_gradient(XPU::kDevMask,
              PowerRBackward_<XPU>, kInplaceOutIn);
}  // namespace op
}  // namespace mxnet
#endif   // MXNET_OPERATOR_ELEMENTWISE_BINARY_SCALAR_OP_INL_H_
