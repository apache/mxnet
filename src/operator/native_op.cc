/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op.cc
 * \brief
 * \author Junyuan Xie
*/
#include "./native_op-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(NativeOpParam param) {
  return new NativeOp<cpu>(param);
}

Operator* NativeOpProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(NativeOpParam);

MXNET_REGISTER_OP_PROPERTY(_Native, NativeOpProp)
.describe("Stub for implementing an operator implemented in native frontend language.")
.add_arguments(NativeOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
