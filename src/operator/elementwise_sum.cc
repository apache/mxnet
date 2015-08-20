/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_sum.cc
 * \brief elementwise sum operator
*/
#include <mxnet/registry.h>
#include "./elementwise_sum-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ElementWiseSumParam param) {
  return new ElementWiseSumOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* ElementWiseSumProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

REGISTER_OP_PROPERTY(ElementWiseSum, ElementWiseSumProp);
}  // namespace op
}  // namespace mxnet
