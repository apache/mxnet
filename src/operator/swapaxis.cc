/*!
 * Copyright (c) 2015 by Contributors
 * \file swapaxis.cc
 * \brief
 * \author Ming Zhang
*/

#include "./swapaxis-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(SwapAxisParam param) {
  return new SwapAxisOp<cpu>(param);
}

Operator* SwapAxisProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}


DMLC_REGISTER_PARAMETER(SwapAxisParam);

MXNET_REGISTER_OP_PROPERTY(SwapAxis, SwapAxisProp)
.add_argument("data", "Symbol", "Input data to the SwapAxisOp.")
.add_arguments(SwapAxisParam::__FIELDS__())
.describe("Apply swapaxis to input.");
}  // namespace op
}  // namespace mxnet
