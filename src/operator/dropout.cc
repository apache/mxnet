/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu
*/

#include "./dropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DropoutParam param) {
  return new DropoutOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DropoutProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(DropoutParam);

MXNET_REGISTER_OP_PROPERTY(Dropout, DropoutProp)
.describe("Apply dropout to input")
.add_argument("data", "Symbol", "Input data to dropout.")
.add_arguments(DropoutParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


