/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

#include "./reshape-inl.h"


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ReshapeParam param) {
  return new ReshapeOp<cpu>(param);
}

Operator* ReshapeProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ReshapeParam);

MXNET_REGISTER_OP_PROPERTY(Reshape, ReshapeProp)
.describe("Reshape input to target shape")
.add_argument("data", "Symbol", "Input data to reshape.")
.add_arguments(ReshapeParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Flatten, FlattenProp)
.describe("Flatten input")
.add_argument("data", "Symbol", "Input data to flatten.");
}  // namespace op
}  // namespace mxnet
