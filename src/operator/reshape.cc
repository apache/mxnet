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
Operator *CreateOp<cpu>() {
  return new ReshapeOp<cpu>();
}

Operator* ReshapeProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp);
}

DMLC_REGISTER_PARAMETER(ReshapeParam);

MXNET_REGISTER_OP_PROPERTY(Reshape, ReshapeProp)
.add_argument("data", "Symbol", "Input data to  flatten.")
.describe("Reshape input to target shape");

MXNET_REGISTER_OP_PROPERTY(Flatten, FlattenProp)
.add_argument("data", "Symbol", "Input data to  flatten.")
.describe("Flatten input");
}  // namespace op
}  // namespace mxnet
