/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

#include "./flatten-inl.h"


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>() {
  return new FlattenOp<cpu>();
}

Operator* FlattenProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp);
}

MXNET_REGISTER_OP_PROPERTY(Flatten, FlattenProp)
.add_argument("data", "Symbol", "Input data to  flatten.")
.describe("Flatten 4D input to form batch-1-1-feature format");

}  // namespace op
}  // namespace mxnet
