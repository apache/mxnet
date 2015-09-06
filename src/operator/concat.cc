/*!
 * Copyright (c) 2015 by Contributors
 * \file concat.cc
 * \brief
 * \author Bing Xu
*/

#include "./concat-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ConcatParam param) {
  return new ConcatOp<cpu>(param);
}

Operator* ConcatProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ConcatParam);

MXNET_REGISTER_OP_PROPERTY(Concat, ConcatProp)
.describe("Perform an feature concat over all the inputs.")
.add_arguments(ConcatParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

