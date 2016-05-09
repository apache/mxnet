/*!
 * Copyright (c) 2015 by Contributors
 * \file make_loss.cc
 * \brief special layer for propagating loss
*/
#include "./make_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MakeLossParam param) {
  return new MakeLossOp<cpu>(param);
}

Operator *MakeLossProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MakeLossParam);

MXNET_REGISTER_OP_PROPERTY(MakeLoss, MakeLossProp)
.describe("Get output from a symbol and pass 1 gradient back. "
"This is used as a terminal loss if unary and binary operator "
"are used to composite a loss with no declaration of backward "
"dependency")
.add_argument("data", "Symbol", "Input data.")
.add_arguments(MakeLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
