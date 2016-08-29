/*!
 * Copyright (c) 2015 by Contributors
 * \file anchorclassifycost.cc
 * \brief
 * \author Ming Zhang
*/

#include "./anchorclassifycost-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(AnchorClsCostParam param) {
  return new AnchorClsCostOp<cpu>(param);
}

Operator* AnchorClsCostProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}


DMLC_REGISTER_PARAMETER(AnchorClsCostParam);

MXNET_REGISTER_OP_PROPERTY(AnchorClsCost, AnchorClsCostProp)
.add_argument("data", "Symbol", "Input data to the AnchorClsCostOp.")
.add_argument("label", "Symbol", "Label data to the AnchorClsCostOp.")
.add_argument("marklabel", "Symbol", "MarkLabel data to the AnchorClsCostOp.")
.add_arguments(AnchorClsCostParam::__FIELDS__())
.describe("Apply AnchorClsCost to input.");

}  // namespace op
}  // namespace mxnet
