/*!
 * Copyright (c) 2015 by Contributors
 * \file anchor_multibbs_regressioncost.cc
 * \brief
 * \author Ming Zhang
*/

#include "./anchor_multibbs_regressioncost-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(AnchorMultiBBsRegCostParam param) {
  return new AnchorMultiBBsRegCostOp<cpu>(param);
}

Operator* AnchorMultiBBsRegCostProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}


DMLC_REGISTER_PARAMETER(AnchorMultiBBsRegCostParam);

MXNET_REGISTER_OP_PROPERTY(AnchorMultiBBsRegCost, AnchorMultiBBsRegCostProp)
.add_argument("data", "Symbol", "Input data to the AnchorMultiBBsRegCostOp.")
.add_argument("label", "Symbol", "Label data to the AnchorMultiBBsRegCostOp.")
.add_argument("coordlabel", "Symbol", "CoordLabel data to the AnchorMultiBBsRegCostOp.")
.add_argument("allbbslabel", "Symbol", "All BBsLabel data to the AnchorMultiBBsRegCostOp.")
.add_argument("infolabel", "Symbol", "AnchorInfoLabel data to the AnchorMultiBBsRegCostOp.")
.add_arguments(AnchorMultiBBsRegCostParam::__FIELDS__())
.describe("Apply AnchorMultiBBsRegCost to input.");

}  // namespace op
}  // namespace mxnet
