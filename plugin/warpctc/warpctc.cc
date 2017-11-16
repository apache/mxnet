/*!
 * Copyright (c) 2015 by Contributors
 * \file warpctc.cc
 * \brief warpctc op
 * \author Liang Xiang
*/

#include "./warpctc-inl.h"
#include "../../src/operator/mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(WarpCTCParam param) {
  return new WarpCTCOp<cpu>(param);
}

Operator *WarpCTCProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(WarpCTCParam);

MXNET_REGISTER_OP_PROPERTY(WarpCTC, WarpCTCProp)
.describe("warp ctc.")
.add_arguments(WarpCTCParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
