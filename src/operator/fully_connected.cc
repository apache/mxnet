/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include <mxnet/registry.h>
#include "./fully_connected-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FullyConnectedParam param) {
  return new FullyConnectedOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* FullyConnectedProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(FullyConnectedParam);

REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedProp)
.describe("Apply matrix multiplication to input then add a bias.")
.add_argument("data", "Input data to the FullyConnectedOp.")
.add_argument("weight", "Weight matrix.")
.add_argument("bias", "Bias parameter.")
.set_param_doc(FullyConnectedParam::__DOC__());
}  // namespace op
}  // namespace mxnet
