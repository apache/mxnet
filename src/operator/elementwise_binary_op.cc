/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op.cc
 * \brief elementwise binary operator
*/
#include "./elementwise_binary_op-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateElementWiseBinaryOp<cpu>(elembinary::ElementWiseBinaryOpType type) {
  return CreateElementWiseBinaryOp_<cpu>(type);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
template<typename ForwardOp>
Operator* ElementWiseBinaryOpProp<ForwardOp>::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateElementWiseBinaryOp, GetOpType<ForwardOp>());
}

MXNET_REGISTER_OP_PROPERTY(_Plus, ElementWiseBinaryOpProp<mshadow::op::plus>)
.describe("Perform an elementwise plus.");
MXNET_REGISTER_OP_PROPERTY(_Minus, ElementWiseBinaryOpProp<mshadow::op::minus>)
.describe("Perform an elementwise minus.");
MXNET_REGISTER_OP_PROPERTY(_Mul, ElementWiseBinaryOpProp<mshadow::op::mul>)
.describe("Perform an elementwise mul.");
MXNET_REGISTER_OP_PROPERTY(_Div, ElementWiseBinaryOpProp<mshadow::op::div>)
.describe("Perform an elementwise div.");
MXNET_REGISTER_OP_PROPERTY(_Power, ElementWiseBinaryOpProp<mshadow_op::power>)
.describe("Perform an elementwise power.");
MXNET_REGISTER_OP_PROPERTY(_Maximum, ElementWiseBinaryOpProp<mshadow_op::maximum>)
.describe("Perform an elementwise power.");
MXNET_REGISTER_OP_PROPERTY(_Minimum, ElementWiseBinaryOpProp<mshadow_op::minimum>)
.describe("Perform an elementwise power.");

}  // namespace op
}  // namespace mxnet
