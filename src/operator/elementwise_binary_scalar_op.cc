/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op.cc
 * \brief elementwise binary operator
*/
#include "./elementwise_binary_scalar_op-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateElementwiseBinaryScalarOp<cpu>(elembinary::ElementwiseBinaryScalarOpType type,
                                               ScalarOpParam param) {
    return CreateElementwiseBinaryScalarOp_<cpu>(type, param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
template<typename ForwardOp>
Operator* ElementwiseBinaryScalarOpProp<ForwardOp>::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateElementwiseBinaryScalarOp, GetOpType<ForwardOp>(), param_);
}

DMLC_REGISTER_PARAMETER(ScalarOpParam);

MXNET_REGISTER_OP_PROPERTY(_PlusScalar, ElementwiseBinaryScalarOpProp<mshadow::op::plus>)
.describe("Perform an elementwise plus.")
.add_argument("array", "Symbol", "Input array operand to the operation.")
.add_arguments(ScalarOpParam::__FIELDS__());
MXNET_REGISTER_OP_PROPERTY(_MinusScalar, ElementwiseBinaryScalarOpProp<mshadow::op::minus>)
.describe("Perform an elementwise minus.")
.add_argument("array", "Symbol", "Input array operand to the operation.")
.add_arguments(ScalarOpParam::__FIELDS__());
MXNET_REGISTER_OP_PROPERTY(_MulScalar, ElementwiseBinaryScalarOpProp<mshadow::op::mul>)
.describe("Perform an elementwise mul.")
.add_argument("array", "Symbol", "Input array operand to the operation.")
.add_arguments(ScalarOpParam::__FIELDS__());
MXNET_REGISTER_OP_PROPERTY(_DivScalar, ElementwiseBinaryScalarOpProp<mshadow::op::div>)
.describe("Perform an elementwise div.")
.add_argument("array", "Symbol", "Input array operand to the operation.")
.add_arguments(ScalarOpParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(_PowerScalar, ElementwiseBinaryScalarOpProp<mshadow_op::power>)
.describe("Perform an elementwise power.")
.add_argument("array", "Symbol", "Input array operand to the operation.")
.add_arguments(ScalarOpParam::__FIELDS__());
MXNET_REGISTER_OP_PROPERTY(_MaximumScalar, ElementwiseBinaryScalarOpProp<mshadow_op::maximum>)
.describe("Perform an elementwise maximum.")
.add_argument("array", "Symbol", "Input array operand to the operation.")
.add_arguments(ScalarOpParam::__FIELDS__());
MXNET_REGISTER_OP_PROPERTY(_MinimumScalar, ElementwiseBinaryScalarOpProp<mshadow_op::minimum>)
.describe("Perform an elementwise minimum.")
.add_argument("array", "Symbol", "Input array operand to the operation.")
.add_arguments(ScalarOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
