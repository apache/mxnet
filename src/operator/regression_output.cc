/*!
 * Copyright (c) 2015 by Contributors
 * \file regression_output.cc
 * \brief regression output operator
*/
#include "./regression_output-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateRegressionOutputOp<cpu>(RegressionOutputType type) {
  switch (type) {
    case kLinear:
      return new RegressionOutputOp<cpu, mshadow::op::identity, mshadow::op::minus>();
    case kLogistic:
      return new RegressionOutputOp<cpu, mshadow_op::sigmoid, mshadow::op::minus>();
    default:
      LOG(FATAL) << "unknown activation type " << type;
  }
  return nullptr;
}

// DO_BIND_DISPATCH comes from operator_common.h
template<RegressionOutputType type>
Operator *RegressionOutputProp<type>::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateRegressionOutputOp, type);
}

MXNET_REGISTER_OP_PROPERTY(LinearRegressionOutput, RegressionOutputProp<kLinear>)
.describe("Use linear regression for final output, this is used on final output of a net.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.");

MXNET_REGISTER_OP_PROPERTY(LogisticRegressionOutput, RegressionOutputProp<kLogistic>)
.describe("Use Logistic regression for final output, this is used on final output of a net.\n"
          "Logistic regression is suitable for binary classification "
          "or probability prediction tasks.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.");
}  // namespace op
}  // namespace mxnet
