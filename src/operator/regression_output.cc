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
Operator *CreateRegressionOutputOp<cpu>(reg_enum::RegressionOutputType type,
                                        RegressionOutputParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (type) {
      case reg_enum::kLinear:
        op = new RegressionOutputOp
          <cpu, mshadow::op::identity, mshadow::op::minus, DType>(param);
        break;
      case reg_enum::kLogistic:
        op = new RegressionOutputOp
          <cpu, mshadow_op::sigmoid, mshadow::op::minus, DType>(param);
        break;
      case reg_enum::kMAE:
        op = new RegressionOutputOp
          <cpu, mshadow::op::identity, mshadow_op::minus_sign, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown RegressionOutput type " << type;
    }
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
template<reg_enum::RegressionOutputType type>
Operator *RegressionOutputProp<type>::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateRegressionOutputOp, type, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(RegressionOutputParam);

MXNET_REGISTER_OP_PROPERTY(LinearRegressionOutput, RegressionOutputProp<reg_enum::kLinear>)
.describe("Use linear regression for final output, this is used on final output of a net.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(MAERegressionOutput, RegressionOutputProp<reg_enum::kMAE>)
.describe("Use mean absolute error regression for final output, "
          "this is used on final output of a net.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(LogisticRegressionOutput, RegressionOutputProp<reg_enum::kLogistic>)
.describe("Use Logistic regression for final output, this is used on final output of a net.\n"
          "Logistic regression is suitable for binary classification "
          "or probability prediction tasks.")
.add_argument("data", "Symbol", "Input data to function.")
.add_argument("label", "Symbol", "Input label to function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
