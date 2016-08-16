/*!
 * Copyright (c) 2015 by Contributors
 * \file regression_output.cu
 * \brief regression output operator
*/
#include "./regression_output-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateRegressionOutputOp<gpu>(reg_enum::RegressionOutputType type,
                                        RegressionOutputParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (type) {
      case reg_enum::kLinear:
        op = new RegressionOutputOp
          <gpu, mshadow::op::identity, mshadow::op::minus, DType>(param);
        break;
      case reg_enum::kLogistic:
        op = new RegressionOutputOp
          <gpu, mshadow_op::sigmoid, mshadow::op::minus, DType>(param);
        break;
      case reg_enum::kMAE:
        op = new RegressionOutputOp
          <gpu, mshadow::op::identity, mshadow_op::minus_sign, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown RegressionOutput type " << type;
    }
  });
  return op;
}
}  // namespace op
}  // namespace mxnet

