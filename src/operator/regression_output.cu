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
                                        RegressionOutputParam param) {
  switch (type) {
    case reg_enum::kLinear:
      return new RegressionOutputOp<gpu, mshadow::op::identity, mshadow::op::minus>(param);
    case reg_enum::kLogistic:
      return new RegressionOutputOp<gpu, mshadow_op::sigmoid, mshadow::op::minus>(param);
    case reg_enum::kMAE:
      return new RegressionOutputOp<gpu, mshadow::op::identity, mshadow_op::minus_sign>(param);
    default:
      LOG(FATAL) << "unknown activation type " << type;
  }
  return NULL;
}
}  // namespace op
}  // namespace mxnet

