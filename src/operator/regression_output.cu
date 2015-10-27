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
Operator *CreateRegressionOutputOp<gpu>(RegressionOutputType type) {
  switch (type) {
    case kLinear:
      return new RegressionOutputOp<gpu, mshadow::op::identity, mshadow::op::minus>();
    case kLogistic:
      return new RegressionOutputOp<gpu, mshadow_op::sigmoid, mshadow::op::minus>();
    default:
      LOG(FATAL) << "unknown activation type " << type;
  }
  return NULL;
}
}  // namespace op
}  // namespace mxnet

