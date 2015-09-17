/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ActivationParam param) {
  switch (param.act_type) {
    case kReLU:
      return new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad>();
    case kSigmoid:
      return new ActivationOp<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>();
    case kTanh:
      return new ActivationOp<cpu, mshadow_op::tanh, mshadow_op::tanh_grad>();
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ActivationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_REGISTER_OP_PROPERTY(Activation, ActivationProp)
.describe("Apply activation function to input.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

