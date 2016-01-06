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
    case activation::kReLU:
      return new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad>();
    case activation::kSigmoid:
      return new ActivationOp<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>();
    case activation::kTanh:
      return new ActivationOp<cpu, mshadow_op::tanh, mshadow_op::tanh_grad>();
    case activation::kSoftReLU:
      return new ActivationOp<cpu, mshadow_op::softrelu, mshadow_op::softrelu_grad>();
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
.describe("Apply activation function to input."
          "Softmax Activation is only available with CUDNN on GPU"
          "and will be computed at each location across channel if input is 4D.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

