/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cu
 * \brief
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(ActivationParam param) {
  switch(param.act_type) {
    case kReLU: return new ActivationOp<gpu, mshadow_op::relu, mshadow_op::relu_grad>();
    case kSigmoid: return new ActivationOp<gpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>();
    case kTanh: return new ActivationOp<gpu, mshadow_op::tanh, mshadow_op::tanh_grad>();
    default:
      LOG(FATAL) << "unknown activation";
      return NULL;
  }
}
}  // op
}  // namespace mxnet

