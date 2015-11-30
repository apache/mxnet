/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cu
 * \brief
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "./mshadow_op.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_activation-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(ActivationParam param) {
  // SoftReLU not supported by CUDNN yet
  if (param.act_type == activation::kSoftReLU)
      return new ActivationOp<gpu, mshadow_op::softrelu, mshadow_op::softrelu_grad>();

#if MXNET_USE_CUDNN == 1
  return new CuDNNActivationOp(param);
#else
  switch(param.act_type) {
    case activation::kReLU:
      return new ActivationOp<gpu, mshadow_op::relu, mshadow_op::relu_grad>();
    case activation::kSigmoid:
      return new ActivationOp<gpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>();
    case activation::kTanh:
      return new ActivationOp<gpu, mshadow_op::tanh, mshadow_op::tanh_grad>();
    default:
      LOG(FATAL) << "unknown activation";
      return NULL;
  }
#endif  // MXNET_USE_CUDNN
}
}  // op
}  // namespace mxnet

