/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_activation.cu
 * \brief
 * \author Junyuan Xie
*/
#include "./softmax_activation-inl.h"
#include "./mshadow_op.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_softmax_activation-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SoftmaxActivationParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNSoftmaxActivationOp(param);
#else
  LOG(FATAL) << "Softmax Activation for internal layers is only supported "
                "on GPU with cuDNN. Use SoftmaxOutput for loss layer.";
  return new SoftmaxActivationOp<gpu>(param);
#endif  // MXNET_USE_CUDNN
}
}  // op
}  // namespace mxnet

