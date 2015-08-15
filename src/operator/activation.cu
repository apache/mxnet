/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cu
 * \brief
 * \author Bing Xu
*/

#include "./activation-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateActivationOp<gpu>(ActivationOpType type) {
  switch(type) {
    case kReLU: return new ActivationOp<gpu, act::relu, act::relu_grad>();
    default: return NULL;
  }
}
}  // op
}  // namespace mxnet

