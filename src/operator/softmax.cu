/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax.cu
 * \brief
 * \author Bing Xu
*/

#include "./softmax-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SoftmaxParam param) {
  return new SoftmaxOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

