/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cu
 * \brief
 * \author Bing Xu
*/

#include "./softmax_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SoftmaxOutputParam param) {
  return new SoftmaxOutputOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

