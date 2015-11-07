/*!
 * Copyright (c) 2015 by Contributors
 * \file block_grad.cc
 * \brief
 * \author Bing Xu
*/
#include "./block_grad-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>() {
  return new BlockGradientOp<gpu>();
}

}  // namespace op
}  // namespace mxnet

