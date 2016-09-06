/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu.cc
 * \brief
 * \author Bing Xu
*/

#include "./leaky_relu-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(LeakyReLUParam param) {
  return new LeakyReLUOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

