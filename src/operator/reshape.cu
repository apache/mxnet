/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

#include "./reshape-inl.h"


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>() {
  return new ReshapeOp<gpu>();
}

}  // namespace op
}  // namespace mxnet
