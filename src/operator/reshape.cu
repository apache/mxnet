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
Operator *CreateOp<gpu>(ReshapeParam param) {
  return new ReshapeOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
