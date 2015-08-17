/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(FullyConnectedParam param) {
  return new FullyConnectedOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
