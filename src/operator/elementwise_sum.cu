/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_sum.cu
 * \brief elementwise sum operator
*/
#include "./elementwise_sum-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ElementWiseSumParam param) {
  return new ElementWiseSumOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
