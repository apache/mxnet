/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_binary_op.cu
 * \brief elementwise binary operator
*/
#include "./elementwise_binary_op-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateElementWiseBinaryOp<gpu>(ElementWiseBinaryOpType type) {
  return CreateElementWiseBinaryOp_<gpu>(type);
}
}  // namespace op
}  // namespace mxnet
