/*!
 * Copyright (c) 2015 by Contributors
 * \file native_op.cu
 * \brief
 * \author Junyuan Xie
*/
#include "./native_op-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(NativeOpParam param) {
  return new NativeOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
