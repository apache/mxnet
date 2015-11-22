/*!
 * Copyright (c) 2015 by Contributors
 * \file ndarray_op.cu
 * \brief
 * \author Junyuan Xie
*/
#include "./ndarray_op-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(NDArrayOpParam param) {
  return new NDArrayOp<gpu>(param);
}

template<>
Context NDArrayOp<gpu>::get_ctx() {
  int dev_id;
  CHECK_EQ(cudaGetDevice(&dev_id), cudaSuccess);
  return Context::GPU(dev_id);
}
}  // namespace op
}  // namespace mxnet
