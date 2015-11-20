/*!
 * Copyright (c) 2015 by Contributors
 * \file onehot.cu
 * \brief
 * \author Bing Xu
*/

#include "./onehot-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(OnehotParam param) {
  return new OnehotEmbeddingOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet

