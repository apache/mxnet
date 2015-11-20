/*!
 * Copyright (c) 2015 by Contributors
 * \file embedding.cu
 * \brief
 * \author Bing Xu
*/

#include "./embedding-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(EmbeddingParam param) {
  return new EmbeddingOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet

