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
Operator* CreateOp<gpu>(EmbeddingParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new EmbeddingOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet

