/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cu
 * \brief
 * \author Bing Xu
*/

#include "./pooling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(PoolingParam param) {
  switch (param.type) {
    case kMaxPooling: return new PoolingOp<gpu, mshadow::red::maximum>(param);
    case kAvgPooling: return new PoolingOp<gpu, mshadow::red::sum>(param);
    case kSumPooling: return new PoolingOp<gpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
}

}  // namespace op
}  // namespace mxnet

