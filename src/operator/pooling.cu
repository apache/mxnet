/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cu
 * \brief
 * \author Bing Xu
*/

#include "./pooling-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_pooling-inl.h"
#endif // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(PoolingParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNPoolingOp(param);
#else
  switch (param.pool_type) {
    case kMaxPooling:
      return new PoolingOp<gpu, mshadow::red::maximum>(param);
    case kAvgPooling:
      return new PoolingOp<gpu, mshadow::red::sum>(param);
    case kSumPooling:
      return new PoolingOp<gpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown activation type";
      return NULL;
  }
#endif  // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet

