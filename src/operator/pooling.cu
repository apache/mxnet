/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling.cu
 * \brief
 * \author Bing Xu
*/

#include "./pooling-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_pooling-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(PoolingParam param) {
#if MXNET_USE_CUDNN == 1
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return new CuDNNPoolingOp(param);
    case pool_enum::kAvgPooling:
      return new CuDNNPoolingOp(param);
    case pool_enum::kSumPooling:
      LOG(WARNING) << "Sum pooling is not supported by cudnn, MxNet sum pooling is applied.";
      return new PoolingOp<gpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown pooling type";
      return NULL;
  }
#else
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return new PoolingOp<gpu, mshadow::red::maximum>(param);
    case pool_enum::kAvgPooling:
      return new PoolingOp<gpu, mshadow::red::sum>(param);
    case pool_enum::kSumPooling:
      return new PoolingOp<gpu, mshadow::red::sum>(param);
    default:
      LOG(FATAL) << "unknown pooling type";
      return NULL;
  }
#endif  // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet

