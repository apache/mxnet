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
Operator *CreateOp<gpu>(PoolingParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.pool_type) {
      case pool_enum::kMaxPooling:
        op = new CuDNNPoolingOp<DType>(param);
        break;
      case pool_enum::kAvgPooling:
        op = new CuDNNPoolingOp<DType>(param);
        break;
      case pool_enum::kSumPooling:
        LOG(WARNING) << "Sum pooling is not supported by cudnn, MxNet sum pooling is applied.";
        op = new PoolingOp<gpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
  });
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.pool_type) {
      case pool_enum::kMaxPooling:
        op = new PoolingOp<gpu, mshadow::red::maximum, DType>(param);
        break;
      case pool_enum::kAvgPooling:
        op = new PoolingOp<gpu, mshadow::red::sum, DType>(param);
        break;
      case pool_enum::kSumPooling:
        op = new PoolingOp<gpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
  });
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet

