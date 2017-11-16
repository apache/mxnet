/*!
 * Copyright (c) 2017 by Contributors
 * \file pooling.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/
#include <vector>
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
  if (!param.cudnn_off && param.kernel.ndim() > 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      switch (param.pool_type) {
        case pool_enum::kMaxPooling:
          op = new CuDNNPoolingOp<DType>(param);
          break;
        case pool_enum::kAvgPooling:
          op = new CuDNNPoolingOp<DType>(param);
          break;
        case pool_enum::kSumPooling:
          LOG(WARNING) << "Sum pooling is not supported by cudnn, MXNet sum pooling is applied.";
          break;
      }
    });
  }
  if (op) return op;
#endif  // MXNET_USE_CUDNN
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      op = new PoolingOp<gpu, DType>(param);
    } else {
      LOG(FATAL) << "unknown pooling type";
    }
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
