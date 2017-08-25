/*!
 * Copyright (c) 2015 by Contributors
 * \file pooling_v1.cu
 * \brief
 * \author Bing Xu
*/
#include <vector>
#include "./pooling_v1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(PoolingV1Param param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.pool_type) {
      case pool_v1_enum::kMaxPooling:
        op = new PoolingV1Op<gpu, mshadow::red::maximum, DType>(param);
        break;
      case pool_v1_enum::kAvgPooling:
        op = new PoolingV1Op<gpu, mshadow::red::sum, DType>(param);
        break;
      case pool_v1_enum::kSumPooling:
        op = new PoolingV1Op<gpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

