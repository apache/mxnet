/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cu
 * \brief
 * \author Bing Xu
*/

#include "./batch_norm-inl.h"
#include "./cudnn_batch_norm-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(BatchNormParam param) {
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  return new CuDNNBatchNormOp(param);
#else
  return new BatchNormOp<gpu>(param);
#endif
}

}  // namespace op
}  // namespace mxnet

