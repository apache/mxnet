/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_batch_norm.cu
 * \brief
 * \author Junyuan Xie
*/

#include "./cudnn_batch_norm-inl.h"
#include <vector>

namespace mxnet {
namespace op {
#if CUDNN_MAJOR == 4
template<>
Operator *CreateOp_CuDNNv4<gpu>(BatchNormParam param) {
  return new CuDNNBatchNormOp<float>(param);
}
#endif  // CUDNN_MAJOR == 4
}  // namespace op
}  // namespace mxnet

