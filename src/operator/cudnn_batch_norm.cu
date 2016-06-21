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
Operator *CreateOp_CuDNNv4<gpu>(BatchNormParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNBatchNormOp<DType>(param);
  })
  return op;
}
#endif  // CUDNN_MAJOR == 4
}  // namespace op
}  // namespace mxnet

