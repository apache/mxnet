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
Operator *CreateOp<gpu>(BatchNormParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNBatchNormOp<DType>(param);
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BatchNormOp<gpu, DType>(param);
  })
#endif
  return op;
}

}  // namespace op
}  // namespace mxnet

