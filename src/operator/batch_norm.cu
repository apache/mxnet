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
  if (!param.use_global_stats) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new CuDNNBatchNormOp<DType>(param);
    })
  } else {
    op = new BatchNormOp<gpu>(param);
  }
#else
  op = new BatchNormOp<gpu>(param);
#endif
  return op;
}

}  // namespace op
}  // namespace mxnet

