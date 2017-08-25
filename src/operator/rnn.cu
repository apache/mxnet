/*!
 * Copyright (c) 2015 by Contributors
 * \file rnn.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./rnn-inl.h"
#include <algorithm>
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_rnn-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(RNNParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNRNNOp<DType>(param);
  })
#else
  LOG(FATAL) << "RNN is only available for cuDNN at the moment.";
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
  return op;
}

}  // namespace op
}  // namespace mxnet
