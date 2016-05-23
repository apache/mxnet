/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn.cu
 * \brief
 * \author Bing Xu
*/

#include "./lrn-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_lrn-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(LRNParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNLocalResponseNormOp(param);
#else
#if CUDA_VERSION == 7000
  LOG(FATAL) << "Due to old CUDA compiler bug, LRN is disabled."
             << "Please upgrade CUDA to 7.5+ or use CUDNN";
  return NULL;
#else
  return new LocalResponseNormOp<gpu>(param);
#endif  // CUDA_VERSION
#endif  // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet


