/*!
 * Copyright (c) 2015 by Contributors
 * \file deconvolution.cu
 * \brief
 * \author Wei Wu
*/

#include "./deconvolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_deconvolution-inl.h"
#endif // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(DeconvolutionParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNDeconvolutionOp(param);
#else
  return new DeconvolutionOp<gpu>(param);
#endif // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet
