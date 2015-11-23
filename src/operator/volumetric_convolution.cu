/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu
*/

#include "./volumetric_convolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./volumetric_cudnn_convolution-inl.h"
#endif // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(VolumetricConvolutionParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNVolumetricConvolutionOp(param);
#else
  return new VolumetricConvolutionOp<gpu>(param);
#endif // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet

