/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ConvolutionParam param) {
#if MXNET_USE_CUDNN == 1
  return new CuDNNConvolutionOp(param);
#else
  return new ConvolutionOp<gpu>(param);
#endif // MXNET_USE_CUDNN
}

}  // namespace op
}  // namespace mxnet

