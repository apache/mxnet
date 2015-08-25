/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ConvolutionParam param) {
  return new ConvolutionOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

