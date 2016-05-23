/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ConvolutionParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1
  if (param.dilate[0] == 1 && param.dilate[1] == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new CuDNNConvolutionOp<DType>(param);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new ConvolutionOp<gpu, DType>(param);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet

