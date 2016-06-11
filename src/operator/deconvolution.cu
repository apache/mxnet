/*!
 * Copyright (c) 2015 by Contributors
 * \file deconvolution.cu
 * \brief
 * \author Wei Wu
*/

#include "./deconvolution-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_deconvolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
template<>
Operator* CreateOp<gpu>(DeconvolutionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNDeconvolutionOp<DType>(param);
  });
  return op;
}
#else
template<>
Operator* CreateOp<gpu>(DeconvolutionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DeconvolutionOp<gpu, DType>(param);
  });
  return op;
}
#endif  // MXNET_USE_CUDNN

}  // namespace op
}  // namespace mxnet
