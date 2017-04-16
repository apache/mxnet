/*!
 * Copyright (c) 2017 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/

#include "./convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  // If 1D convolution, use MXNet implementation
  if (param.kernel.ndim() == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new ConvolutionOp<gpu, DType>(param);
    })
    return op;
  }
#if MXNET_USE_CUDNN == 1
  if (param.dilate.Size() == 1 && !param.cudnn_off) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new CuDNNConvolutionOp<DType>(param, *in_shape, *out_shape, ctx);
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

