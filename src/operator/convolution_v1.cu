/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution_v1.cu
 * \brief
 * \author Bing Xu
*/

#include "./convolution_v1-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ConvolutionV1Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionV1Op<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

