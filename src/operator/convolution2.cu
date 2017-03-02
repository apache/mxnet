/*!
 * Copyright (c) 2017 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/

#include "./convolution2-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
// TODO(junwu): enable cudnn
//#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN
#include "./nn/im2col.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
inline
void Convolution2Op<xpu, DType>::ConvIm2Col(const index_t n, const TBlob& data,
                                            TBlob* col_buffer, const gpu& dev_gpu) const {
  const DType* data_ptr = data.dptr<DType>() + n * input_dim_;
  if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
    im2col_gpu(data_ptr, conv_in_channels_, data.size(2), data.size(3),
               param_.kernel[0], param_.kernel[1], param_.pad[0], param_.pad[1],
               param_.stride[0], param_.stride[1], param_.dilate[0], param_.dilate[1],
               col_buffer->dptr<DType>());
  } else {
    im2col_nd_gpu(data_ptr, num_spatial_axes_, num_kernels_im2col_,
                  reinterpret_cast<const int*>(&(data.shape_[1])),
                  reinterpret_cast<int*>(col_buffer->shape_.data()),
                  reinterpret_cast<const int*>(param_.kernel.data()),
                  reinterpret_cast<const int*>(param_.pad.data()),
                  reinterpret_cast<const int*>(param_.stride.data()),
                  reinterpret_cast<const int*>(param_.dilate.data()),
                  col_buffer->dptr<DType>());
  }
}

template<typename xpu, typename DType>
inline
void Convolution2Op<xpu, DType>::ConvCol2Im(const index_t n, const TBlob& col_buffer, TBlob* gdata,
                                            OpReqType req, const gpu& dev_gpu) const {
}

template<>
Operator* CreateOp<gpu>(Convolution2Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1
  if (param.dilate.Size() == 1 && !param.cudnn_off) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      // TODO(junwu): enable cudnn
      // op = new CuDNNConvolutionOp<DType>(param, *in_shape, *out_shape, ctx);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new Convolution2Op<gpu, DType>(param);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Convolution2Op<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet

