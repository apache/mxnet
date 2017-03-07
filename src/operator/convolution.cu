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
#include "./nn/im2col.cuh"

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
inline
void ConvolutionOp<xpu, DType>::ConvIm2Col(mshadow::Stream<gpu>* s,
                                           const DType* data_ptr,
                                           const TShape& data_shape,
                                           DType* col_buffer_ptr,
                                           const TShape& col_buffer_shape) const {
  if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
    im2col_gpu(s, data_ptr, conv_in_channels_, data_shape[2], data_shape[3],
               param_.kernel[0], param_.kernel[1], param_.pad[0], param_.pad[1],
               param_.stride[0], param_.stride[1], param_.dilate[0], param_.dilate[1],
               col_buffer_ptr);
  } else {
    im2col_nd_gpu(s, data_ptr, num_spatial_axes_, num_kernels_im2col_,
                  data_shape, col_buffer_shape, param_.kernel, param_.pad,
                  param_.stride, param_.dilate, col_buffer_ptr);
  }
}

template<typename xpu, typename DType>
inline
void ConvolutionOp<xpu, DType>::ConvCol2Im(mshadow::Stream<gpu>* s,
                                           const DType* col_buffer_ptr,
                                           const TShape& col_buffer_shape,
                                           DType* data_ptr,
                                           const TShape& data_shape,
                                           OpReqType req) const {
  if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
    col2im_gpu(s, col_buffer_ptr, conv_in_channels_, data_shape[2], data_shape[3],
               param_.kernel[0], param_.kernel[1], param_.pad[0], param_.pad[1],
               param_.stride[0], param_.stride[1], param_.dilate[0], param_.dilate[1],
               data_ptr, req);
  } else {
    col2im_nd_gpu(s, col_buffer_ptr, num_spatial_axes_, num_kernels_col2im_,
                  data_shape, col_buffer_shape, param_.kernel, param_.pad,
                  param_.stride, param_.dilate, data_ptr, req);
  }
}

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

