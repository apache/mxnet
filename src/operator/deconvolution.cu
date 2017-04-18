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
template<>
Operator* CreateOp<gpu>(DeconvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  // Logic here parallels that in Convolution.cu
  Operator *op = NULL;
  // If 1D deconvolution, use MXNet implementation
  if (param.kernel.ndim() == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new DeconvolutionOp<gpu, DType>(param);
    })
    return op;
  }
#if MXNET_USE_CUDNN == 1
  // The NVIDIA Pascal architecture was the first to include 16-bit ALUs.
  // Thus, when the framework is compiled with MSHADOW_USE_PASCAL == 1, we
  // perform the deconvolution calculation in 16-bit when the tensor type is
  // also 16-bit.  For NVIDIA architectures earlier than Pascal (so Maxwell
  // and Kepler), the computation precision is always at least 32-bits.
#if MSHADOW_USE_PASCAL == 1
  // true fp16
  int desired_forward_compute_type = dtype;
  int desired_backward_compute_type = dtype;
#else
  // pseudo fp16
  int desired_forward_compute_type =
    (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;
  int desired_backward_compute_type =
    (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;
#endif  // MSHADOW_USE_PASCAL == 1

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      op = new DeconvolutionOp<gpu, DType>(param);
    } else {
      int forward_compute_type = desired_forward_compute_type;
      int backward_compute_type = desired_backward_compute_type;
      bool deconvolutionIsSupported = CuDNNDeconvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type);

      // If cuDNN can't handle this case with fp16 backprop kernels, try fp32 backprop.
      if (!deconvolutionIsSupported && backward_compute_type == mshadow::kFloat16) {
        backward_compute_type = mshadow::kFloat32;
        deconvolutionIsSupported = CuDNNDeconvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type);
      }

      // If cuDNN can't handle this case with fp16 forward kernels, try fp32
      if (!deconvolutionIsSupported && forward_compute_type == mshadow::kFloat16) {
        forward_compute_type = mshadow::kFloat32;
        deconvolutionIsSupported = CuDNNDeconvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type);
      }
      if (!deconvolutionIsSupported) {
        LOG(WARNING) <<
          "This deconvolution is not supported by cudnn, MXNET deconvolution is applied.";
        op = new DeconvolutionOp<gpu, DType>(param);
      } else {
        if ((forward_compute_type != desired_forward_compute_type) ||
            (backward_compute_type != desired_backward_compute_type)) {
          LOG(WARNING) <<
            "True fp16 deconvolution by cudnn not supported in this configuration.  " <<
            "Falling back to pseudo fp16.";
        }
        op = new CuDNNDeconvolutionOp<DType>(param,
                                         forward_compute_type,
                                         backward_compute_type,
                                         *in_shape, *out_shape, ctx);
      }
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DeconvolutionOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet
