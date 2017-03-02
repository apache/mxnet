/*!
 * Copyright (c) 2017 by Contributors
 * \file im2col.h
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, and dilation.
 * These functions are mainly used in convolution operators.
 */

#ifndef MXNET_OPERATOR_IM2COL_H_
#define MXNET_OPERATOR_IM2COL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <cstring>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

template <typename DType>
inline void fill_array(const int N, const DType val, DType* a) {
  if (static_cast<DType>(0) == val) {
    std::memset(a, static_cast<DType>(0), sizeof(DType) * N);
    return;
  }

  for (int i = 0; i < N; ++i) {
    a[i] = val;
  }
}

template <typename DType>
void im2col_nd_cpu(const DType* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, DType* data_col);

template <typename DType>
void im2col_cpu(const DType* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    DType* data_col);

template <typename DType>
void col2im_nd_cpu(const DType* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, DType* data_im, OpReqType req);

template <typename DType>
void col2im_cpu(const DType* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    DType* data_im, OpReqType req);

template <typename DType>
void im2col_nd_gpu(const DType* data_im, const int num_spatial_axes,
    const int col_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, DType* data_col);

template <typename DType>
void im2col_gpu(const DType* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    DType* data_col);

template <typename DType>
void col2im_nd_gpu(const DType* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, DType* data_im, OpReqType req);

template <typename DType>
void col2im_gpu(const DType* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    DType* data_im, OpReqType req);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IM2COL_H_
