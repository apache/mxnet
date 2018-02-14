/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2017 by Contributors
 * \file im2col.h
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, and dilation.
 * These functions are mainly used in convolution operators.
 * The implementation of the im2col and col2im algorithms
 * are copied from Caffe with minor interface modifications
 * adapting to MXNet data structures.
 */

#ifndef MXNET_OPERATOR_NN_IM2COL_CUH_
#define MXNET_OPERATOR_NN_IM2COL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief im2col gpu kernel.
 * DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename DType>
__global__ void im2col_gpu_kernel(const int n, const DType* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    DType* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    DType* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const DType* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : static_cast<DType>(0);
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

/*!
 * \brief DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename DType>
inline void im2col_gpu(mshadow::Stream<gpu>* s,
                       const DType* data_im, const int channels,
                       const int height, const int width,
                       const int kernel_h, const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       DType* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  using namespace mxnet_op;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                             0, mshadow::Stream<gpu>::GetStream(s)>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  MSHADOW_CUDA_POST_KERNEL_CHECK(im2col_gpu_kernel);
}

/*!
 * \brief DO NOT call this directly. Use wrapper function col2im() instead;
 */
template <typename DType>
__global__ void col2im_gpu_kernel(const int n, const DType* data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    DType* data_im, OpReqType req) {
  CUDA_KERNEL_LOOP(index, n) {
    DType val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO(caffe): use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    KERNEL_ASSIGN(data_im[index], req, val);
  }
}

/*!
 * \brief DO NOT call this directly. Use wrapper function col2im() instead;
 */
using mshadow::Shape;
template <typename DType, int num_axes>
__global__ void im2col_nd_gpu_kernel(const int n, const DType* data_im,
    const Shape<num_axes+2> im_shape, const Shape<num_axes+1> col_shape,
    const Shape<num_axes> kernel_shape, const Shape<num_axes> pad, const Shape<num_axes> stride,
    const Shape<num_axes> dilation, DType* data_col) {
  int d_temp[num_axes];  // NOLINT(runtime/arrays)
  int d_iter[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x+1];  // skip batch dim
  }
  __syncthreads();

  int i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= shared_col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      channel_in *= shared_im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= shared_col_shape[i + 1];
      d_iter[i] = 0;
    }
    DType* data_col_ptr = data_col + channel_out;
    const DType* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_im_offset = d_iter[0] * shared_dilation[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

/*!\brief im2col gpu version
 * \param s device stream
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param col_shape column buffer shape (#channels, output_im_height, output_im_width, ...)
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_col column buffer pointer
 */
template <typename DType>
inline void im2col(mshadow::Stream<gpu>* s,
                   const DType* data_im, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_col) {
  // num_axes should be smaller than block size
  index_t num_spatial_axes = kernel_shape.ndim();
  CHECK_LT(num_spatial_axes, mshadow::cuda::kBaseThreadNum);
  index_t num_kernels = im_shape[1] * col_shape.ProdShape(1, col_shape.ndim());
  using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 1:
    im2col_nd_gpu_kernel<DType, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, im_shape.get<3>(), col_shape.get<2>(),
        kernel_shape.get<1>(), pad.get<1>(), stride.get<1>(), dilation.get<1>(), data_col);
    break;
  case 2:
    im2col_gpu_kernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1],
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        col_shape[1], col_shape[2], data_col);
    break;
  case 3:
    im2col_nd_gpu_kernel<DType, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, im_shape.get<5>(), col_shape.get<4>(),
        kernel_shape.get<3>(), pad.get<3>(), stride.get<3>(), dilation.get<3>(), data_col);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(im2col_nd_gpu_kernel);
}

/*!
 * \brief DO NOT call this directly. Use wrapper function col2im() instead;
 */
template <typename DType>
inline void col2im_gpu(mshadow::Stream<gpu>* s, const DType* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    DType* data_im, OpReqType req) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  using namespace mxnet_op;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                             0, mshadow::Stream<gpu>::GetStream(s)>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im, req);
  MSHADOW_CUDA_POST_KERNEL_CHECK(col2im_gpu_kernel);
}

/*!
 * \brief DO NOT call this directly. Use wrapper function col2im() instead;
 */
template <typename DType, int num_axes>
__global__ void col2im_nd_gpu_kernel(const int n, const DType* data_col,
    const Shape<num_axes+2> im_shape, const Shape<num_axes+1> col_shape,
    const Shape<num_axes> kernel_shape, const Shape<num_axes> pad, const Shape<num_axes> stride,
    const Shape<num_axes> dilation, DType* data_im, OpReqType req) {
  int d_im[num_axes];  // NOLINT(runtime/arrays)
  int d_col_iter[num_axes];  // NOLINT(runtime/arrays)
  int d_col_start[num_axes];  // NOLINT(runtime/arrays)
  int d_col_end[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x+1];  // skip batch dim
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int c_im = index;
    // Calculate d_im (image dimensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
      c_im /= shared_im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int i = 0; i < num_axes; ++i) {
      const int kernel_extent =
          shared_dilation[i] * (shared_kernel_shape[i] - 1) + 1;
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_extent) ? 0 :
          (d_im[i] - kernel_extent) / shared_stride[i] + 1;
      d_col_end[i] =
          min(d_im[i] / shared_stride[i] + 1, shared_col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    DType val = 0;
    bool incremented = true;
    bool skip = false;
    do {
      // Compute the final offset.
      int final_offset = 0;
      int kernel_shape_prod = 1;
      int kernel_index;
      for (int i = num_axes - 1; i >= 0; --i) {
        kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
        if (kernel_index % shared_dilation[i]) {
          skip = true;
          break;
        } else {
          kernel_index /= shared_dilation[i];
          final_offset += kernel_index * kernel_shape_prod;
          kernel_shape_prod *= shared_kernel_shape[i];
        }
      }
      if (!skip) {
        final_offset += kernel_shape_prod * c_im;
        for (int i = 0; i < num_axes; ++i) {
          final_offset *= shared_col_shape[i + 1];
          final_offset += d_col_iter[i];
        }
        val += data_col[final_offset];
      }
      skip = false;
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    KERNEL_ASSIGN(data_im[index], req, val);
  }  // CUDA_KERNEL_LOOP(index, n)
}

/*!\brief
 * gpu function of col2im algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename DType>
inline void col2im(mshadow::Stream<gpu>* s,
                   const DType* data_col, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_im, OpReqType req) {
  index_t num_spatial_axes = kernel_shape.ndim();
  index_t im_size = im_shape.ProdShape(1, im_shape.ndim());
  // num_axes should be smaller than block size
  CHECK_LT(num_spatial_axes, mshadow::cuda::kBaseThreadNum);
  using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 1:
    col2im_nd_gpu_kernel<DType, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(im_size), mshadow::cuda::kBaseThreadNum,
             0, mshadow::Stream<gpu>::GetStream(s)>>>(
          im_size, data_col, im_shape.get<3>(), col_shape.get<2>(),
          kernel_shape.get<1>(), pad.get<1>(), stride.get<1>(), dilation.get<1>(),
          data_im, req);
    MSHADOW_CUDA_POST_KERNEL_CHECK(col2im_nd_gpu_kernel);
    break;
  case 2:
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
    col2im_gpu_kernel<DType><<<cuda_get_num_blocks(im_size), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
        im_size, data_col, im_shape[1], im_shape[2], im_shape[3],
        kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
        dilation[0], dilation[1], col_shape[1], col_shape[2], data_im, req);
    MSHADOW_CUDA_POST_KERNEL_CHECK(col2im_gpu_kernel);
    break;
  case 3:
    col2im_nd_gpu_kernel<DType, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<cuda_get_num_blocks(im_size), mshadow::cuda::kBaseThreadNum,
             0, mshadow::Stream<gpu>::GetStream(s)>>>(
          im_size, data_col, im_shape.get<5>(), col_shape.get<4>(),
          kernel_shape.get<3>(), pad.get<3>(), stride.get<3>(), dilation.get<3>(),
          data_im, req);
    MSHADOW_CUDA_POST_KERNEL_CHECK(col2im_nd_gpu_kernel);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_IM2COL_CUH_
