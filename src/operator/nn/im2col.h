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

#ifndef MXNET_OPERATOR_NN_IM2COL_H_
#define MXNET_OPERATOR_NN_IM2COL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <cstring>
#include <vector>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/*!
 * \brief im2col 2D cpu version.
 * DO NOT call this function directly.
 * Use the wrapper function im2col() instead.
 */
template <typename DType>
inline void im2col_cpu(const DType* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    DType* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  // TODO(junwu): we tested adding openmp (w/ & w/o collapse clause) here
  // for testing the performance of convolution operator,
  // but the total runtime increased by 0.8s for images of shape
  // (8, 32, 64, 64) and decreased by 0.2s for images of shape
  // (16, 64, 64, 64). Both kernel shapes are (8, 8). We think the
  // bottleneck of the convolution operator probably lies in dot().
  // Hence, adding more threads to the loops contributes little
  // toward improving the convolution operator's performance.
  // We will revisit this issue in the future.
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

/*!
 * \brief core function of im2col algorithm. DO NOT call this function directly.
 * Use wrapper function im2col() instead.
 * \param data_input image pointer pointing the first element of channel dim
 * \param im2col determine whether the algorithm is im2col or col2im
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_output start pointer of the column buffer to be filled
 */
template <typename DType>
inline void im2col_nd_core_cpu(const DType* data_input, const bool im2col,
    const TShape& im_shape, const TShape& col_shape,
    const TShape& kernel_shape, const TShape& pad, const TShape& stride,
    const TShape& dilation, DType* data_output, OpReqType req = mxnet::kWriteTo) {
  if (mxnet::kNullOp == req) return;
  index_t num_spatial_axes = kernel_shape.ndim();
  if (!im2col) {
    index_t im_size = im_shape[1];  // skip batch dim
    for (index_t i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[2 + i];
    }
    if (mxnet::kAddTo != req) {
      std::fill(data_output, data_output+im_size, static_cast<DType>(0));
    }
  }
  index_t kernel_size = 1;
  for (index_t i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const index_t channels_col = col_shape[0];
  std::vector<index_t> d_offset(num_spatial_axes, 0);
  std::vector<index_t> d_iter(num_spatial_axes, 0);
  for (index_t c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    index_t offset = c_col;
    for (int d_i = static_cast<int>(num_spatial_axes) - 1; d_i >= 0; --d_i) {
      if (d_i < static_cast<int>(num_spatial_axes) - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      index_t index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (index_t d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const index_t d = d_iter[d_i];
        const int d_im = static_cast<int>(d * stride[d_i] + d_offset[d_i] * dilation[d_i])
          - static_cast<int>(pad[d_i]);
        is_padding |= d_im < 0 || d_im >= static_cast<int>(im_shape[d_i + 2]);
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= static_cast<int>(im_shape[d_i + 2]);
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = static_cast<int>(num_spatial_axes) - 1; d_i >= 0; --d_i) {
        const index_t d_max = col_shape[d_i + 1];
        CHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] + 1 == d_max) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented)
  }  // for (int c = 0; c < channels_col; ++c)
}

/*!
 * \brief cpu function of im2col algorithm
 * \param data_im pointer of a image (C, H, W,...) in the image batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_col start pointer of the column buffer to be filled
 */
template <typename DType>
inline void im2col(mshadow::Stream<cpu>* s,
                   const DType* data_im, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_col) {
  if (2 == kernel_shape.ndim()) {
    im2col_cpu(data_im, im_shape[1], im_shape[2], im_shape[3],
               kernel_shape[0], kernel_shape[1], pad[0], pad[1],
               stride[0], stride[1], dilation[0], dilation[1], data_col);
  } else {
    im2col_nd_core_cpu(data_im, true, im_shape, col_shape,
                       kernel_shape, pad, stride, dilation, data_col);
  }
}

/*!
 * \brief col2im 2D cpu version.
 * DO NOT call this function directly. Use wrapper function col2im() instead.
 */
template <typename DType>
inline void col2im_cpu(const DType* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    DType* data_im, OpReqType req) {
  if (mxnet::kNullOp == req) return;
  if (mxnet::kAddTo != req) {
    std::fill(data_im, data_im+height*width*channels, static_cast<DType>(0));
  }
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  // TODO(junwu): we tested adding openmp (w/ & w/o collapse clause) here
  // for testing the performance of convolution operator,
  // but the total runtime increased by 0.8s for images of shape
  // (8, 32, 64, 64) and decreased by 0.2s for images of shape
  // (16, 64, 64, 64). Both kernel shapes are (8, 8). We think the
  // bottleneck of the convolution operator probably lies in dot().
  // Hence, adding more threads to the loops contributes little
  // toward improving the convolution operator's performance.
  // We will revisit this issue in the future.
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

/*!\brief
 * cpu function of col2im algorithm
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
inline void col2im(mshadow::Stream<cpu>* s,
                   const DType* data_col, const TShape& im_shape,
                   const TShape& col_shape, const TShape& kernel_shape,
                   const TShape& pad, const TShape& stride,
                   const TShape& dilation, DType* data_im, OpReqType req) {
  index_t num_spatial_axes = kernel_shape.ndim();
  if (2 == num_spatial_axes) {
    col2im_cpu(data_col, im_shape[1], im_shape[2], im_shape[3],
               kernel_shape[0], kernel_shape[1], pad[0], pad[1],
               stride[0], stride[1], dilation[0], dilation[1], data_im, req);
  } else {
    im2col_nd_core_cpu(data_col, false, im_shape, col_shape,
                       kernel_shape, pad, stride, dilation, data_im, req);
  }
}

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./im2col.cuh"
#endif
#endif  // MXNET_OPERATOR_NN_IM2COL_H_
