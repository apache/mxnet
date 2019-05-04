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
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_im2col.h
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai
 */

#ifndef MXNET_OPERATOR_CONTRIB_NN_DEFORMABLE_IM2COL_H_
#define MXNET_OPERATOR_CONTRIB_NN_DEFORMABLE_IM2COL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <cstring>
#include <vector>
#include "../../mxnet_op.h"

namespace mxnet {
namespace op {

template <typename DType>
inline DType deformable_im2col_bilinear_cpu(const DType* data,
    const int height, const int width, DType h, DType w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;

  if (h_low >= height - 1) {
    h_high = height - 1;
    h = (DType)h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = width - 1;
    w = (DType)w_low;
  } else {
    w_high = w_low + 1;
  }

  DType lh = h - h_low;
  DType lw = w - w_low;
  DType hh = 1 - lh, hw = 1 - lw;

  DType v1 = data[h_low * width + w_low];
  DType v2 = data[h_low * width + w_high];
  DType v3 = data[h_high * width + w_low];
  DType v4 = data[h_high * width + w_high];
  DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

/*!
 * \brief deformable_im2col 2D cpu version.
 * DO NOT call this function directly.
 * Use the wrapper function im2col() instead.
 */
template <typename DType>
inline void deformable_im2col_cpu(const DType* data_im, const DType* data_offset,
    const int channels, const int height, const int width,
    const int output_h, const int output_w,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const uint32_t deformable_group,
    DType* data_col) {
  const int channel_size = height * width;
  const int offset_size = 2 * kernel_h * kernel_w * output_h * output_w;
  const int channel_per_deformable_group = channels / deformable_group;
  for (int channel = 0; channel < channels; channel++, data_im += channel_size) {
    if (channel % channel_per_deformable_group == 0 && channel != 0) {
      data_offset += offset_size;
    }
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_row = 0; output_row < output_h; output_row++) {
          int input_col = -pad_w + kernel_col * dilation_w;
          for (int output_col = 0; output_col < output_w; output_col++) {
            int offset_h_ptr = ((2 * (kernel_row * kernel_w + kernel_col)) *
              output_h + output_row) * output_w + output_col;
            int offset_w_ptr = offset_h_ptr + output_h * output_w;
            DType im_row = input_row + data_offset[offset_h_ptr];
            DType im_col = input_col + data_offset[offset_w_ptr];
            if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
              *(data_col++) = deformable_im2col_bilinear_cpu(data_im, height, width, im_row, im_col);
            } else {
              *(data_col++) = 0;
            }
            input_col += stride_w;
          }
          input_row += stride_h;
        }
      }
    }
  }
}

/*!\brief
 * cpu function of deformable_im2col algorithm
 * \param s device stream
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape (#channels, output_im_height, output_im_width, ...)
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param data_col column buffer pointer
 */
template <typename DType>
inline void deformable_im2col(mshadow::Stream<cpu>* s,
  const DType* data_im, const DType* data_offset,
  const mxnet::TShape& im_shape, const mxnet::TShape& col_shape, const mxnet::TShape& kernel_shape,
  const mxnet::TShape& pad, const mxnet::TShape& stride, const mxnet::TShape& dilation,
  const uint32_t deformable_group, DType* data_col) {
  if (2 == kernel_shape.ndim()) {
    deformable_im2col_cpu(data_im, data_offset,
        im_shape[1], im_shape[2], im_shape[3],
        col_shape[1], col_shape[2],
        kernel_shape[0], kernel_shape[1],
        pad[0], pad[1],
        stride[0], stride[1],
        dilation[0], dilation[1],
        deformable_group, data_col);
  } else {
    LOG(FATAL) << "not implemented";
  }
}


/*!\brief
 * cpu function of deformable_col2im algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename DType>
inline void deformable_col2im(mshadow::Stream<cpu>* s,
  const DType* data_col, const DType* data_offset,
  const mxnet::TShape& im_shape, const mxnet::TShape& col_shape, const mxnet::TShape& kernel_shape,
  const mxnet::TShape& pad, const mxnet::TShape& stride,
  const mxnet::TShape& dilation, const uint32_t deformable_group,
  DType* grad_im, OpReqType req) {
  LOG(FATAL) << "only implemented in GPU";
}


/*!\brief
 * cpu function of deformable_col2im_coord algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_offset pointer of the offset (C, H, W,...) in the offset batch
 */

template <typename DType>
inline void deformable_col2im_coord(mshadow::Stream<cpu>* s,
  const DType* data_col, const DType* data_im,
  const DType* data_offset, const mxnet::TShape& im_shape,
  const mxnet::TShape& col_shape, const mxnet::TShape& kernel_shape,
  const mxnet::TShape& pad, const mxnet::TShape& stride,
  const mxnet::TShape& dilation, const uint32_t deformable_group,
  DType* grad_offset, OpReqType req) {
  LOG(FATAL) << "only implemented in GPU";
}

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./deformable_im2col.cuh"
#endif
#endif  // MXNET_OPERATOR_CONTRIB_NN_DEFORMABLE_IM2COL_H_
