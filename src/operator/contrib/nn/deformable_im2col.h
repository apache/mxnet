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
#include <algorithm>
#include "../../mxnet_op.h"

namespace mxnet {
namespace op {

template <typename DType>
inline DType im2col_bilinear_cpu(const DType* data,
                                 const index_t height,
                                 const index_t width,
                                 DType h, DType w) {
  index_t h_low = floor(h);
  index_t w_low = floor(w);
  index_t h_high;
  index_t w_high;

  if (h_low >= height - 1) {
    h_high = height - 1;
    h = static_cast<DType>(h_low);
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = width - 1;
    w = static_cast<DType>(w_low);
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


template <typename DType>
inline DType get_gradient_weight_cpu(DType argmax_h, DType argmax_w,
                                    const index_t h, const index_t w,
                                    const index_t height, const index_t width) {
  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  argmax_h = std::max(argmax_h, static_cast<DType>(0.0f));
  argmax_w = std::max(argmax_w, static_cast<DType>(0.0f));

  index_t argmax_h_low = static_cast<index_t>(argmax_h);
  index_t argmax_w_low = static_cast<index_t>(argmax_w);
  index_t argmax_h_high;
  index_t argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = static_cast<DType>(argmax_h_low);
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = static_cast<DType>(argmax_w_low);
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  DType weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  } else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}


template <typename DType>
inline DType get_coordinate_weight_cpu(DType argmax_h, DType argmax_w,
                                       const index_t height, const index_t width,
                                       const DType* im_data,
                                       const index_t data_width, const index_t bp_dir) {
  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  if (argmax_h < 0) argmax_h = 0;
  if (argmax_w < 0) argmax_w = 0;

  index_t argmax_h_low = static_cast<index_t>(argmax_h);
  index_t argmax_w_low = static_cast<index_t>(argmax_w);
  index_t argmax_h_high;
  index_t argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = static_cast<DType>(argmax_h_low);
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = static_cast<DType>(argmax_w_low);
  } else {
    argmax_w_high = argmax_w_low + 1;
  }

  DType weight = 0;
  DType im_ll = im_data[argmax_h_low * data_width + argmax_w_low];
  DType im_lh = im_data[argmax_h_low * data_width + argmax_w_high];
  DType im_hl = im_data[argmax_h_high * data_width + argmax_w_low];
  DType im_hh = im_data[argmax_h_high * data_width + argmax_w_high];
  if (bp_dir == 0) {
    weight += -1 * (argmax_w_low + 1 - argmax_w) * im_ll;
    weight += -1 * (argmax_w - argmax_w_low) * im_lh;
    weight += (argmax_w_low + 1 - argmax_w) * im_hl;
    weight += (argmax_w - argmax_w_low) * im_hh;
  } else if (bp_dir == 1) {
    weight += -1 * (argmax_h_low + 1 - argmax_h) * im_ll;
    weight += (argmax_h_low + 1 - argmax_h) * im_lh;
    weight += -1 * (argmax_h - argmax_h_low) * im_hl;
    weight += (argmax_h - argmax_h_low) * im_hh;
  }

  return weight;
}


/*!
 * \brief deformable_im2col 2D cpu version.
 * DO NOT call this function directly.
 * Use the wrapper function im2col() instead.
 */
template <typename DType>
inline void deformable_im2col_cpu(const DType* data_im,
                                  const DType* data_offset,
                                  const index_t channels,
                                  const index_t height, const index_t width,
                                  const index_t kernel_h, const index_t kernel_w,
                                  const index_t pad_h, const index_t pad_w,
                                  const index_t stride_h, const index_t stride_w,
                                  const index_t dilation_h, const index_t dilation_w,
                                  const index_t deformable_group,
                                  const index_t height_col, const index_t width_col,
                                  DType* data_col) {
  const index_t channel_size = height * width;
  const index_t offset_size = 2 * kernel_h * kernel_w * height_col * width_col;
  const index_t channel_per_group = channels / deformable_group;
  for (index_t channel = 0; channel < channels; channel++, data_im += channel_size) {
    if (channel % channel_per_group == 0 && channel != 0) {
      data_offset += offset_size;
    }
    for (index_t i = 0; i < kernel_h; i++) {
      for (index_t j = 0; j < kernel_w; j++) {
        index_t input_row = -pad_h + i * dilation_h;
        for (index_t h_col = 0; h_col < height_col; h_col++) {
          index_t input_col = -pad_w + j * dilation_w;
          for (index_t w_col = 0; w_col < width_col; w_col++) {
            index_t offset_h_ptr = ((2 * (i * kernel_w + j)) *
              height_col + h_col) * width_col + w_col;
            index_t offset_w_ptr = offset_h_ptr + height_col * width_col;
            DType im_row = input_row + data_offset[offset_h_ptr];
            DType im_col = input_col + data_offset[offset_w_ptr];
            if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
              *(data_col++) = im2col_bilinear_cpu(data_im, height, width, im_row, im_col);
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
                              const mxnet::TShape& im_shape,
                              const mxnet::TShape& col_shape,
                              const mxnet::TShape& kernel_shape,
                              const mxnet::TShape& pad,
                              const mxnet::TShape& stride,
                              const mxnet::TShape& dilation,
                              const index_t deformable_group,
                              DType* data_col) {
  if (2 == kernel_shape.ndim()) {
    deformable_im2col_cpu(data_im, data_offset,
                          im_shape[1], im_shape[2], im_shape[3],
                          kernel_shape[0], kernel_shape[1],
                          pad[0], pad[1],
                          stride[0], stride[1],
                          dilation[0], dilation[1],
                          deformable_group,
                          col_shape[1], col_shape[2], data_col);
  } else {
    LOG(FATAL) << "not implemented";
  }
}


/*!
 * \brief deformable_col2im cpu version.
 * DO NOT call this directly.
 * Use wrapper function deformable_col2im() instead;
 */
template <typename DType>
inline void deformable_col2im_cpu(const DType* data_col,
                                  const DType* data_offset, const index_t channels,
                                  const index_t height, const index_t width,
                                  const index_t kernel_h, const index_t kernel_w,
                                  const index_t pad_h, const index_t pad_w,
                                  const index_t stride_h, const index_t stride_w,
                                  const index_t dilation_h, const index_t dilation_w,
                                  const index_t deformable_group,
                                  const index_t height_col, const index_t width_col,
                                  DType* grad_im) {
  index_t channel_per_group = channels / deformable_group;
  index_t count = channels * kernel_h * kernel_w * height_col * width_col;
  for (index_t index = 0; index < count; ++index) {
    const index_t j = (index / width_col / height_col) % kernel_w;
    const index_t i = (index / width_col / height_col / kernel_w) % kernel_h;
    const index_t c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    const index_t group_index = c / channel_per_group;
    const index_t group_offset_step = 2 * kernel_h * kernel_w * height_col * width_col;

    index_t w_col = index % width_col;
    index_t h_col = (index / width_col) % height_col;
    index_t w_in = w_col * stride_w - pad_w;
    index_t h_in = h_col * stride_h - pad_h;

    const DType* data_offset_ptr = data_offset + group_index * group_offset_step;
    const index_t data_offset_h_ptr = ((2 * (i * kernel_w + j)) *
      height_col + h_col) * width_col + w_col;
    const index_t data_offset_w_ptr = data_offset_h_ptr + height_col * width_col;
    const DType offset_h = data_offset_ptr[data_offset_h_ptr];
    const DType offset_w = data_offset_ptr[data_offset_w_ptr];
    const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const DType cur_top_grad = data_col[index];
    const index_t cur_h = static_cast<index_t>(cur_inv_h_data);
    const index_t cur_w = static_cast<index_t>(cur_inv_w_data);
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
          cur_w + dx >= 0 && cur_w + dx < width &&
          std::abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
          std::abs(cur_inv_w_data - (cur_w + dx)) < 1
          ) {
          index_t cur_bottom_grad_pos = (c * height + cur_h + dy) * width + cur_w + dx;
          DType weight = get_gradient_weight_cpu(cur_inv_h_data, cur_inv_w_data,
                                                 cur_h + dy, cur_w + dx, height, width);
          grad_im[cur_bottom_grad_pos] += weight * cur_top_grad;
        }
      }
    }
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
                              const DType* data_col,
                              const DType* data_offset,
                              const mxnet::TShape& im_shape,
                              const mxnet::TShape& col_shape,
                              const mxnet::TShape& kernel_shape,
                              const mxnet::TShape& pad,
                              const mxnet::TShape& stride,
                              const mxnet::TShape& dilation,
                              const index_t deformable_group,
                              DType* grad_im) {
  if (2 == kernel_shape.ndim()) {
    deformable_col2im_cpu(data_col, data_offset,
                          im_shape[1], im_shape[2], im_shape[3],
                          kernel_shape[0], kernel_shape[1],
                          pad[0], pad[1], stride[0], stride[1],
                          dilation[0], dilation[1],
                          deformable_group,
                          col_shape[1], col_shape[2], grad_im);
  } else {
    LOG(FATAL) << "not implemented";
  }
}


/*!
 * \brief deformable_col2im_coord cpu version.
 * DO NOT call this directly.
 * Use wrapper function deformable_col2im_coord() instead;
 */
template <typename DType>
inline void deformable_col2im_coord_cpu(const DType* data_col,
                                        const DType* data_im,
                                        const DType* data_offset,
                                        const index_t channels,
                                        const index_t height, const index_t width,
                                        const index_t kernel_h, const index_t kernel_w,
                                        const index_t pad_h, const index_t pad_w,
                                        const index_t stride_h, const index_t stride_w,
                                        const index_t dilation_h, const index_t dilation_w,
                                        const index_t deformable_group,
                                        const index_t height_col, const index_t width_col,
                                        DType* grad_offset) {
  index_t channel_per_group = channels * kernel_h * kernel_w / deformable_group;
  index_t count = height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  for (index_t index = 0; index < count; ++index) {
    DType val = 0;
    index_t w = index % width_col;
    index_t h = (index / width_col) % height_col;
    index_t c = index / width_col / height_col;
    // compute the start and end of the output

    const index_t group_index = c / (2 * kernel_h * kernel_w);
    const index_t group_col_step = channel_per_group * width_col * height_col;
    const index_t group_im_step = channel_per_group / kernel_h / kernel_w * height * width;
    const index_t group_offset_step = 2 * kernel_h * kernel_w * height_col * width_col;
    const index_t col_step = kernel_h * kernel_w;
    const DType* data_col_ptr = data_col + group_index * group_col_step;
    const DType* data_im_ptr = data_im + group_index * group_im_step;
    const DType* data_offset_ptr = data_offset + group_index * group_offset_step;

    index_t cnt = 0;
    const index_t offset_c = c - group_index * 2 * kernel_h * kernel_w;

    for (index_t col_c = (offset_c / 2); col_c < channel_per_group; col_c += col_step) {
      const index_t col_pos = ((col_c * height_col) + h) * width_col + w;
      const index_t bp_dir = offset_c % 2;

      index_t j = (col_pos / width_col / height_col) % kernel_w;
      index_t i = (col_pos / width_col / height_col / kernel_w) % kernel_h;
      index_t w_col = col_pos % width_col;
      index_t h_col = (col_pos / width_col) % height_col;
      index_t w_in = w_col * stride_w - pad_w;
      index_t h_in = h_col * stride_h - pad_h;
      const index_t data_offset_h_ptr = ((2 * (i * kernel_w + j)) *
        height_col + h_col) * width_col + w_col;
      const index_t data_offset_w_ptr = data_offset_h_ptr + height_col * width_col;
      const DType offset_h = data_offset_ptr[data_offset_h_ptr];
      const DType offset_w = data_offset_ptr[data_offset_w_ptr];
      DType inv_h = h_in + i * dilation_h + offset_h;
      DType inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h < 0 || inv_w < 0 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -1;
      }
      const DType weight = get_coordinate_weight_cpu(inv_h, inv_w, height, width,
                                                     data_im_ptr + cnt * height * width,
                                                     width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
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
                                    const DType* data_col,
                                    const DType* data_im,
                                    const DType* data_offset,
                                    const mxnet::TShape& im_shape,
                                    const mxnet::TShape& col_shape,
                                    const mxnet::TShape& kernel_shape,
                                    const mxnet::TShape& pad,
                                    const mxnet::TShape& stride,
                                    const mxnet::TShape& dilation,
                                    const index_t deformable_group,
                                    DType* grad_offset) {
  if (2 == kernel_shape.ndim()) {
    deformable_col2im_coord_cpu(data_col, data_im, data_offset,
                                im_shape[1], im_shape[2], im_shape[3],
                                kernel_shape[0], kernel_shape[1],
                                pad[0], pad[1], stride[0], stride[1],
                                dilation[0], dilation[1],
                                deformable_group,
                                col_shape[1], col_shape[2], grad_offset);
  } else {
    LOG(FATAL) << "not implemented";
  }
}

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./deformable_im2col.cuh"
#endif
#endif  // MXNET_OPERATOR_CONTRIB_NN_DEFORMABLE_IM2COL_H_
