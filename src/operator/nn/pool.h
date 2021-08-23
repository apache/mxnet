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
 * \file pool.h
 * \brief Function definitions of pooling 1/2/3-D images.
 * We adopted looping 2-D image pixels from Caffe and extended it to 1-D and 3-D cases.
 * \ref https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp
 * \author Jun Wu
 */

#ifndef MXNET_OPERATOR_NN_POOL_H_
#define MXNET_OPERATOR_NN_POOL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <vector>
#include <algorithm>
#include "./pool_utils.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace pool_enum {
enum PoolingOpInputs {kData};
enum PoolingOpOutputs {kOut, kMask};
enum PoolingOpType {kMaxPooling, kAvgPooling, kSumPooling, kLpPooling};
enum PoolingOpPadConventionType {kValid, kFull, kSame};
}  // namespace pool_enum

/*!
 * \brief max pooling cpu function for 1-D images in 'ncw' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType>
inline void pool_max_1d_ncw_cpu(const DType *in_data, const mxnet::TShape &ishape,
                                const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                const mxnet::TShape &pad, const mxnet::TShape &stride,
                                DType *out_data) {
  using mshadow::red::limits::MinValue;
  const int width = ishape[2];
  const int pooled_width = oshape[2];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const index_t in_data_offset = ishape[2];
  const index_t out_data_offset = oshape[2];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * stride_w - pad_w;
        int wend = std::min(wstart + kernel_w, width);
        wstart = std::max(wstart, 0);
        DType max_val = MinValue<DType>();
        for (int w = wstart; w < wend; ++w) {
          if (in_data[w] > max_val) {
            max_val = in_data[w];
          }
        }
        out_data[pw] = max_val;
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
    }
  }
}

/*!
 * \brief max pooling cpu function for 1-D images in 'nwc' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType>
inline void pool_max_1d_nwc_cpu(const DType* in_data, const mxnet::TShape& ishape,
                                const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                                const mxnet::TShape& pad, const mxnet::TShape& stride,
                                DType* out_data) {
  using mshadow::red::limits::MinValue;
  const int width = ishape[1];
  const int pooled_width = oshape[1];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const int features = oshape[2];
  const index_t in_data_offset = ishape[1] * features;
  const index_t out_data_offset = oshape[1] * features;
  std::vector<DType> max_vals(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw * stride_w - pad_w;
      int wend = std::min(wstart + kernel_w, width);
      wstart = std::max(wstart, 0);
      std::fill(max_vals.begin(), max_vals.end(), MinValue<DType>());
      for (int w = wstart; w < wend; ++w) {
        for (index_t c = 0; c < features; ++c) {
          if (in_data[w * features + c] > max_vals[c]) {
            max_vals[c] = in_data[w * features + c];
          }
        }
      }
      for (index_t c = 0; c < features; ++c)
        out_data[pw * features + c] = max_vals[c];
    }
    in_data += in_data_offset;
    out_data += out_data_offset;
  }
}

/*!
 * \brief max pooling cpu function for 2-D images in 'nchw' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType>
inline void pool_max_2d_nchw_cpu(const DType *in_data, const mxnet::TShape &ishape,
                                 const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                 const mxnet::TShape &pad, const mxnet::TShape &stride,
                                 DType *out_data) {
  using mshadow::red::limits::MinValue;
  const int height = ishape[2], width = ishape[3];
  const int pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const index_t in_data_offset = ishape[2] * ishape[3];
  const index_t out_data_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          DType max_val = MinValue<DType>();
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int in_index = h * width + w;
              if (in_data[in_index] > max_val) {
                max_val = in_data[in_index];
              }
            }
          }
          out_data[pool_index] = max_val;
        }
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
    }
  }
}

/*!
 * \brief max pooling cpu function for 2-D images in 'nhwc' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType>
inline void pool_max_2d_nhwc_cpu(const DType* in_data, const mxnet::TShape& ishape,
                                 const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                                 const mxnet::TShape& pad, const mxnet::TShape& stride,
                                 DType* out_data) {
  using mshadow::red::limits::MinValue;
  const int height = ishape[1], width = ishape[2];
  const int pooled_height = oshape[1], pooled_width = oshape[2];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const int features = oshape[3];
  const index_t in_data_offset = ishape[1] * ishape[2] * features;
  const index_t out_data_offset = oshape[1] * oshape[2] * features;
  std::vector<DType> max_vals(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = std::min(hstart + kernel_h, height);
        int wend = std::min(wstart + kernel_w, width);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        const int pool_index = ph * pooled_width + pw;
        std::fill(max_vals.begin(), max_vals.end(), MinValue<DType>());
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int in_index = h * width + w;
            for (index_t c = 0; c < features; ++c) {
              if (in_data[in_index * features + c] > max_vals[c]) {
                max_vals[c] = in_data[in_index * features + c];
              }
            }
          }
        }
        for (index_t c = 0; c < features; ++c)
          out_data[pool_index * features + c] = max_vals[c];
      }
    }
    in_data += in_data_offset;
    out_data += out_data_offset;
  }
}

/*!
 * \brief max pooling cpu function for 3-D images in 'ncdhw' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType>
inline void pool_max_3d_ncdhw_cpu(const DType *in_data, const mxnet::TShape &ishape,
                                  const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                  const mxnet::TShape &pad, const mxnet::TShape &stride,
                                  DType *out_data) {
  using mshadow::red::limits::MinValue;
  const int depth = ishape[2], height = ishape[3], width = ishape[4];
  const int pooled_depth = oshape[2], pooled_height = oshape[3], pooled_width = oshape[4];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const index_t in_data_offset = ishape[2] * ishape[3] * ishape[4];
  const index_t out_data_offset = oshape[2] * oshape[3] * oshape[4];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pd = 0; pd < pooled_depth; ++pd) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int dstart = pd * stride_d - pad_d;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int dend = std::min(dstart + kernel_d, depth);
            int hend = std::min(hstart + kernel_h, height);
            int wend = std::min(wstart + kernel_w, width);
            dstart = std::max(dstart, 0);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            const int pool_index = (pd * pooled_height + ph) * pooled_width + pw;
            DType max_val = MinValue<DType>();
            for (int d = dstart; d < dend; ++d) {
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int in_index = (d * height + h) * width + w;
                  if (in_data[in_index] > max_val) {
                    max_val = in_data[in_index];
                  }
                }
              }
            }
            out_data[pool_index] = max_val;
          }
        }
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
    }
  }
}

/*!
 * \brief max pooling cpu function for 3-D images in 'ndhwc' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType>
inline void pool_max_3d_ndhwc_cpu(const DType* in_data, const mxnet::TShape& ishape,
                                  const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                                  const mxnet::TShape& pad, const mxnet::TShape& stride,
                                  DType* out_data) {
  using mshadow::red::limits::MinValue;
  const int depth = ishape[1], height = ishape[2], width = ishape[3];
  const int pooled_depth = oshape[1], pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const int features = oshape[4];
  const index_t in_data_offset = ishape[1] * ishape[2] * ishape[3] * features;
  const index_t out_data_offset = oshape[1] * oshape[2] * oshape[3] * features;
  std::vector<DType> max_vals(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pd = 0; pd < pooled_depth; ++pd) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int dstart = pd * stride_d - pad_d;
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int dend = std::min(dstart + kernel_d, depth);
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          dstart = std::max(dstart, 0);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int pool_index = (pd * pooled_height + ph) * pooled_width + pw;
          std::fill(max_vals.begin(), max_vals.end(), MinValue<DType>());
          for (int d = dstart; d < dend; ++d) {
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int in_index = (d * height + h) * width + w;
                for (index_t c = 0; c < features; ++c) {
                  if (in_data[in_index * features + c] > max_vals[c]) {
                    max_vals[c] = in_data[in_index * features + c];
                  }
                }
              }
            }
          }
          for (index_t c = 0; c < features; ++c)
            out_data[pool_index * features + c] = max_vals[c];
        }
      }
    }
    in_data += in_data_offset;
    out_data += out_data_offset;
  }
}

/*!
 * \brief avg/sum pooling cpu function for 1-D images in 'ncw' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType, int p = 1>
inline void pool_sum_1d_ncw_cpu(const DType *in_data, const mxnet::TShape &ishape,
                                const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                const mxnet::TShape &pad, const mxnet::TShape &stride,
                                DType *out_data,
                                const bool get_avg = false, const bool count_include_pad = true) {
  using AccType = typename PoolingTypes<DType>::AccType;
  const int width = ishape[2];
  const int pooled_width = oshape[2];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const index_t in_data_offset = ishape[2];
  const index_t out_data_offset = oshape[2];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * stride_w - pad_w;
        int wend = std::min(wstart + kernel_w, width + pad_w);
        int pool_size = (get_avg ? (wend - wstart) : 1);
        wstart = std::max(wstart, 0);
        wend = std::min(wend, width);
        if (get_avg && !count_include_pad) {
          pool_size = (wend - wstart);
        }
        AccType sum = 0;
        for (int w = wstart; w < wend; ++w) {
          sum += a_pow_p<AccType, p>::Map(in_data[w]) / pool_size;
        }
        out_data[pw] = a_root_p<AccType, p>::Map(sum);
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
    }
  }
}

/*!
 * \brief avg/sum pooling cpu function for 1-D images in 'nwc' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType, int p = 1>
inline void pool_sum_1d_nwc_cpu(const DType* in_data, const mxnet::TShape& ishape,
                                const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                                const mxnet::TShape& pad, const mxnet::TShape& stride,
                                DType* out_data,
                                const bool get_avg = false, const bool count_include_pad = true) {
  using AccType = typename PoolingTypes<DType>::AccType;
  const int width = ishape[1];
  const int pooled_width = oshape[1];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const int features = oshape[2];
  const index_t in_data_offset = ishape[1] * features;
  const index_t out_data_offset = oshape[1] * features;
  std::vector<AccType> sums(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw * stride_w - pad_w;
      int wend = std::min(wstart + kernel_w, width + pad_w);
      int pool_size = (get_avg ? (wend - wstart) : 1);
      wstart = std::max(wstart, 0);
      wend = std::min(wend, width);
      if (get_avg && !count_include_pad) {
        pool_size = (wend - wstart);
      }
      std::fill(sums.begin(), sums.end(), 0);
      for (int w = wstart; w < wend; ++w) {
        for (index_t c = 0; c < features; ++c) {
          sums[c] += a_pow_p<AccType, p>::Map(in_data[w * features + c]) / pool_size;
        }
      }
      for (index_t c = 0; c < features; ++c)
        out_data[pw * features + c] = a_root_p<AccType, p>::Map(sums[c]);
    }
    in_data += in_data_offset;
    out_data += out_data_offset;
  }
}

/*!
 * \brief avg/sum pooling cpu function for 2-D images in 'nchw' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType, int p = 1>
inline void pool_sum_2d_nchw_cpu(const DType *in_data, const mxnet::TShape &ishape,
                                 const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                 const mxnet::TShape &pad, const mxnet::TShape &stride,
                                 DType *out_data,
                                 const bool get_avg = false, const bool count_include_pad = true) {
  using AccType = typename PoolingTypes<DType>::AccType;
  const int height = ishape[2], width = ishape[3];
  const int pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const index_t in_data_offset = ishape[2] * ishape[3];
  const index_t out_data_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height + pad_h);
          int wend = std::min(wstart + kernel_w, width + pad_w);
          int pool_size = (get_avg ? (hend - hstart) * (wend - wstart) : 1);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          if (get_avg && !count_include_pad) {
            pool_size = (hend - hstart) * (wend - wstart);
          }
          AccType sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              sum += a_pow_p<AccType, p>::Map(in_data[h*width+w]) / pool_size;
            }
          }
          out_data[ph*pooled_width+pw] = a_root_p<AccType, p>::Map(sum);
        }
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
    }
  }
}

/*!
 * \brief avg/sum pooling cpu function for 2-D images in 'nhwc' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType, int p = 1>
inline void pool_sum_2d_nhwc_cpu(const DType* in_data, const mxnet::TShape& ishape,
                                 const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                                 const mxnet::TShape& pad, const mxnet::TShape& stride,
                                 DType* out_data,
                                 const bool get_avg = false, const bool count_include_pad = true) {
  using AccType = typename PoolingTypes<DType>::AccType;
  const int height = ishape[1], width = ishape[2];
  const int pooled_height = oshape[1], pooled_width = oshape[2];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const int features = oshape[3];
  const index_t in_data_offset = ishape[1] * ishape[2] * features;
  const index_t out_data_offset = oshape[1] * oshape[2] * features;
  std::vector<AccType> sums(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = std::min(hstart + kernel_h, height + pad_h);
        int wend = std::min(wstart + kernel_w, width + pad_w);
        int pool_size = (get_avg ? (hend - hstart) * (wend - wstart) : 1);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        hend = std::min(hend, height);
        wend = std::min(wend, width);
        if (get_avg && !count_include_pad) {
          pool_size = (hend - hstart) * (wend - wstart);
        }
        const int pool_index = ph * pooled_width + pw;
        std::fill(sums.begin(), sums.end(), 0);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int in_index = h * width + w;
            for (index_t c = 0; c < features; ++c) {
              sums[c] += a_pow_p<AccType, p>::Map(in_data[in_index * features + c]) / pool_size;
            }
          }
        }
        for (index_t c = 0; c < features; ++c)
          out_data[pool_index * features + c] = a_root_p<AccType, p>::Map(sums[c]);
      }
    }
    in_data += in_data_offset;
    out_data += out_data_offset;
  }
}

/*!
 * \brief avg/sum pooling cpu function for 3-D images in 'ncdhw' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType, int p = 1>
inline void pool_sum_3d_ncdhw_cpu(const DType *in_data, const mxnet::TShape &ishape,
                                  const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                  const mxnet::TShape &pad, const mxnet::TShape &stride,
                                  DType *out_data,
                                  const bool get_avg = false, const bool count_include_pad = true) {
  using AccType = typename PoolingTypes<DType>::AccType;
  const int depth = ishape[2], height = ishape[3], width = ishape[4];
  const int pooled_depth = oshape[2], pooled_height = oshape[3], pooled_width = oshape[4];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const index_t in_data_offset = ishape[2] * ishape[3] * ishape[4];
  const index_t out_data_offset = oshape[2] * oshape[3] * oshape[4];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pd = 0; pd < pooled_depth; ++pd) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int dstart = pd * stride_d - pad_d;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int dend = std::min(dstart + kernel_d, depth + pad_d);
            int hend = std::min(hstart + kernel_h, height + pad_h);
            int wend = std::min(wstart + kernel_w, width + pad_w);
            int pool_size = (get_avg ? (dend - dstart) * (hend - hstart) * (wend - wstart) : 1);
            dstart = std::max(dstart, 0);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            dend = std::min(dend, depth);
            hend = std::min(hend, height);
            wend = std::min(wend, width);
            if (get_avg && !count_include_pad) {
              pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            }
            AccType sum = 0;
            for (int d = dstart; d < dend; ++d) {
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  sum += a_pow_p<AccType, p>::Map(in_data[(d*height+h)*width+w]) / pool_size;
                }
              }
            }
            out_data[(pd*pooled_height+ph)*pooled_width+pw] = (pool_size == 0) ?
                                                              AccType(nanf("")) :
                                                              a_root_p<AccType, p>::Map(sum);
          }
        }
      }
      in_data += in_data_offset;
      out_data += out_data_offset;
    }
  }
}

/*!
 * \brief avg/sum pooling cpu function for 3-D images in 'ndhwc' layout.
 * Do not call this kernel directly. Use the interface pool().
 */
template<typename DType, int p = 1>
inline void pool_sum_3d_ndhwc_cpu(const DType* in_data, const mxnet::TShape& ishape,
                                  const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                                  const mxnet::TShape& pad, const mxnet::TShape& stride,
                                  DType* out_data,
                                  const bool get_avg = false, const bool count_include_pad = true) {
  using AccType = typename PoolingTypes<DType>::AccType;
  const int depth = ishape[1], height = ishape[2], width = ishape[3];
  const int pooled_depth = oshape[1], pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const int features = oshape[4];
  const index_t in_data_offset = ishape[1] * ishape[2] * ishape[3] * features;
  const index_t out_data_offset = oshape[1] * oshape[2] * oshape[3] * features;
  std::vector<AccType> sums(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pd = 0; pd < pooled_depth; ++pd) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int dstart = pd * stride_d - pad_d;
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int dend = std::min(dstart + kernel_d, depth + pad_d);
          int hend = std::min(hstart + kernel_h, height + pad_h);
          int wend = std::min(wstart + kernel_w, width + pad_w);
          int pool_size = (get_avg ? (dend - dstart) * (hend - hstart) * (wend - wstart) : 1);
          dstart = std::max(dstart, 0);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          dend = std::min(dend, depth);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          if (get_avg && !count_include_pad) {
            pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
          }
          const int pool_index = (pd * pooled_height + ph) * pooled_width + pw;
          std::fill(sums.begin(), sums.end(), 0);
          for (int d = dstart; d < dend; ++d) {
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int in_index = (d * height + h) * width + w;
                for (index_t c = 0; c < features; ++c) {
                  sums[c] += a_pow_p<AccType, p>::Map(in_data[in_index * features + c]) / pool_size;
                }
              }
            }
          }
          for (index_t c = 0; c < features; ++c)
            out_data[pool_index * features + c] = (pool_size == 0) ?
                                                            AccType(nanf("")) :
                                                            a_root_p<AccType, p>::Map(sums[c]);
        }
      }
    }
    in_data += in_data_offset;
    out_data += out_data_offset;
  }
}

/*!
 * \brief max unpooling cpu function for 1-D images in 'ncw' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
inline void unpool_max_1d_ncw_cpu(const DType *out_grad, const DType *in_data,
                                  const DType *out_data, const mxnet::TShape &ishape,
                                  const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                  const mxnet::TShape &pad, const mxnet::TShape &stride,
                                  DType *in_grad) {
  const int width = ishape[2];
  const int pooled_width = oshape[2];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const index_t in_offset = ishape[2];
  const index_t out_offset = oshape[2];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * stride_w - pad_w;
        int wend = std::min(wstart + kernel_w, width);
        wstart = std::max(wstart, 0);
        int max_idx = -1;
        for (int w = wstart; w < wend; ++w) {
          if (in_data[w] == out_data[pw]) {
            max_idx = w;
            break;
          }
        }
        // In the case where pad > 0 and kernel = 1, for example,
        // max_idx can be -1 reaching this step.
        if (max_idx >= 0) {
          in_grad[max_idx] += out_grad[pw];
        }
      }
      in_data += in_offset;
      in_grad += in_offset;
      out_data += out_offset;
      out_grad += out_offset;
    }
  }
}

/*!
 * \brief max unpooling cpu function for 1-D images in 'nwc' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
inline void unpool_max_1d_nwc_cpu(const DType* out_grad, const DType* in_data,
                              const DType* out_data, const mxnet::TShape& ishape,
                              const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                              const mxnet::TShape& pad, const mxnet::TShape& stride,
                              DType* in_grad) {
  const int width = ishape[1];
  const int pooled_width = oshape[1];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const int features = oshape[2];
  const index_t in_offset = ishape[1] * features;
  const index_t out_offset = oshape[1] * features;
  std::vector<int> max_idxs(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw * stride_w - pad_w;
      int wend = std::min(wstart + kernel_w, width);
      wstart = std::max(wstart, 0);
      std::fill(max_idxs.begin(), max_idxs.end(), -1);
      for (index_t c = 0; c < features; ++c) {
        for (int w = wstart; w < wend; ++w) {
          if (in_data[w * features + c] == out_data[pw * features + c]) {
            max_idxs[c] = w;
            break;
          }
        }
      }
      // In the case where pad > 0 and kernel = 1, for example,
      // max_idx can be -1 reaching this step.
      for (index_t c = 0; c < features; ++c) {
        if (max_idxs[c] >= 0) {
          in_grad[max_idxs[c] * features + c] += out_grad[pw * features + c];
        }
      }
    }
    in_data += in_offset;
    in_grad += in_offset;
    out_data += out_offset;
    out_grad += out_offset;
  }
}

/*!
 * \brief max unpooling cpu function for 2-D images in 'nchw' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
inline void unpool_max_2d_nchw_cpu(const DType *out_grad, const DType *in_data,
                                   const DType *out_data, const mxnet::TShape &ishape,
                                   const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                   const mxnet::TShape &pad, const mxnet::TShape &stride,
                                   DType *in_grad) {
  const int height = ishape[2], width = ishape[3];
  const int pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const index_t in_offset = ishape[2] * ishape[3];
  const index_t out_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          int max_idx = -1;
          bool found = false;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int idx = h * width + w;
              if (in_data[idx] == out_data[pool_index]) {
                max_idx = idx;
                found = true;
                break;
              }
            }
            if (found) break;
          }
          // In the case where pad > 0 and kernel = 1, for example,
          // max_idx can be -1 reaching this step.
          if (max_idx >= 0) {
            in_grad[max_idx] += out_grad[pool_index];
          }
        }
      }
      in_data += in_offset;
      in_grad += in_offset;
      out_data += out_offset;
      out_grad += out_offset;
    }
  }
}

/*!
 * \brief max unpooling cpu function for 2-D images in 'nhwc' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
inline void unpool_max_2d_nhwc_cpu(const DType* out_grad, const DType* in_data,
                              const DType* out_data, const mxnet::TShape& ishape,
                              const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                              const mxnet::TShape& pad, const mxnet::TShape& stride,
                              DType* in_grad) {
  const int height = ishape[1], width = ishape[2];
  const int pooled_height = oshape[1], pooled_width = oshape[2];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const int features = oshape[3];
  const index_t in_offset = ishape[1] * ishape[2] * features;
  const index_t out_offset = oshape[1] * oshape[2] * features;
  std::vector<int> max_idxs(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = std::min(hstart + kernel_h, height);
        int wend = std::min(wstart + kernel_w, width);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        const int pool_index = ph * pooled_width + pw;
        std::fill(max_idxs.begin(), max_idxs.end(), -1);
        for (index_t c = 0; c < features; ++c) {
          bool found = false;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int idx = h * width + w;
              if (in_data[idx * features + c] == out_data[pool_index * features + c]) {
                max_idxs[c] = idx;
                found = true;
                break;
              }
            }
            if (found) break;
          }
        }
        // In the case where pad > 0 and kernel = 1, for example,
        // max_idx can be -1 reaching this step.
        for (index_t c = 0; c < features; ++c) {
          if (max_idxs[c] >= 0) {
            in_grad[max_idxs[c] * features + c] += out_grad[pool_index * features + c];
          }
        }
      }
    }
    in_data += in_offset;
    in_grad += in_offset;
    out_data += out_offset;
    out_grad += out_offset;
  }
}

/*!
 * \brief max unpooling cpu function for 3-D images in 'ncdhw' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
inline void unpool_max_3d_ncdhw_cpu(const DType *out_grad, const DType *in_data,
                                    const DType *out_data, const mxnet::TShape &ishape,
                                    const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                    const mxnet::TShape &pad, const mxnet::TShape &stride,
                                    DType *in_grad) {
  const int depth = ishape[2], height = ishape[3], width = ishape[4];
  const int pooled_depth = oshape[2], pooled_height = oshape[3], pooled_width = oshape[4];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const index_t in_offset = ishape[2] * ishape[3] * ishape[4];
  const index_t out_offset = oshape[2] * oshape[3] * oshape[4];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pd = 0; pd < pooled_depth; ++pd) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int dstart = pd * stride_d - pad_d;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int dend = std::min(dstart + kernel_d, depth);
            int hend = std::min(hstart + kernel_h, height);
            int wend = std::min(wstart + kernel_w, width);
            dstart = std::max(dstart, 0);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            const int pool_index = (pd * pooled_height + ph) * pooled_width + pw;
            int max_idx = -1;
            bool found = false;
            for (int d = dstart; d < dend; ++d) {
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int idx = (d * height + h) * width + w;
                  if (in_data[idx] == out_data[pool_index]) {
                    max_idx = idx;
                    found = true;
                    break;
                  }
                }
                if (found) break;
              }
              if (found) break;
            }
            // In the case where pad > 0 and kernel = 1, for example,
            // max_idx can be -1 reaching this step.
            if (max_idx >= 0) {
              in_grad[max_idx] += out_grad[pool_index];
            }
          }
        }
      }
      in_data += in_offset;
      in_grad += in_offset;
      out_data += out_offset;
      out_grad += out_offset;
    }
  }
}

/*!
 * \brief max unpooling cpu function for 3-D images in 'ndhwc' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
inline void unpool_max_3d_ndhwc_cpu(const DType* out_grad, const DType* in_data,
                              const DType* out_data, const mxnet::TShape& ishape,
                              const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                              const mxnet::TShape& pad, const mxnet::TShape& stride,
                              DType* in_grad) {
  const int depth = ishape[1], height = ishape[2], width = ishape[3];
  const int pooled_depth = oshape[1], pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const int features = oshape[4];
  const index_t in_offset = ishape[1] * ishape[2] * ishape[3] * features;
  const index_t out_offset = oshape[1] * oshape[2] * oshape[3] * features;
  std::vector<int> max_idxs(features);
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pd = 0; pd < pooled_depth; ++pd) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int dstart = pd * stride_d - pad_d;
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int dend = std::min(dstart + kernel_d, depth);
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          dstart = std::max(dstart, 0);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int pool_index = (pd * pooled_height + ph) * pooled_width + pw;
          std::fill(max_idxs.begin(), max_idxs.end(), -1);
          for (index_t c = 0; c < features; ++c) {
            bool found = false;
            for (int d = dstart; d < dend; ++d) {
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int idx = (d * height + h) * width + w;
                  if (in_data[idx * features + c] == out_data[pool_index * features + c]) {
                    max_idxs[c] = idx;
                    found = true;
                    break;
                  }
                }
                if (found) break;
              }
              if (found) break;
            }
          }
          // In the case where pad > 0 and kernel = 1, for example,
          // max_idx can be -1 reaching this step.
          for (index_t c = 0; c < features; ++c) {
            if (max_idxs[c] >= 0) {
              in_grad[max_idxs[c] * features + c] += out_grad[pool_index * features + c];
            }
          }
        }
      }
    }
    in_data += in_offset;
    in_grad += in_offset;
    out_data += out_offset;
    out_grad += out_offset;
  }
}

/*!
 * \brief avg/sum unpooling cpu function for 1-D images in 'ncw' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType, int p = 1>
inline void unpool_sum_1d_ncw_cpu(const DType *out_grad, const DType *in_data,
                                  const DType *out_data,
                                  const mxnet::TShape &ishape, const mxnet::TShape &oshape,
                                  const mxnet::TShape &kernel, const mxnet::TShape &pad,
                                  const mxnet::TShape &stride, DType *in_grad,
                                  const bool is_avg = false, const bool count_include_pad = true) {
  const int width = ishape[2];
  const int pooled_width = oshape[2];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const index_t in_grad_offset = ishape[2];
  const index_t out_grad_offset = oshape[2];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * stride_w - pad_w;
        int wend = std::min(wstart + kernel_w, width + pad_w);
        int pool_size = (is_avg ? (wend - wstart) : 1);
        wstart = std::max(wstart, 0);
        wend = std::min(wend, width);
        if (is_avg && !count_include_pad) {
          pool_size = (wend - wstart);
        }
        for (int w = wstart; w < wend; ++w) {
          in_grad[w] += lp_grad<DType, p>::Map(out_grad[pw], in_data[w], out_data[pw]) / pool_size;
        }
      }
      in_grad += in_grad_offset;
      in_data += in_grad_offset;
      out_grad += out_grad_offset;
      out_data += out_grad_offset;
    }
  }
}

/*!
 * \brief avg/sum unpooling cpu function for 1-D images in 'nwc' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType, int p = 1>
inline void unpool_sum_1d_nwc_cpu(const DType* out_grad, const DType* in_data,
                                  const DType *out_data, const mxnet::TShape &ishape,
                                  const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                  const mxnet::TShape &pad, const mxnet::TShape &stride,
                                  DType *in_grad, const bool is_avg = false,
                                  const bool count_include_pad = true) {
  const int width = ishape[1];
  const int pooled_width = oshape[1];
  const int kernel_w = kernel[0];
  const int pad_w = pad[0];
  const int stride_w = stride[0];
  const int features = oshape[2];
  const index_t in_grad_offset = ishape[1] * features;
  const index_t out_grad_offset = oshape[1] * features;
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pw = 0; pw < pooled_width; ++pw) {
      int wstart = pw * stride_w - pad_w;
      int wend = std::min(wstart + kernel_w, width + pad_w);
      int pool_size = (is_avg ? (wend - wstart) : 1);
      wstart = std::max(wstart, 0);
      wend = std::min(wend, width);
      if (is_avg && !count_include_pad) {
        pool_size = (wend - wstart);
      }
      for (int w = wstart; w < wend; ++w) {
        for (index_t c = 0; c < features; ++c) {
          in_grad[w * features + c] +=
              lp_grad<DType, p>::Map(out_grad[pw * features + c],
                                     in_data[w * features + c],
                                     out_data[pw * features + c]) / pool_size;
        }
      }
    }
    in_grad += in_grad_offset;
    in_data += in_grad_offset;
    out_grad += out_grad_offset;
    out_data += out_grad_offset;
  }
}

/*!
 * \brief avg/sum unpooling cpu function for 2-D images in 'nchw' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType, int p = 1>
inline void unpool_sum_2d_nchw_cpu(const DType *out_grad, const DType *in_data,
                                   const DType *out_data, const mxnet::TShape &ishape,
                                   const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                   const mxnet::TShape &pad, const mxnet::TShape &stride,
                                   DType *in_grad, const bool is_avg = false,
                                   const bool count_include_pad = true) {
  const int height = ishape[2], width = ishape[3];
  const int pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int stride_h = stride[0], stride_w = stride[1];
  const index_t in_grad_offset = ishape[2] * ishape[3];
  const index_t out_grad_offset = oshape[2] * oshape[3];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height + pad_h);
          int wend = std::min(wstart + kernel_w, width + pad_w);
          int pool_size = (is_avg ? (hend - hstart) * (wend - wstart) : 1);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          if (is_avg && !count_include_pad) {
            pool_size = (hend - hstart) * (wend - wstart);
          }
          const int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              in_grad[h*width+w] +=
                lp_grad<DType, p>::Map(out_grad[pool_index],
                                       in_data[h*width+w],
                                       out_data[pool_index]) / pool_size;
            }
          }
        }
      }
      in_grad += in_grad_offset;
      in_data += in_grad_offset;
      out_grad += out_grad_offset;
      out_data += out_grad_offset;
    }
  }
}

/*!
 * \brief avg/sum unpooling cpu function for 2-D images in 'nhwc' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType, int p = 1>
inline void unpool_sum_2d_nhwc_cpu(const DType* out_grad, const DType* in_data,
                                   const DType *out_data, const mxnet::TShape &ishape,
                                   const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                   const mxnet::TShape &pad, const mxnet::TShape &stride,
                                   DType *in_grad, const bool is_avg = false,
                                   const bool count_include_pad = true) {
  const int height = ishape[1], width = ishape[2];
  const int pooled_height = oshape[1], pooled_width = oshape[2];
  const int kernel_h = kernel[0], kernel_w = kernel[1];
  const int pad_h = pad[0], pad_w = pad[1];
  const int features = oshape[3];
  const int stride_h = stride[0], stride_w = stride[1];
  const index_t in_grad_offset = ishape[1] * ishape[2] * features;
  const index_t out_grad_offset = oshape[1] * oshape[2] * features;
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = std::min(hstart + kernel_h, height + pad_h);
        int wend = std::min(wstart + kernel_w, width + pad_w);
        int pool_size = (is_avg ? (hend - hstart) * (wend - wstart) : 1);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        hend = std::min(hend, height);
        wend = std::min(wend, width);
        if (is_avg && !count_include_pad) {
          pool_size = (hend - hstart) * (wend - wstart);
        }
        const int pool_index = ph * pooled_width + pw;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int in_index = h * width + w;
            for (index_t c = 0; c < features; ++c) {
              in_grad[in_index * features + c] +=
                  lp_grad<DType, p>::Map(out_grad[pool_index * features + c],
                                         in_data[in_index * features + c],
                                         out_data[pool_index * features + c]) / pool_size;
            }
          }
        }
      }
    }
    in_grad += in_grad_offset;
    in_data += in_grad_offset;
    out_grad += out_grad_offset;
    out_data += out_grad_offset;
  }
}

/*!
 * \brief avg/sum unpooling cpu function for 3-D images in 'ncdhw' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType, int p = 1>
inline void unpool_sum_3d_ncdhw_cpu(const DType *out_grad, const DType *in_data,
                                    const DType *out_data, const mxnet::TShape &ishape,
                                    const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                    const mxnet::TShape &pad, const mxnet::TShape &stride,
                                    DType *in_grad, const bool is_avg = false,
                                    const bool count_include_pad = true) {
  const int depth = ishape[2], height = ishape[3], width = ishape[4];
  const int pooled_depth = oshape[2], pooled_height = oshape[3], pooled_width = oshape[4];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const index_t in_grad_offset = ishape[2] * ishape[3] * ishape[4];
  const index_t out_grad_offset = oshape[2] * oshape[3] * oshape[4];
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (index_t c = 0; c < oshape[1]; ++c) {
      for (int pd = 0; pd < pooled_depth; ++pd) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int dstart = pd * stride_d - pad_d;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int dend = std::min(dstart + kernel_d, depth + pad_d);
            int hend = std::min(hstart + kernel_h, height + pad_h);
            int wend = std::min(wstart + kernel_w, width + pad_w);
            int pool_size = (is_avg ? (dend - dstart) * (hend - hstart) * (wend - wstart) : 1);
            dstart = std::max(dstart, 0);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            dend = std::min(dend, depth);
            hend = std::min(hend, height);
            wend = std::min(wend, width);
            if (is_avg && !count_include_pad) {
              pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            }
            const int pool_index = (pd * pooled_height + ph) * pooled_width + pw;
            for (int d = dstart; d < dend; ++d) {
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  in_grad[(d*height+h)*width+w] +=
                    lp_grad<DType, p>::Map(out_grad[pool_index],
                                           in_data[(d*height+h)*width+w],
                                           out_data[pool_index]) / pool_size;
                }
              }
            }
          }
        }
      }
      in_grad += in_grad_offset;
      in_data += in_grad_offset;
      out_grad += out_grad_offset;
      out_data += out_grad_offset;
    }
  }
}

/*!
 * \brief avg/sum unpooling cpu function for 3-D images in 'ndhwc' layout.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType, int p = 1>
inline void unpool_sum_3d_ndhwc_cpu(const DType* out_grad, const DType* in_data,
                                    const DType *out_data, const mxnet::TShape &ishape,
                                    const mxnet::TShape &oshape, const mxnet::TShape &kernel,
                                    const mxnet::TShape &pad, const mxnet::TShape &stride,
                                    DType *in_grad, const bool is_avg = false,
                                    const bool count_include_pad = true) {
  const int depth = ishape[1], height = ishape[2], width = ishape[3];
  const int pooled_depth = oshape[1], pooled_height = oshape[2], pooled_width = oshape[3];
  const int kernel_d = kernel[0], kernel_h = kernel[1], kernel_w = kernel[2];
  const int pad_d = pad[0], pad_h = pad[1], pad_w = pad[2];
  const int stride_d = stride[0], stride_h = stride[1], stride_w = stride[2];
  const int features = oshape[4];
  const index_t in_grad_offset = ishape[1] * ishape[2] * ishape[3] * features;
  const index_t out_grad_offset = oshape[1] * oshape[2] * oshape[3] * features;
  for (index_t n = 0; n < oshape[0]; ++n) {
    for (int pd = 0; pd < pooled_depth; ++pd) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int dstart = pd * stride_d - pad_d;
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int dend = std::min(dstart + kernel_d, depth + pad_d);
          int hend = std::min(hstart + kernel_h, height + pad_h);
          int wend = std::min(wstart + kernel_w, width + pad_w);
          int pool_size = (is_avg ? (dend - dstart) * (hend - hstart) * (wend - wstart) : 1);
          dstart = std::max(dstart, 0);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          dend = std::min(dend, depth);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          if (is_avg && !count_include_pad) {
            pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
          }
          const int pool_index = (pd * pooled_height + ph) * pooled_width + pw;
          for (int d = dstart; d < dend; ++d) {
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int in_index = (d * height + h) * width + w;
                for (index_t c = 0; c < features; ++c) {
                  in_grad[in_index * features + c] +=
                      lp_grad<DType, p>::Map(out_grad[pool_index * features + c],
                                             in_data[in_index * features + c],
                                             out_data[pool_index * features + c]) / pool_size;
                }
              }
            }
          }
        }
      }
    }
    in_grad += in_grad_offset;
    in_data += in_grad_offset;
    out_grad += out_grad_offset;
    out_data += out_grad_offset;
  }
}

/*!
 * \brief This function serves as an interface for 1/2/3-D pooling operations.
 * \param s context stream defining the device in use is cpu
 * \param in_data pointer of the input tensor data in the format of NCW, NCHW, or NCDHW
 * \param ishape input tensor shape
 * \param oshape output tensor shape
 * \param kernel kernel shape
 * \param pad pad shape
 * \param stride stride shape
 * \param pool_type supported pooling type: max, avg, sum
 * \param req_type operator request type, only support kWriteTo for now
 * \param out_data pointer of the output tensor data in the format of NCW, NCHW, or NCDHW
 * \param p_value value of p for Lp pooling
 */
template<typename DType, int p>
inline void pool(mshadow::Stream<cpu>* s, const DType* in_data, const mxnet::TShape& ishape,
                 const mxnet::TShape& oshape, const mxnet::TShape& kernel, const mxnet::TShape& pad,
                 const mxnet::TShape& stride, const int pool_type, OpReqType req_type,
                 DType* out_data, const bool count_include_pad, int layout) {
  CHECK_EQ(req_type, kWriteTo) << "Only support req=kWriteTo in pooling operations";
  if (kernel.ndim() == 1) {
    if (layout == mshadow::kNWC) {
      if (pool_enum::kMaxPooling == pool_type) {
        pool_max_1d_nwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kAvgPooling == pool_type) {
        pool_sum_1d_nwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data,
                        true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        pool_sum_1d_nwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kLpPooling == pool_type) {
        pool_sum_1d_nwc_cpu<DType, p>(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else if (layout == mshadow::kNCW) {
      if (pool_enum::kMaxPooling == pool_type) {
        pool_max_1d_ncw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kAvgPooling == pool_type) {
        pool_sum_1d_ncw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data,
                            true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        pool_sum_1d_ncw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kLpPooling == pool_type) {
        pool_sum_1d_ncw_cpu<DType, p>(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else {
      LOG(FATAL) << "Unsupported layout, expecting kNCW or kNWC, saw: " << layout;
    }
  } else if (kernel.ndim() == 2) {
    if (layout == mshadow::kNHWC) {
      if (pool_enum::kMaxPooling == pool_type) {
        pool_max_2d_nhwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kAvgPooling == pool_type) {
        pool_sum_2d_nhwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data,
                        true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        pool_sum_2d_nhwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kLpPooling == pool_type) {
        pool_sum_2d_nhwc_cpu<DType, p>(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else if (layout == mshadow::kNCHW) {
      if (pool_enum::kMaxPooling == pool_type) {
        pool_max_2d_nchw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kAvgPooling == pool_type) {
        pool_sum_2d_nchw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data,
                             true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        pool_sum_2d_nchw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kLpPooling == pool_type) {
        pool_sum_2d_nchw_cpu<DType, p>(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else {
      LOG(FATAL) << "Unsupported layout, expecting kNCHW or kNHWC, saw: " << layout;
    }
  } else if (kernel.ndim() == 3) {
    if (layout == mshadow::kNDHWC) {
      if (pool_enum::kMaxPooling == pool_type) {
        pool_max_3d_ndhwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kAvgPooling == pool_type) {
        pool_sum_3d_ndhwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data,
                        true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        pool_sum_3d_ndhwc_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kLpPooling == pool_type) {
        pool_sum_3d_ndhwc_cpu<DType, p>(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else if (layout == mshadow::kNCDHW) {
      if (pool_enum::kMaxPooling == pool_type) {
        pool_max_3d_ncdhw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kAvgPooling == pool_type) {
        pool_sum_3d_ncdhw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data,
                              true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        pool_sum_3d_ncdhw_cpu(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else if (pool_enum::kLpPooling == pool_type) {
        pool_sum_3d_ncdhw_cpu<DType, p>(in_data, ishape, oshape, kernel, pad, stride, out_data);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else {
      LOG(FATAL) << "Unsupported layout, expecting kNCDHW or kNDHWC, saw: " << layout;
    }
  } else {
    LOG(FATAL) << "Unsupported " << kernel.ndim() << "-D pooling";
  }
}

/*!
 * \brief This function serves as an interface for 1/2/3-D unpooling operations.
 * \param s context stream defining the device in use is cpu
 * \param out_grad pointer of the gradient of operator's output tensor
 * \param in_data pointer of the input tensor in the format of NCW, NCHW, or NCDHW
 * \param out_data pointer of the output tensor in the format of NCW, NCHW, or NCDHW
 * \param ishape input tensor shape
 * \param oshape output tensor shape
 * \param kernel kernel shape
 * \param pad pad shape
 * \param stride stride shape
 * \param pool_type supported pooling type: max, avg, sum
 * \param req_type operator request type: kNullOp, kNullWriteInplace, kNullWriteTo, kNullAddTo
 * \param in_grad pointer of the gradient of the operator's input tensor
 * \param p_value value of p for Lp pooling
 */
template<typename DType, int p>
inline void unpool(mshadow::Stream<cpu>* s, const DType* out_grad, const DType* in_data,
                   const DType* out_data, const mxnet::TShape& ishape,
                   const mxnet::TShape& oshape, const mxnet::TShape& kernel,
                   const mxnet::TShape& pad, const mxnet::TShape& stride,
                   const int pool_type, OpReqType req_type, DType* in_grad,
                   const bool count_include_pad, int layout) {
  if (mxnet::kNullOp == req_type) return;
  if (mxnet::kAddTo != req_type) {
    mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, ishape.Size(), in_grad);
  }
  if (kernel.ndim() == 1) {
    if (layout == mshadow::kNWC) {
      if (pool_enum::kMaxPooling == pool_type) {
        unpool_max_1d_nwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                          in_grad);
      } else if (pool_enum::kAvgPooling == pool_type) {
        unpool_sum_1d_nwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                              in_grad, true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        unpool_sum_1d_nwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                          in_grad);
      } else if (pool_enum::kLpPooling == pool_type) {
        unpool_sum_1d_nwc_cpu<DType, p>(out_grad, in_data, out_data, ishape, oshape, kernel, pad,
                                    stride,
                                    in_grad);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else if (layout == mshadow::kNCW) {
      if (pool_enum::kMaxPooling == pool_type) {
        unpool_max_1d_ncw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                              in_grad);
      } else if (pool_enum::kAvgPooling == pool_type) {
        unpool_sum_1d_ncw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                              in_grad,
                              true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        unpool_sum_1d_ncw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                              in_grad);
      } else if (pool_enum::kLpPooling == pool_type) {
        unpool_sum_1d_ncw_cpu<DType, p>(out_grad, in_data, out_data, ishape, oshape, kernel, pad,
                                        stride,
                                        in_grad);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else {
      LOG(FATAL) << "Unsupported layout, expecting kNCW or kNWC, saw: " << layout;
    }
  } else if (kernel.ndim() == 2) {
    if (layout == mshadow::kNHWC) {
      if (pool_enum::kMaxPooling == pool_type) {
        unpool_max_2d_nhwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                          in_grad);
      } else if (pool_enum::kAvgPooling == pool_type) {
        unpool_sum_2d_nhwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                          in_grad,
                          true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        unpool_sum_2d_nhwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                          in_grad);
      } else if (pool_enum::kLpPooling == pool_type) {
        unpool_sum_2d_nhwc_cpu<DType, p>(out_grad, in_data, out_data, ishape, oshape, kernel, pad,
                                    stride,
                                    in_grad);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else if (layout == mshadow::kNCHW) {
      if (pool_enum::kMaxPooling == pool_type) {
        unpool_max_2d_nchw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                               in_grad);
      } else if (pool_enum::kAvgPooling == pool_type) {
        unpool_sum_2d_nchw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                               in_grad,
                               true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        unpool_sum_2d_nchw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                               in_grad);
      } else if (pool_enum::kLpPooling == pool_type) {
        unpool_sum_2d_nchw_cpu<DType, p>(out_grad, in_data, out_data, ishape, oshape, kernel, pad,
                                         stride,
                                         in_grad);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else {
      LOG(FATAL) << "Unsupported layout, expecting kNCHW or kNHWC, saw: " << layout;
    }
  } else if (kernel.ndim() == 3) {
    if (layout == mshadow::kNDHWC) {
      if (pool_enum::kMaxPooling == pool_type) {
        unpool_max_3d_ndhwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                          in_grad);
      } else if (pool_enum::kAvgPooling == pool_type) {
        unpool_sum_3d_ndhwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                                in_grad, true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        unpool_sum_3d_ndhwc_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                          in_grad);
      } else if (pool_enum::kLpPooling == pool_type) {
        unpool_sum_3d_ndhwc_cpu<DType, p>(out_grad, in_data, out_data, ishape, oshape, kernel, pad,
                                    stride,
                                    in_grad);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else if (layout == mshadow::kNCDHW) {
      if (pool_enum::kMaxPooling == pool_type) {
        unpool_max_3d_ncdhw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                                in_grad);
      } else if (pool_enum::kAvgPooling == pool_type) {
        unpool_sum_3d_ncdhw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                                in_grad,
                                true, count_include_pad);
      } else if (pool_enum::kSumPooling == pool_type) {
        unpool_sum_3d_ncdhw_cpu(out_grad, in_data, out_data, ishape, oshape, kernel, pad, stride,
                                in_grad);
      } else if (pool_enum::kLpPooling == pool_type) {
        unpool_sum_3d_ncdhw_cpu<DType, p>(out_grad, in_data, out_data, ishape, oshape, kernel, pad,
                                          stride,
                                          in_grad);
      } else {
        LOG(FATAL) << "Unknown pooling type " << pool_type;
      }
    } else {
      LOG(FATAL) << "Unsupported layout, expecting kNCDHW or kNDHWC, saw: " << layout;
    }
  } else {
    LOG(FATAL) << "Unsupported " << kernel.ndim() << "-D unpooling";
  }
}

}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./pool.cuh"
#endif

#endif  // MXNET_OPERATOR_NN_POOL_H_
