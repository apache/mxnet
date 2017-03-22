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
 * \file pool.cuh
 * \brief Function definitions of pooling 1/2/3-D images.
 * We adopted looping 2-D image pixels from Caffe and extended it to 1-D and 3-D cases.
 * \ref https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu
 * \author Jun Wu
 */

#ifndef MXNET_OPERATOR_NN_POOL_CUH_
#define MXNET_OPERATOR_NN_POOL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "../mxnet_op.h"
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {

/*!
 * \brief max pooling gpu kernel for 1-D images.
 * Do not call this kernel directly. Use the interface pool().
 */
template <typename DType>
__global__ void pool_max_1d_gpu_kernel(const int nthreads, const DType* in_data,
                                       const int channels, const int width,
                                       const int pooled_width, const int kernel_w,
                                       const int stride_w, const int pad_w,
                                       DType* out_data) {
  using mshadow::red::limits::MinValue;
  // index is the output image's pixel index in NCW
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int c = (index / pooled_width) % channels;
    const int n = index / pooled_width / channels;
    int wstart = pw * stride_w - pad_w;
    const int wend = min(wstart + kernel_w, width);
    wstart = max(wstart, 0);
    const DType* in_slice =
        in_data + (n * channels + c) * width;
    DType max_val = MinValue<DType>();
    for (int w = wstart; w < wend; ++w) {
      const DType in_val = in_slice[w];
      if (in_val > max_val) {
        max_val = in_val;
      }
    }
    out_data[index] = max_val;
  }
}

/*!
 * \brief max pooling gpu kernel for 2-D images.
 * Do not call this kernel directly. Use the interface pool().
 */
template <typename DType>
__global__ void pool_max_2d_gpu_kernel(const int nthreads, const DType* in_data,
                                       const int channels, const int height, const int width,
                                       const int pooled_height, const int pooled_width,
                                       const int kernel_h, const int kernel_w, const int stride_h,
                                       const int stride_w, const int pad_h, const int pad_w,
                                       DType* out_data) {
  using mshadow::red::limits::MinValue;
  // index is the output image's pixel index in NCHW
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    const DType* in_slice =
        in_data + (n * channels + c) * height * width;
    DType max_val = MinValue<DType>();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const DType in_val = in_slice[h * width + w];
        if (in_val > max_val) {
          max_val = in_val;
        }
      }
    }
    out_data[index] = max_val;
  }
}

/*!
 * \brief max pooling gpu kernel for 3-D images.
 * Do not call this kernel directly. Use the interface pool().
 */
template <typename DType>
__global__ void pool_max_3d_gpu_kernel(const int nthreads, const DType* in_data, const int channels,
                                       const int depth, const int height, const int width,
                                       const int pooled_depth, const int pooled_height,
                                       const int pooled_width, const int kernel_d,
                                       const int kernel_h, const int kernel_w, const int stride_d,
                                       const int stride_h, const int stride_w, const int pad_d,
                                       const int pad_h, const int pad_w,
                                       DType* out_data) {
  using mshadow::red::limits::MinValue;
  // index is the output image's pixel index in NCDHW
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int pd = (index / pooled_width / pooled_height) % pooled_depth;
    const int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    const int n = index / pooled_width / pooled_height / pooled_depth / channels;
    int dstart = pd * stride_d - pad_d;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int dend = min(dstart + kernel_d, depth);
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    const DType* in_slice =
        in_data + (n * channels + c) * depth * height * width;
    DType max_val = MinValue<DType>();
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          const DType in_val = in_slice[(d * height + h) * width + w];
          if (in_val > max_val) {
            max_val = in_val;
          }
        }
      }
    }
    out_data[index] = max_val;
  }
}

/*!
 * \brief avg/sum pooling gpu kernel for 1-D images.
 * Do not call this kernel directly. Use the interface pool().
 */
template <typename DType>
__global__ void pool_sum_1d_gpu_kernel(const int nthreads, const DType* in_data, const int channels,
                                       const int width, const int pooled_width, const int kernel_w,
                                       const int stride_w, const int pad_w,
                                       DType* out_data, bool getAvg = false) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int pw = index % pooled_width;
	  const int c = (index / pooled_width) % channels;
	  const int n = index / pooled_width / channels;
	  int wstart = pw * stride_w - pad_w;
	  int wend = min(wstart + kernel_w, width + pad_w);
	  const int pool_size = (getAvg? (wend - wstart) : 1);
	  wstart = max(wstart, 0);
	  wend = min(wend, width);
	  DType sum = 0;
	  const DType* out_slice =
	 		in_data + (n * channels + c) * width;
    for (int w = wstart; w < wend; ++w) {
      sum += out_slice[w];
    }
    out_data[index] = sum / pool_size;
  }
}

/*!
 * \brief avg/sum pooling gpu kernel for 2-D images.
 * Do not call this kernel directly. Use the interface pool().
 */
template <typename DType>
__global__ void pool_sum_2d_gpu_kernel(const int nthreads, const DType* in_data, const int channels,
                                       const int height, const int width,
                                       const int pooled_height, const int pooled_width,
                                       const int kernel_h, const int kernel_w,
                                       const int stride_h, const int stride_w,
                                       const int pad_h, const int pad_w,
                                       DType* out_data, bool getAvg = false) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int pw = index % pooled_width;
	  const int ph = (index / pooled_width) % pooled_height;
	  const int c = (index / pooled_width / pooled_height) % channels;
	  const int n = index / pooled_width / pooled_height / channels;
	  int hstart = ph * stride_h - pad_h;
	  int wstart = pw * stride_w - pad_w;
	  int hend = min(hstart + kernel_h, height + pad_h);
	  int wend = min(wstart + kernel_w, width + pad_w);
	  const int pool_size = (getAvg? (hend - hstart) * (wend - wstart) : 1);
	  hstart = max(hstart, 0);
	  wstart = max(wstart, 0);
	  hend = min(hend, height);
	  wend = min(wend, width);
	  DType sum = 0;
	  const DType* out_slice =
	 		in_data + (n * channels + c) * height * width;
	  for (int h = hstart; h < hend; ++h) {
		  for (int w = wstart; w < wend; ++w) {
		    sum += out_slice[h * width + w];
		  }
	  }
    out_data[index] = sum / pool_size;
  }
}

/*!
 * \brief avg/sum pooling gpu kernel for 3-D images.
 * Do not call this kernel directly. Use the interface pool().
 */
template <typename DType>
__global__ void pool_sum_3d_gpu_kernel(const int nthreads, const DType* in_data, const int channels,
                                       const int depth, const int height, const int width,
                                       const int pooled_depth, const int pooled_height,
                                       const int pooled_width, const int kernel_d,
                                       const int kernel_h, const int kernel_w,
                                       const int stride_d, const int stride_h, const int stride_w,
                                       const int pad_d, const int pad_h, const int pad_w,
                                       DType* out_data, bool getAvg = false) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int pw = index % pooled_width;
	  const int ph = (index / pooled_width) % pooled_height;
    const int pd = (index / pooled_width / pooled_height) % pooled_depth;
	  const int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
	  const int n = index / pooled_width / pooled_height / pooled_depth / channels;
    int dstart = pd * stride_d - pad_d;
	  int hstart = ph * stride_h - pad_h;
	  int wstart = pw * stride_w - pad_w;
    int dend = min(dstart + kernel_d, depth + pad_d);
	  int hend = min(hstart + kernel_h, height + pad_h);
	  int wend = min(wstart + kernel_w, width + pad_w);
	  const int pool_size = (getAvg? (dend - dstart) * (hend - hstart) * (wend - wstart) : 1);
    dstart = max(dstart, 0);
	  hstart = max(hstart, 0);
	  wstart = max(wstart, 0);
    dend = min(dend, depth);
	  hend = min(hend, height);
	  wend = min(wend, width);
	  DType sum = 0;
	  const DType* out_slice =
	 		in_data + (n * channels + c) * depth * height * width;
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          sum += out_slice[(d * height + h) * width + w];
        }
      }
    }
    out_data[index] = sum / pool_size;
  }
}

/*!
 * \brief max unpooling gpu kernel for 1-D images.
 * Do not call this kernel directly. Use the interface unpool().
 */
template <typename DType>
__global__ void unpool_max_1d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const DType* in_data, const DType* out_data,
                                         const int channels, const int width,
                                         const int pooled_width, const int kernel_w,
                                         const int stride_w, const int pad_w,
                                         DType* in_grad) {
  // index is the output image's pixel index in NCHW
  // the order has to be consistent with pooling max
  // to avoid adding out_grad to the wrong in_grad
  // in the case where there are multiple max pixels
  // covered by a kernel window
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int c = (index / pooled_width) % channels;
    const int n = index / pooled_width / channels;
    int wstart = pw * stride_w - pad_w;
    const int wend = min(wstart + kernel_w, width);
    wstart = max(wstart, 0);
    // in data/grad offset batch and channel dims
    int in_offset = (n * channels + c) * width;
    const DType* in_data_slice = in_data + in_offset;
    int max_idx = -1;
    DType max_val = out_data[index];
    for (int w = wstart; w < wend; ++w) {
      if (in_data_slice[w] == max_val) {
        max_idx = w;
        break;
      }
    }

    // In the case where pad > 0 and kernel = 1, for example,
    // max_idx can be -1 reaching this step.
    if (max_idx >= 0) {
      atomicAdd(&in_grad[in_offset+max_idx], out_grad[index]);
    }
  }
}

/*!
 * \brief max unpooling gpu kernel for 2-D images.
 * Do not call this kernel directly. Use the interface unpool().
 */
template <typename DType>
__global__ void unpool_max_2d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const DType* in_data, const DType* out_data,
                                         const int channels, const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         DType* in_grad) {
  // index is the output image's pixel index in NCHW
  // the order has to be consistent with pooling max
  // to avoid adding out_grad to the wrong in_grad
  // in the case where there are multiple max pixels
  // covered by a kernel window
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    // in data/grad offset batch and channel dims
    int in_offset = (n * channels + c) * height * width;
    const DType* in_data_slice = in_data + in_offset;
    int max_idx = -1;
    DType max_val = out_data[index];
    bool found = false;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int idx = h * width + w;
        if (in_data_slice[idx] == max_val) {
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
      atomicAdd(&in_grad[in_offset+max_idx], out_grad[index]);
    }
  }
}

/*!
 * \brief max unpooling gpu kernel for 3-D images.
 * Do not call this kernel directly. Use the interface unpool().
 */
template <typename DType>
__global__ void unpool_max_3d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const DType* in_data, const DType* out_data,
                                         const int channels, const int depth, const int height,
                                         const int width, const int pooled_depth,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_d, const int kernel_h,
                                         const int kernel_w, const int stride_d,
                                         const int stride_h, const int stride_w, const int pad_d,
                                         const int pad_h, const int pad_w,
                                         DType* in_grad) {
  // index is the output image's pixel index in NCDHW
  // the order has to be consistent with pooling max
  // to avoid adding out_grad to the wrong in_grad
  // in the case where there are multiple max pixels
  // covered by a kernel window
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int pd = (index / pooled_width / pooled_height) % pooled_depth;
    const int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    const int n = index / pooled_width / pooled_height / pooled_depth / channels;
    int dstart = pd * stride_d - pad_d;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int dend = min(dstart + kernel_d, depth);
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    // in data/grad offset batch and channel dims
    int in_offset = (n * channels + c) * depth * height * width;
    const DType* in_data_slice = in_data + in_offset;
    int max_idx = -1;
    DType max_val = out_data[index];
    bool found = false;
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          const int idx = (d * height + h) * width + w;
          if (in_data_slice[idx] == max_val) {
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
      atomicAdd(&in_grad[in_offset+max_idx], out_grad[index]);
    }
  }
}

/*!
 * \brief avg/sum unpooling gpu kernel for 1-D images.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
__global__ void unpool_sum_1d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const int channels, const int width,
                                         const int pooled_width, const int kernel_w,
                                         const int stride_w, const int pad_w,
                                         DType* in_grad, bool isAvg = false) {
  // index is the input image index in NCW
  CUDA_KERNEL_LOOP(index, nthreads) {
	  // find out the local index
	  // find out the local offset
	  const int w = index % width + pad_w;
	  const int c = (index / width) % channels;
	  const int n = index / width / channels;
	  const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
	  const int pwend = min(w / stride_w + 1, pooled_width);
	  DType gradient = 0;
	  const DType* out_grad_slice =
      out_grad + (n * channels + c) * pooled_width;
    for (int pw = pwstart; pw < pwend; ++pw) {
      // figure out the pooling size
      int wstart = pw * stride_w - pad_w;
      int wend = min(wstart + kernel_w, width + pad_w);
      int pool_size = (isAvg? (wend - wstart) : 1);
      gradient += out_grad_slice[pw] / pool_size;
    }
    // if req=kWriteTo, in_grad has already been assigned zero values in unpool()
    // use "+=" here instead of "=" to accommodate when req=kAddTo
	  in_grad[index] += gradient;
  }
}

/*!
 * \brief avg/sum unpooling gpu kernel for 2-D images.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
__global__ void unpool_sum_2d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const int channels, const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         DType* in_grad, bool isAvg = false) {
  // index is the input image index in NCHW
  CUDA_KERNEL_LOOP(index, nthreads) {
	  // find out the local index
	  // find out the local offset
	  const int w = index % width + pad_w;
	  const int h = (index / width) % height + pad_h;
	  const int c = (index / width / height) % channels;
	  const int n = index / width / height / channels;
	  const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
	  const int phend = min(h / stride_h + 1, pooled_height);
	  const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
	  const int pwend = min(w / stride_w + 1, pooled_width);
	  DType gradient = 0;
	  const DType* out_grad_slice =
      out_grad + (n * channels + c) * pooled_height * pooled_width;
	  for (int ph = phstart; ph < phend; ++ph) {
	 	  for (int pw = pwstart; pw < pwend; ++pw) {
		    // figure out the pooling size
			  int hstart = ph * stride_h - pad_h;
			  int wstart = pw * stride_w - pad_w;
			  int hend = min(hstart + kernel_h, height + pad_h);
			  int wend = min(wstart + kernel_w, width + pad_w);
			  int pool_size = (isAvg? (hend - hstart) * (wend - wstart) : 1);
			  gradient += out_grad_slice[ph * pooled_width + pw] / pool_size;
		  }
	  }
    // if req=kWriteTo, in_grad has already been assigned zero values in unpool()
    // use "+=" here instead of "=" to accommodate when req=kAddTo
	  in_grad[index] += gradient;
  }
}

/*!
 * \brief avg/sum unpooling gpu kernel for 3-D images.
 * Do not call this kernel directly. Use the interface unpool().
 */
template<typename DType>
__global__ void unpool_sum_3d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const int channels, const int depth, const int height,
                                         const int width, const int pooled_depth,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_d, const int kernel_h,
                                         const int kernel_w, const int stride_d, const int stride_h,
                                         const int stride_w, const int pad_d, const int pad_h,
                                         const int pad_w, DType* in_grad, bool isAvg = false) {
  // index is the input image index in NCDHW
  CUDA_KERNEL_LOOP(index, nthreads) {
	  // find out the local index
	  // find out the local offset
	  const int w = index % width + pad_w;
	  const int h = (index / width) % height + pad_h;
    const int d = (index / width / height) % depth + pad_d;
	  const int c = (index / width / height / depth) % channels;
	  const int n = index / width / height / depth / channels;
    const int pdstart = (d < kernel_d) ? 0 : (d - kernel_d) / stride_d + 1;
    const int pdend = min(d / stride_d + 1, pooled_depth);
	  const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
	  const int phend = min(h / stride_h + 1, pooled_height);
	  const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
	  const int pwend = min(w / stride_w + 1, pooled_width);
	  DType gradient = 0;
	  const DType* out_grad_slice =
      out_grad + (n * channels + c) * pooled_depth * pooled_height * pooled_width;
    for (int pd = pdstart; pd < pdend; ++pd) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int dstart = pd * stride_d - pad_d;
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int dend = min(dstart + kernel_d, depth + pad_d);
          int hend = min(hstart + kernel_h, height + pad_h);
          int wend = min(wstart + kernel_w, width + pad_w);
          int pool_size = (isAvg? (dend - dstart) * (hend - hstart) * (wend - wstart) : 1);
          gradient += out_grad_slice[(pd * pooled_height + ph) * pooled_width + pw] / pool_size;
        }
      }
    }
    // if req=kWriteTo, in_grad has already been assigned zero values in unpool()
    // use "+=" here instead of "=" to accommodate when req=kAddTo
	  in_grad[index] += gradient;
  }
}

/*!
 * \brief This function serves as an interface for 1/2/3-D pooling operations.
 * \param s context stream defining the device in use is gpu
 * \param in_data pointer of the input tensor data in the format of NCW, NCHW, or NCDHW
 * \param ishape input tensor shape
 * \param oshape output tensor shape
 * \param kernel kernel shape
 * \param pad pad shape
 * \param stride stride shape
 * \param pool_type supported pooling type: max, avg, sum
 * \param req_type operator request type, only support kWriteTo for now
 * \param out_data pointer of the output tensor data in the format of NCW, NCHW, or NCDHW
 */
template<typename DType>
inline void pool(mshadow::Stream<gpu>* s, const DType* in_data, const TShape& ishape,
                 const TShape& oshape, const TShape& kernel, const TShape& pad,
                 const TShape& stride, const int pool_type, OpReqType req_type,
                 DType* out_data) {
  CHECK_EQ(req_type, kWriteTo) << "Only support req=kWriteTo in pooling operations";
  using namespace mxnet_op;
  if (kernel.ndim() == 1) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_max_1d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2],
                                   oshape[2], kernel[0], stride[0], pad[0], out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_max_1d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_1d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], oshape[2],
                                   kernel[0], stride[0], pad[0], out_data, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_1d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_1d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], oshape[2],
                                   kernel[0], stride[0], pad[0], out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_1d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  } else if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_max_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_max_2d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], out_data, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_2d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_2d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  } else if (kernel.ndim() == 3) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_max_3d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   ishape[4], oshape[2], oshape[3], oshape[4],
                                   kernel[0], kernel[1], kernel[2], stride[0],
                                   stride[1], stride[2], pad[0], pad[1], pad[2], out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_max_3d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_3d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   ishape[4], oshape[2], oshape[3], oshape[4], kernel[0],
                                   kernel[1], kernel[2], stride[0], stride[1], stride[2],
                                   pad[0], pad[1], pad[2], out_data, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_3d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      pool_sum_3d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   ishape[4], oshape[2], oshape[3], oshape[4], kernel[0],
                                   kernel[1], kernel[2], stride[0], stride[1], stride[2],
                                   pad[0], pad[1], pad[2], out_data);
      MSHADOW_CUDA_POST_KERNEL_CHECK(pool_sum_3d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  }
}

/*!
 * \brief This function serves as an interface for 1/2/3-D unpooling operations.
 * \param s context stream defining the device in use is gpu
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
 */
template<typename DType>
inline void unpool(mshadow::Stream<gpu>* s, const DType* out_grad, const DType* in_data,
                   const DType* out_data, const TShape& ishape, const TShape& oshape,
                   const TShape& kernel, const TShape& pad, const TShape& stride,
                   const int pool_type, OpReqType req_type, DType* in_grad) {
  if (mxnet::kNullOp == req_type) return;
  if (mxnet::kAddTo != req_type) {
    mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, ishape.Size(), in_grad);
  }
  using namespace mxnet_op;
  if (kernel.ndim() == 1) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_max_1d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     oshape.Size(), out_grad, in_data, out_data,
                                     ishape[1], ishape[2], oshape[2], kernel[0], stride[0], pad[0],
                                     in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_max_1d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_1d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], oshape[2], kernel[0],
                                     stride[0], pad[0], in_grad, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_1d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_1d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], oshape[2], kernel[0],
                                     stride[0], pad[0], in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_1d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  } else  if (kernel.ndim() == 2) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_max_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     oshape.Size(), out_grad, in_data, out_data,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_max_2d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], in_grad, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_2d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_2d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_2d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  } else if (kernel.ndim() == 3) {
    if (pool_enum::kMaxPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_max_3d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     oshape.Size(), out_grad, in_data, out_data,
                                     ishape[1], ishape[2], ishape[3], ishape[4],
                                     oshape[2], oshape[3], oshape[4], kernel[0], kernel[1],
                                     kernel[2], stride[0], stride[1], stride[2],
                                     pad[0], pad[1], pad[2], in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_max_3d_gpu_kernel);
    } else if (pool_enum::kAvgPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_3d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3], ishape[4],
                                     oshape[2], oshape[3], oshape[4], kernel[0], kernel[1],
                                     kernel[2], stride[0], stride[1], stride[2], pad[0], pad[1],
                                     pad[2], in_grad, true);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_3d_gpu_kernel);
    } else if (pool_enum::kSumPooling == pool_type) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      unpool_sum_3d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3], ishape[4],
                                     oshape[2], oshape[3], oshape[4], kernel[0], kernel[1],
                                     kernel[2], stride[0], stride[1], stride[2], pad[0], pad[1],
                                     pad[2], in_grad);
      MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_sum_3d_gpu_kernel);
    } else {
      LOG(FATAL) << "Unknown pooling type " << pool_type;
    }
  } else {
    LOG(FATAL) << "Unsupported " << kernel.ndim() << "-D unpooling";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_POOL_CUH_
