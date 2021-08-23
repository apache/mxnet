/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2019 by Contributors
 * \file bilinear_resize-inl.cuh
 * \brief bilinear resize operator cuda implementation
 * \author Hang Zhang, Jake Lee
*/

#ifndef MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_CUH_
#define MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_CUH_

#include <cuda_runtime_api.h>
#include <algorithm>

namespace mxnet {
namespace op {

using namespace mshadow;

enum ImageLayout {
  HWC,
  NHWC,
  NCHW
};

template<typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ MSHADOW_FORCE_INLINE Out to(const In v) { return (Out) v; }
};

// The maximum number of threads in a block
static const unsigned MAX_BLOCK_SIZE = 512U;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static unsigned getNumThreads(int nElem, const bool smaller) {
  unsigned threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  const int maxi = smaller ? 4 : 5;
  for (int i = 0; i != maxi; ++i) {
    if (static_cast<unsigned>(nElem) <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return smaller ? (MAX_BLOCK_SIZE >> 1) : MAX_BLOCK_SIZE;
}

__device__ MSHADOW_FORCE_INLINE size_t
idx(const size_t nc,
  const size_t height,
  const size_t width,
  const size_t y,
  const size_t x) {
  return (nc * height + y) * width + x;
}

template <typename Acctype>
__host__ MSHADOW_FORCE_INLINE static Acctype cu_area_pixel_compute_scale(
  int input_size,
  int output_size,
  bool align_corners) {
  if (output_size > 1) {
    return align_corners
      ? (Acctype)(input_size - 1) / (output_size - 1)
      : (Acctype)input_size / output_size;
  } else {
    return static_cast<Acctype>(0);
  }
}

template <typename Acctype>
__device__ MSHADOW_FORCE_INLINE static Acctype cu_area_pixel_compute_source_index(
  Acctype scale,
  int dst_index,
  bool align_corners,
  bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    Acctype src_idx = scale * (dst_index + static_cast<Acctype>(0.5)) -
      static_cast<Acctype>(0.5);
    // See Note[Follow Opencv resize logic]
    return (!cubic && src_idx < static_cast<Acctype>(0))
      ? static_cast<Acctype>(0)
      : src_idx;
  }
}

// caffe_gpu_interp2_kernel overloading with Tensor<xpu, 4, DType> for NHWC layout
template<typename xpu, typename Dtype, typename Acctype>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
caffe_gpu_interp2_kernel(const int n,
    const Acctype rheight, const Acctype rwidth,
  const bool align_corners,
    const Tensor<xpu, 4, Dtype> data1,
    Tensor<xpu, 4, Dtype> data2) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batch_size = data1.size(0);
  const int height1 = data1.size(1);
  const int width1 = data1.size(2);
  const int height2 = data2.size(1);
  const int width2 = data2.size(2);
  const int channels = data1.size(3);

  if (index < n) {
    const int w2 = index % width2;  // 0:width2-1
    const int h2 = index / width2;  // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
    for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < channels; ++c) {
      const Dtype val = data1[b][h1][w1][c];
      data2[b][h2][w2][c] = val;
    }
    }
      return;
    }
    //
    const Acctype h1r = cu_area_pixel_compute_source_index<Acctype>(
    rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = cu_area_pixel_compute_source_index<Acctype>(
    rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < channels; ++c) {
      const Acctype val = h0lambda * (w0lambda * data1[b][h1][w1][c]
                        + w1lambda * data1[b][h1][w1 + w1p][c])
                        + h1lambda * (w0lambda * data1[b][h1 + h1p][w1][c]
                        + w1lambda * data1[b][h1 + h1p][w1 + w1p][c]);
      data2[b][h2][w2][c] = ScalarConvert<Acctype, Dtype>::to(val);
    }
  }
  }
}

// caffe_gpu_interp2_kernel overloading with Tensor<xpu, 3, DType> for HWC layout
template<typename xpu, typename Dtype, typename Acctype>
__global__ void 
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
caffe_gpu_interp2_kernel(const int n,
  const Acctype rheight, const Acctype rwidth,
  const bool align_corners,
  const Tensor<xpu, 3, Dtype> data1,
  Tensor<xpu, 3, Dtype> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int height1 = data1.size(0);
  const int width1 = data1.size(1);
  const int channels = data1.size(2);
  const int height2 = data2.size(0);
  const int width2 = data2.size(1);

  if (index < n) {
    const int w2 = index % width2;  // 0:width2-1
    const int h2 = index / width2;  // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[h1][w1][c];
          data2[h2][w2][c] = val;
        }
      return;
    }
    //
    const Acctype h1r = cu_area_pixel_compute_source_index<Acctype>(
      rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = cu_area_pixel_compute_source_index<Acctype>(
      rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
      for (int c = 0; c < channels; ++c) {
        const Acctype val = h0lambda * (w0lambda * data1[h1][w1][c]
                          + w1lambda * data1[h1][w1 + w1p][c])
                          + h1lambda * (w0lambda * data1[h1 + h1p][w1][c]
                          + w1lambda * data1[h1 + h1p][w1 + w1p][c]);
        data2[h2][w2][c] = ScalarConvert<Acctype, Dtype>::to(val);
      }
  }
}

// caffe_gpu_interp2_kernel overloading with Tensor<xpu, 4, DType>
template<typename xpu, typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel(
  const size_t nc,
  const int height1,
  const int width1,
  const int height2,
  const int width2,
    const Acctype rheight,
  const Acctype rwidth,
  const bool align_corners,
  const Dtype* __restrict__ idata,
  Dtype* __restrict__ odata) {
  const size_t i_numel = nc * width1 * height1;
  const size_t o_numel = nc * width2 * height2;
  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x; index < o_numel;
    index += blockDim.x * gridDim.x) {
    size_t index_temp = index;
    const int w2 = index_temp % width2;  // 0:width2-1
    index_temp /= width2;
    const int h2 = index_temp % height2;  // 0:height2-1
    const size_t nc = index_temp / height2;

    const Acctype h1r = cu_area_pixel_compute_source_index<Acctype>(
      rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = static_cast<Acctype>(1) - h1lambda;
    //
    const Acctype w1r = cu_area_pixel_compute_source_index<Acctype>(
      rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = static_cast<Acctype>(1) - w1lambda;

        const Acctype val = h0lambda * (w0lambda * *(idata + idx(nc, height1, width1, h1, w1))
                          + w1lambda * *(idata + idx(nc, height1, width1, h1, w1 + w1p)))
                          + h1lambda * (w0lambda * *(idata + idx(nc, height1, width1, h1 + h1p, w1))
                          + w1lambda * *(idata + idx(nc, height1, width1, h1 + h1p, w1 + w1p)));
        *(odata + idx(nc, height2, width2, h2, w2)) = ScalarConvert<Acctype, Dtype>::to(val);
  }
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_CUH_
