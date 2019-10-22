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
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
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

// caffe_gpu_interp2_kernel overloading with Tensor<xpu, 3, DType>
template<typename xpu, typename Dtype, typename Acctype>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
caffe_gpu_interp2_kernel(const int n,
    const Acctype rheight, const Acctype rwidth,
    const Tensor<xpu, 3, Dtype> data1,
    Tensor<xpu, 3, Dtype> data2,
    ImageLayout layout) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int channels = data1.size(2);
  const int height1 = data1.size(0);
  const int width1 = data1.size(1);
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
    const Acctype h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    for (int c = 0; c < channels; ++c) {
      const Acctype val = h0lambda * (w0lambda * data1[h1][w1][c]
                            + w1lambda * data1[h1][w1+w1p][c])
                            + h1lambda * (w0lambda * data1[h1+h1p][w1][c]
                            + w1lambda * data1[h1+h1p][w1+w1p][c]);
      data2[h2][w2][c] = ScalarConvert<Acctype, Dtype>::to(val);
    }
  }
}

// caffe_gpu_interp2_kernel overloading with Tensor<xpu, 4, DType>
template<typename xpu, typename Dtype, typename Acctype>
__global__ void
__launch_bounds__(cuda::kMaxThreadsPerBlock, 1)
caffe_gpu_interp2_kernel(const int n,
    const Acctype rheight, const Acctype rwidth,
    const Tensor<xpu, 4, Dtype> data1,
    Tensor<xpu, 4, Dtype> data2,
    ImageLayout layout) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_size = (layout == NHWC) ? data1.size(0) : data1.size(0);
  int channels = (layout == NHWC) ? data1.size(3) : data1.size(1);
  int height1 = (layout == NHWC) ? data1.size(1) : data1.size(2);
  int width1 = (layout == NHWC) ? data1.size(2) : data1.size(3);
  int height2 = (layout == NHWC) ? data2.size(1) : data2.size(2);
  int width2 = (layout == NHWC) ? data2.size(2): data2.size(3);

  if (index < n) {
    const int w2 = index % width2;  // 0:width2-1
    const int h2 = index / width2;  // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;

      for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
          if (layout == NHWC) {
            const Dtype val = data1[n][h1][w1][c];
            data2[n][h2][w2][c] = val;
          } else {
            const Dtype val = data1[n][c][h1][w1];
            data2[n][c][h2][w2] = val;
          }
        }
      }
      return;
    }
    //
    const Acctype h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;

    for (auto n = 0; n < batch_size; ++n) {
      for (int c = 0; c < channels; ++c) {
        if (layout == NHWC) {
          const Acctype val = h0lambda * (w0lambda * data1[n][h1][w1][c]
                            + w1lambda * data1[n][h1][w1+w1p][c])
                            + h1lambda * (w0lambda * data1[n][h1+h1p][w1][c]
                            + w1lambda * data1[n][h1+h1p][w1+w1p][c]);
          data2[n][h2][w2][c] = ScalarConvert<Acctype, Dtype>::to(val);
        } else {
          const Acctype val = h0lambda * (w0lambda * data1[n][c][h1][w1]
                            + w1lambda * data1[n][c][h1][w1+w1p])
                            + h1lambda * (w0lambda * data1[n][c][h1+h1p][w1]
                            + w1lambda * data1[n][c][h1+h1p][w1+w1p]);
          data2[n][c][h2][w2] = ScalarConvert<Acctype, Dtype>::to(val);
        }
      }
    }
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_CUH_