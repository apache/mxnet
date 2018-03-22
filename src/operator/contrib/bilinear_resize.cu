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
 * Copyright (c) 2018 by Contributors
 * \file bilinear_resize.cu
 * \brief bilinear resize operator
 * \author Hang Zhang
*/
#include <cuda_runtime_api.h>
#include <algorithm>
#include "bilinear_resize-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

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

template<typename xpu, typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel(const int n,
    const Acctype rheight, const Acctype rwidth,
    const Tensor<xpu, 4, Dtype> data1,
    Tensor<xpu, 4, Dtype> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.size(0);
  const int channels = data1.size(1);
  const int height1 = data1.size(2);
  const int width1 = data1.size(3);
  const int height2 = data2.size(2);
  const int width2 = data2.size(3);

  if (index < n) {
    const int w2 = index % width2;  // 0:width2-1
    const int h2 = index / width2;  // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++) {
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][h1][w1];
          data2[n][c][h2][w2] = val;
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
    //
    for (int n = 0; n < batchsize ; n++) {
        for (int c = 0; c < channels; ++c) {
        const Acctype val = h0lambda * (w0lambda * data1[n][c][h1][w1]
                            + w1lambda * data1[n][c][h1][w1+w1p])
                            + h1lambda * (w0lambda * data1[n][c][h1+h1p][w1]
                            + w1lambda * data1[n][c][h1+h1p][w1+w1p]);
        data2[n][c][h2][w2] = ScalarConvert<Acctype, Dtype>::to(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template<typename xpu, typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel_backward(const int n,
    const Acctype rheight, const Acctype rwidth,
    Tensor<xpu, 4, Dtype> data1, const Tensor<xpu, 4, Dtype> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.size(0);
  const int channels = data1.size(1);
  const int height1 = data1.size(2);
  const int width1 = data1.size(3);
  const int height2 = data2.size(2);
  const int width2 = data2.size(3);
  if (index < n) {
    const int w2 = index % width2;  // 0:width2-1
    const int h2 = index / width2;  // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++) {
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][h1][w1];
          data1[n][c][h2][w2] += val;
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
    //
    for (int n = 0; n < batchsize ; n++) {
      for (int c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][h2][w2];
        atomicAdd(&data1[n][c][h1][w1],
                  ScalarConvert<Acctype, Dtype>::to(h0lambda * w0lambda * d2val));
        atomicAdd(&data1[n][c][h1][w1+w1p],
                  ScalarConvert<Acctype, Dtype>::to(h0lambda * w1lambda * d2val));
        atomicAdd(&data1[n][c][h1+h1p][w1],
                  ScalarConvert<Acctype, Dtype>::to(h1lambda * w0lambda * d2val));
        atomicAdd(&data1[n][c][h1+h1p][w1+w1p],
                  ScalarConvert<Acctype, Dtype>::to(h1lambda * w1lambda * d2val));
      }
    }
  }
}

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateOutput(mshadow::Stream<gpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> idata = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> odata = output[0].get<xpu, 4, DType>(s);
  int outputHeight = odata.size(2);
  int outputWidth = odata.size(3);
  int inputHeight = idata.size(2);
  int inputWidth = idata.size(3);

  const AccReal rheight = (outputHeight > 1) ? (AccReal)(inputHeight - 1)/
                         (outputHeight - 1) : AccReal(0);
  const AccReal rwidth = (outputWidth > 1) ? (AccReal)(inputWidth - 1)/
                         (outputWidth - 1) : AccReal(0);
  const int num_kernels = outputHeight * outputWidth;
  const int num_threads = getNumThreads(inputHeight*inputWidth, false);
  dim3 blocks(static_cast<int>(num_kernels / num_threads) + 1);
  dim3 threads(num_threads);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  caffe_gpu_interp2_kernel<xpu, DType, AccReal>
  <<<blocks, threads , 0, stream>>>(
    num_kernels, rheight, rwidth, idata, odata);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SpatialUpSamplingBilinearUpdateOutput);
}

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateGradInput(mshadow::Stream<gpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> data1 = output[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> data2 = input[0].get<xpu, 4, DType>(s);
  int height1 = data1.size(2);
  int width1 = data1.size(3);
  int height2 = data2.size(2);
  int width2 = data2.size(3);
  const AccReal rheight = (height2 > 1) ? (AccReal)(height1 - 1)/(height2 - 1) : AccReal(0);
  const AccReal rwidth = (width2 > 1) ? (AccReal)(width1 - 1) / (width2 - 1) : AccReal(0);
  const int num_kernels = height2 * width2;
  const int num_threads = getNumThreads(height1*width1, false);
  dim3 blocks(static_cast<int>(num_kernels / num_threads) + 1);
  dim3 threads(num_threads);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  caffe_gpu_interp2_kernel_backward<xpu, DType, AccReal>
  <<<blocks, threads, 0, stream>>>(
    num_kernels, rheight, rwidth, data1, data2);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SpatialUpSamplingBilinearUpdateGradInput);
}

NNVM_REGISTER_OP(_contrib_BilinearResize2D)
.set_attr<FCompute>("FCompute<gpu>", BilinearSampleOpForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_BilinearResize2D)
.set_attr<FCompute>("FCompute<gpu>", BilinearSampleOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
