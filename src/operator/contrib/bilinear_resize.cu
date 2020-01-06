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
#include "bilinear_resize-inl.cuh"

namespace mxnet {
namespace op {

using namespace mshadow;

// fastSpecializedAtomicAdd adapted from Torch
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/KernelUtils.cuh
template <
  typename Dtype,
  typename std::enable_if<std::is_same<mshadow::half::half_t, Dtype>::value>::type* =
  nullptr>
  __device__ MSHADOW_FORCE_INLINE void fastSpecializedAtomicAdd(
    Dtype* tensor,
    size_t index,
    const size_t numel,
    Dtype value) {
#if (                         \
    (CUDA_VERSION < 10000) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  atomicAdd(
    reinterpret_cast<mshadow::half::half_t*>(tensor) + index,
    static_cast<mshadow::half::half_t>(value));
#else
  bool low_bit = (index % 2 == 0) &&
    (reinterpret_cast<std::uintptr_t>(tensor) % sizeof(__half2) == 0);

  if (low_bit && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(tensor) + index / 2, value2);
  } else if (!low_bit && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(tensor) + index / 2, value2);

  } else {
    atomicAdd(
      reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
  }
#endif
}

template <
  typename Dtype,
  typename std::enable_if<!std::is_same<mshadow::half::half_t, Dtype>::value>::type* =
  nullptr>
  __device__ MSHADOW_FORCE_INLINE void fastSpecializedAtomicAdd(
    Dtype* tensor,
    size_t index,
    const size_t numel,
    Dtype value) {
  atomicAdd(tensor + index, value);
}

template <class Dtype>
__device__ MSHADOW_FORCE_INLINE void fastAtomicAdd(
  Dtype* tensor,
  size_t index,
  const size_t numel,
  Dtype value,
  bool fast_atomics) {
  if (fast_atomics) {
    fastSpecializedAtomicAdd(tensor, index, numel, value);
  } else {
    atomicAdd(tensor + index, value);
  }
}


template<typename xpu, typename Dtype, typename Acctype>
__global__ void like_mode_kernel_backward(const int n,
    Tensor<xpu, 4, Dtype> dataLike) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = dataLike.size(0);
  const int channels = dataLike.size(1);
  const int height = dataLike.size(2);
  const int width = dataLike.size(3);
  if (index < n) {
    const int w = index % width;
    const int h = index / width;
    for (int n = 0; n < batchsize ; n++) {
      for (int c = 0; c < channels; ++c) {
        dataLike[n][c][h][w] = 0;
      }
    }
    return;
  }
}

// caffe_gpu_interp2_kernel_backward adapted from Torch
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu
// Backward (adjoint) operation 1 <- 2 (accumulates)
template<typename xpu, typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel_backward(
  const size_t nc,
  const int height1,
  const int width1,
  const int height2,
  const int width2,
    const Acctype rheight,
  const Acctype rwidth,
  const bool align_corners,
  Dtype* __restrict__ idata,
  const Dtype* __restrict__ odata) {
  const size_t o_numel = nc * width2 * height2;
  const size_t i_numel = nc * width1 * height1;
  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x; index < o_numel;
    index += blockDim.x * gridDim.x) {
    size_t index_temp = index;
    const int w2 = index_temp % width2;  // 0:width2-1
    index_temp /= width2;
    const int h2 = index_temp % height2;  // 0:height2-1
    const size_t nc = index_temp / height2;
    //
    const Acctype h1r = cu_area_pixel_compute_source_index<Acctype>(
      rheight, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = static_cast<Acctype>(1) - h1lambda;
    //
    const Acctype w1r = cu_area_pixel_compute_source_index<Acctype>(
      rwidth, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = static_cast<Acctype>(1) - w1lambda;

    const Dtype d2val = odata[index];
    fastAtomicAdd(
      idata,
      idx(nc, height1, width1, h1, w1),
      i_numel,
      ScalarConvert<Acctype, Dtype>::to(h0lambda * w0lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx(nc, height1, width1, h1, w1 + w1p),
      i_numel,
      ScalarConvert<Acctype, Dtype>::to(h0lambda * w1lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx(nc, height1, width1, h1 + h1p, w1),
      i_numel,
      ScalarConvert<Acctype, Dtype>::to(h1lambda * w0lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx(nc, height1, width1, h1 + h1p, w1 + w1p),
      i_numel,
      ScalarConvert<Acctype, Dtype>::to(h1lambda * w1lambda * d2val),
      true);
  }
}

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateOutput(mshadow::Stream<gpu> *s,
                                           const std::vector<TBlob> &input,
                                           const std::vector<TBlob> &output,
                                         bool align_corners) {
  Tensor<xpu, 4, DType> idata = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> odata = output[0].get<xpu, 4, DType>(s);
  int outputHeight = odata.size(2);
  int outputWidth = odata.size(3);
  int nbatch = idata.size(0);
  int channels = idata.size(1);
  int inputHeight = idata.size(2);
  int inputWidth = idata.size(3);

  const AccReal rheight = cu_area_pixel_compute_scale<AccReal>(
    inputHeight, outputHeight, align_corners);
  const AccReal rwidth = cu_area_pixel_compute_scale<AccReal>(
    inputWidth, outputWidth, align_corners);

  const int num_kernels = nbatch * channels * outputHeight * outputWidth;
  const int num_threads = getNumThreads(inputHeight*inputWidth, false);
  dim3 blocks(static_cast<int>(num_kernels / num_threads) + 1);
  dim3 threads(num_threads);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  caffe_gpu_interp2_kernel<xpu, DType, AccReal>
  <<<blocks, threads , 0, stream>>>(
    nbatch * channels,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    rheight,
    rwidth,
    align_corners,
    idata.dptr_,
    odata.dptr_);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SpatialUpSamplingBilinearUpdateOutput);
}

template<typename xpu, typename DType, typename AccReal>
void SpatialUpSamplingBilinearUpdateGradInput(mshadow::Stream<gpu> *s,
                                              const std::vector<TBlob> &input,
                                              const std::vector<TBlob> &output,
                                              bool modeLike,
                                            bool align_corners) {
  Tensor<xpu, 4, DType> gradOutput = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> gradInput = output[0].get<xpu, 4, DType>(s);
  int outputHeight = gradOutput.size(2);
  int outputWidth = gradOutput.size(3);
  int nbatch = gradInput.size(0);
  int channels = gradInput.size(1);
  int inputHeight = gradInput.size(2);
  int inputWidth = gradInput.size(3);

  const AccReal rheight = cu_area_pixel_compute_scale<AccReal>(
    inputHeight, outputHeight, align_corners);
  const AccReal rwidth = cu_area_pixel_compute_scale<AccReal>(
    inputWidth, outputWidth, align_corners);
  const int num_kernels = nbatch * channels * outputHeight * outputWidth;
  const int num_threads = getNumThreads(inputHeight*inputWidth, false);
  dim3 blocks(static_cast<int>(num_kernels / num_threads) + 1);
  dim3 threads(num_threads);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  caffe_gpu_interp2_kernel_backward<xpu, DType, AccReal>
  <<<blocks, threads, 0, stream>>>(
    nbatch * channels,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    rheight,
    rwidth,
    align_corners,
    gradInput.dptr_,
    gradOutput.dptr_);

  if (modeLike) {
    Tensor<xpu, 4, DType> dataLike = output[1].get<xpu, 4, DType>(s);
    int heightLike = dataLike.size(2);
    int widthLike = dataLike.size(3);
    const int num_kernels_like = heightLike * widthLike;
    const int num_threads_like = getNumThreads(num_kernels_like, false);
    dim3 blocksLike(static_cast<int>(num_kernels_like / num_threads_like) + 1);
    dim3 threadsLike(num_threads_like);
    like_mode_kernel_backward<xpu, DType, AccReal>
    <<<blocksLike, threadsLike, 0, stream>>>(
      num_kernels_like, dataLike);
  }

  MSHADOW_CUDA_POST_KERNEL_CHECK(SpatialUpSamplingBilinearUpdateGradInput);
}

NNVM_REGISTER_OP(_contrib_BilinearResize2D)
.set_attr<FCompute>("FCompute<gpu>", BilinearSampleOpForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_BilinearResize2D)
.set_attr<FCompute>("FCompute<gpu>", BilinearSampleOpBackward<gpu>);
}  // namespace op
}  // namespace mxnet
