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
 * \file corner_pooling.cu
 * \brief corner pooling operator
 * \author Jiajie Tang
*/

#include "./corner_pooling-inl.h"

#include <vector>

#include <mshadow/cuda/tensor_gpu-inl.cuh>
#include <mshadow/tensor.h>


namespace mxnet {
namespace op {

template<typename DType>
__global__ void CornerPoolingForwardTBKernel(const int count,
        const DType *in_data, const int batch,
        const int channel,  const int height,
        const int width, DType *out_data,
        int h_step, int h_start, int h_end) {
  for (int index = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    using mshadow::red::limits::MinValue;
    const int b = index / (channel * width);
    const int c = index % (channel * width) / width;
    const int w = index % (channel * width) % width;
    DType max_val = MinValue<DType>();
    in_data += (b*channel + c)*height*width + w;
    out_data += (b*channel + c)*height*width + w;

    for (int h{h_start}; h != h_end; h += h_step) {
      const int index = h * width;
      max_val = max_val > in_data[index] ? max_val : in_data[index];
      out_data[index] = max_val;
    }
  }
}


template<typename DType>
__global__ void CornerPoolingBackwardTBKernel(const int count,
        const DType *out_data, const int batch,
        const int channel,  const int height,
        const int width, const DType *out_grad, DType *in_grad,
        int h_step, int h_start, int h_end) {
  for (int index = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int b = index / (channel * width);
    const int c = index % (channel * width) / width;
    const int w = index % (channel * width) % width;

    out_data += (b*channel + c)*height*width + w;
    out_grad += (b*channel + c)*height*width + w;
    in_grad += (b*channel + c)*height*width + w;

    int max_h_idx = h_start;
    for (int h{h_start}; h != h_end; h += h_step) {
      const int index = h * width;
      if (out_data[index] != out_data[max_h_idx]) {
        max_h_idx = index;
      }
      in_grad[max_h_idx] += out_grad[index];
    }
  }
}


template<typename DType>
__global__ void CornerPoolingForwardLRKernel(const int count,
        const DType *in_data, const int batch,
        const int channel, const int height,
        const int width, DType *out_data,
        int w_step, int w_start, int w_end) {
  for (int index = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    using mshadow::red::limits::MinValue;
    const int b = index / (channel * height);
    const int c = index % (channel * height) / height;
    const int h = index % (channel * height) % height;
    DType max_val = MinValue<DType>();
    in_data += ((b*channel + c)*height + h)*width;
    out_data += ((b*channel + c)*height + h)*width;

    for (int w{w_start}; w != w_end; w += w_step) {
      const int index = w;
      max_val = max_val > in_data[index] ? max_val : in_data[index];
      out_data[index] = max_val;
    }
  }
}


template<typename DType>
__global__ void CornerPoolingBackwardLRKernel(const int count,
        const DType *out_data, const int batch,
        const int channel, const int height,
        const int width, const DType *out_grad, DType *in_grad,
        int w_step, int w_start, int w_end) {
  for (int index = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int b = index / (channel * height);
    const int c = index % (channel * height) / height;
    const int h = index % (channel * height) % height;

    out_data += ((b*channel + c)*height + h)*width;
    out_grad += ((b*channel + c)*height + h)*width;
    in_grad += ((b*channel + c)*height + h)*width;

    int max_w_idx = w_start;
    for (int w{w_start}; w != w_end; w += w_step) {
      const int index = w;
      if (out_data[index] != out_data[max_w_idx]) {
        max_w_idx = index;
      }
      in_grad[max_w_idx] += out_grad[index];
    }
  }
}



template<typename DType>
inline void corner_pool(mshadow::Stream<gpu> *s, const DType *in_data,
        const TShape &ishape, const int corner_pooling_type,
        OpReqType req_type, DType *out_data) {
  using mshadow::red::limits::MinValue;
  CHECK_EQ(req_type, kWriteTo)
      << "Only support req=kWriteTo in pooling operations";
  int height = ishape[2], width = ishape[3];
  if (corner_pooling_type == 0 || corner_pooling_type == 1) {
    // top or bottom
    int h_end = 0, h_start = 0, h_step = 0;
    if (corner_pooling_type == 0) {
      h_step = -1;
      h_start = height - 1;
      h_end = -1;
    } else {
      h_step = +1;
      h_start = 0;
      h_end = height;
    }
    const int count = ishape[0] * ishape[1] * width;
    const int gridSize = (count + mshadow::cuda::kMaxThreadsPerBlock - 1) /
                         mshadow::cuda::kMaxThreadsPerBlock;
    dim3 dimGrid(mshadow::cuda::kMaxGridDim,
                 (gridSize + mshadow::cuda::kMaxGridDim - 1) /
                  mshadow::cuda::kMaxGridDim);
    dim3 dimBlock(mshadow::cuda::kMaxThreadsPerBlock);
    mshadow::cuda::CheckLaunchParam(dimGrid,
            dimBlock, "Corner Pooling Forward");
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    CornerPoolingForwardTBKernel<DType><<<  dimGrid, dimBlock, 0, stream>>>
        (count,
         in_data, ishape[0], ishape[1], ishape[2], ishape[3], out_data,
         h_step, h_start, h_end);
    MSHADOW_CUDA_POST_KERNEL_CHECK(CornerPoolingForwardTBKernel);
  } else if (corner_pooling_type == 2 || corner_pooling_type == 3) {
    // left or right
    int w_end = 0, w_start = 0, w_step = 0;
    if (corner_pooling_type == 2) {
      w_step = -1;
      w_start = width - 1;
      w_end = -1;
    } else {
      w_step = +1;
      w_start = 0;
      w_end = width;
    }
    const int count = ishape[0] * ishape[1] * height;
    const int gridSize = (count + mshadow::cuda::kMaxThreadsPerBlock - 1) /
                          mshadow::cuda::kMaxThreadsPerBlock;
    dim3 dimGrid(mshadow::cuda::kMaxGridDim,
                 (gridSize + mshadow::cuda::kMaxGridDim - 1) /
                  mshadow::cuda::kMaxGridDim);
    dim3 dimBlock(mshadow::cuda::kMaxThreadsPerBlock);
    mshadow::cuda::CheckLaunchParam(dimGrid,
            dimBlock, "Corner Pooling Forward");
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    CornerPoolingForwardLRKernel<DType><<<dimGrid, dimBlock, 0, stream>>>
        (count,
         in_data, ishape[0], ishape[1], ishape[2], ishape[3], out_data,
         w_step, w_start, w_end);
    MSHADOW_CUDA_POST_KERNEL_CHECK(CornerPoolingForwardLRKernel);
  } else {
    LOG(FATAL) << "Unsupported corner pooling type";
  }
}

template<typename DType>
inline void corner_pool_grad(mshadow::Stream<gpu> *s,
                   const DType *out_grad, const DType *in_data,
                   const DType *out_data, const TShape &ishape,
                   const int corner_pooling_type, OpReqType req_type,
                   DType *in_grad) {
  const int height = ishape[2], width = ishape[3];
  if (corner_pooling_type == 0 || corner_pooling_type == 1) {
    // top or bottom
    int h_end = 0, h_start = 0, h_step = 0;
    if (corner_pooling_type == 0) {
      h_step = -1;
      h_start = height - 1;
      h_end = -1;
    } else {
      h_step = +1;
      h_start = 0;
      h_end = height;
    }

    const int count = ishape[0] * ishape[1] * width;
    const int gridSize = (count + mshadow::cuda::kMaxThreadsPerBlock - 1) /
                          mshadow::cuda::kMaxThreadsPerBlock;
    dim3 dimGrid(mshadow::cuda::kMaxGridDim,
                 (gridSize + mshadow::cuda::kMaxGridDim - 1) /
                 mshadow::cuda::kMaxGridDim);
    dim3 dimBlock(mshadow::cuda::kMaxThreadsPerBlock);
    mshadow::cuda::CheckLaunchParam(dimGrid,
            dimBlock, "Corner Pooling Backward");
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    CornerPoolingBackwardTBKernel<DType><<< dimGrid, dimBlock, 0, stream>>>
        (count,
         out_data, ishape[0], ishape[1], ishape[2], ishape[3],
         out_grad, in_grad, h_step, h_start, h_end);
    MSHADOW_CUDA_POST_KERNEL_CHECK(CornerPoolingBackwardTBKernel);
  } else if (corner_pooling_type == 2 || corner_pooling_type == 3) {
    // left or right
    int w_end = 0, w_start = 0, w_step = 0;
    if (corner_pooling_type == 2) {
      w_step = -1;
      w_start = width - 1;
      w_end = -1;
    } else {
      w_step = +1;
      w_start = 0;
      w_end = width;
    }
    const int count = ishape[0] * ishape[1] * height;
    const int gridSize = (count + mshadow::cuda::kMaxThreadsPerBlock - 1) /
                          mshadow::cuda::kMaxThreadsPerBlock;
    dim3 dimGrid(mshadow::cuda::kMaxGridDim,
                 (gridSize + mshadow::cuda::kMaxGridDim - 1) /
                 mshadow::cuda::kMaxGridDim);
    dim3 dimBlock(mshadow::cuda::kMaxThreadsPerBlock);
    mshadow::cuda::CheckLaunchParam(dimGrid,
            dimBlock, "Corner Pooling Backward");
    cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
    CornerPoolingBackwardLRKernel<DType><<<dimGrid, dimBlock, 0, stream>>>
        (count,
         out_data, ishape[0], ishape[1], ishape[2], ishape[3],
         out_grad, in_grad, w_step, w_start, w_end);
    MSHADOW_CUDA_POST_KERNEL_CHECK(CornerPoolingBackwardLRKernel);
  }
}


NNVM_REGISTER_OP(CornerPooling)
.set_attr<FCompute>("FCompute<gpu>", CornerPoolingCompute<gpu>);

NNVM_REGISTER_OP(_backward_CornerPooling)
.set_attr<FCompute>("FCompute<gpu>", CornerPoolingGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
