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
 * Copyright (c) 2016 by Contributors
 * \file multibox_prior.cu
 * \brief generate multibox prior boxes cuda kernels
 * \author Joshua Zhang
*/

#include "./multibox_prior-inl.h"
#include <mshadow/cuda/tensor_gpu-inl.cuh>

#define MULTIBOXPRIOR_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void AssignPriors(DType *out, const float size,
                             const float sqrt_ratio, const int in_width,
                             const int in_height, const float step_x,
                             const float step_y, const float center_offy,
                             const float center_offx, const int stride,
                             const int offset) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= in_width * in_height) return;
  int r = index / in_width;
  int c = index % in_width;
  float center_x = (c + center_offx) * step_x;
  float center_y = (r + center_offy) * step_y;
  float w = size * in_height / in_width * sqrt_ratio / 2;  // half width
  float h = size / sqrt_ratio / 2;  // half height
  DType *ptr = out + index * stride + 4 * offset;
  *(ptr++) = center_x - w;  // xmin
  *(ptr++) = center_y - h;  // ymin
  *(ptr++) = center_x + w;  // xmax
  *(ptr++) = center_y + h;  // ymax
}
}  // namespace cuda

template<typename DType>
inline void MultiBoxPriorForward(const Tensor<gpu, 2, DType> &out,
                            const std::vector<float> &sizes,
                            const std::vector<float> &ratios,
                            const int in_width, const int in_height,
                            const std::vector<float> &steps,
                            const std::vector<float> &offsets) {
  CHECK_EQ(out.CheckContiguous(), true);
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  DType *out_ptr = out.dptr_;
  const float step_x = steps[1];
  const float step_y = steps[0];
  const float offset_x = offsets[1];
  const float offset_y = offsets[0];
  const int num_sizes = static_cast<int>(sizes.size());
  const int num_ratios = static_cast<int>(ratios.size());

  const int num_thread = cuda::kMaxThreadsPerBlock;
  dim3 dimBlock(num_thread);
  dim3 dimGrid((in_width * in_height - 1) / num_thread + 1);
  cuda::CheckLaunchParam(dimGrid, dimBlock, "MultiBoxPrior Forward");

  const int stride = 4 * (num_sizes + num_ratios - 1);
  int offset = 0;
  // ratio = 1, various sizes
  for (int i = 0; i < num_sizes; ++i) {
    cuda::AssignPriors<DType><<<dimGrid, dimBlock, 0, stream>>>(out_ptr,
      sizes[i], 1.f, in_width, in_height, step_x, step_y, offset_y, offset_x, stride, offset);
    ++offset;
  }
  MULTIBOXPRIOR_CUDA_CHECK(cudaPeekAtLastError());

  // size = sizes[0], various ratios
  for (int j = 1; j < num_ratios; ++j) {
    cuda::AssignPriors<DType><<<dimGrid, dimBlock, 0, stream>>>(out_ptr,
      sizes[0], sqrtf(ratios[j]), in_width, in_height, step_x, step_y,
       offset_y, offset_x, stride, offset);
    ++offset;
  }
  MULTIBOXPRIOR_CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(MultiBoxPriorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxPriorOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
