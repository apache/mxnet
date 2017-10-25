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
 * \file upsampling_mask_max_2d.cu
 * \brief
 * \author Pengfei Li
*/
#include <vector>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "mxnet_op.h"
#include "../common/cuda_utils.h"
#include "./upsampling_mask_max_2d-inl.h"

namespace mxnet {
namespace op {

template <typename DType>
__global__ void upsample_max_2d_gpu_kernel(const int nthreads, const DType* in_data,
                                       const int channels, const int height, const int width,
                                       const int upsampleed_height, const int upsampleed_width,
                                       DType* out_data, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    int out_offset = (n * channels + c) * upsampleed_height * upsampleed_width;
    int max_idx = mask[index];
    if (max_idx < out_offset) {
      out_data[out_offset+max_idx] = in_data[index];
    }
  }
}

template <typename DType>
__global__ void unupsample_max_2d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const int channels, const int height, const int width,
                                         const int upsampleed_height, const int upsampleed_width,
                                         DType* in_grad, const int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    int out_offset = (n * channels + c) * upsampleed_height * upsampleed_width;
    int max_idx = mask[index];
    if (max_idx < out_offset) {
      atomicAdd(&in_grad[index], out_grad[out_offset+max_idx]);
    }
  }
}

template<typename xpu, typename DType>
void UpSamplingMaskOp<xpu, DType>::upsample_mask_forward(mshadow::Stream<gpu>* s, 
                              const DType* in_data, DType* out_data, int* mask,
                              const TShape& ishape, const TShape& oshape) {
  using namespace mxnet_op;                              
  upsample_max_2d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   ishape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], out_data, mask);
  MSHADOW_CUDA_POST_KERNEL_CHECK(upsample_max_2d_gpu_kernel);                                  
}

template<typename xpu, typename DType>
void UpSamplingMaskOp<xpu, DType>::upsample_mask_backward(mshadow::Stream<gpu>* s, 
                              DType* in_grad, const DType* out_grad, const int* mask,
                              const TShape& ishape, const TShape& oshape) {
  using namespace mxnet_op;                                 
  unupsample_max_2d_gpu_kernel<<<cuda_get_num_blocks(ishape.Size()), mshadow::cuda::kBaseThreadNum, 
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     ishape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], in_grad, mask);
  MSHADOW_CUDA_POST_KERNEL_CHECK(unupsample_max_2d_gpu_kernel);
}

template<>
Operator *CreateOp<gpu>(UpSamplingMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new UpSamplingMaskOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
