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
 * \file pooling_mask_max_2d.cu
 * \brief
 * \author Pengfei Li
*/

#include <vector>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "mxnet_op.h"
#include "../common/cuda_utils.h"
#include "./pooling_mask_max_2d-inl.h"

namespace mxnet {
namespace op {

template <typename DType>
__global__ void pool_max_2d_gpu_kernel(const int nthreads, const DType* in_data,
                                       const int channels, const int height, const int width,
                                       const int pooled_height, const int pooled_width,
                                       const int kernel_h, const int kernel_w, const int stride_h,
                                       const int stride_w, const int pad_h, const int pad_w,
                                       DType* out_data, int* mask) {
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
    int max_index = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int cur_index = h * width + w;  
        const DType in_val = in_slice[cur_index];
        if (in_val > max_val) {
          max_val = in_val;
          max_index = cur_index;
        }
      }
    }
    out_data[index] = max_val;
    mask[index] = max_index;
  }
}

template <typename DType>
__global__ void unpool_max_2d_gpu_kernel(const int nthreads, const DType* out_grad,
                                         const int channels, const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const int kernel_h, const int kernel_w,
                                         const int stride_h, const int stride_w,
                                         const int pad_h, const int pad_w,
                                         DType* in_grad, const int* mask) {
  // index is the output image's pixel index in NCHW
  // the order has to be consistent with pooling max
  // to avoid adding out_grad to the wrong in_grad
  // in the case where there are multiple max pixels
  // covered by a kernel window
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    // in data/grad offset batch and channel dims
    int in_offset = (n * channels + c) * height * width;
    int max_idx = mask[index];

    // In the case where pad > 0 and kernel = 1, for example,
    // max_idx can be -1 reaching this step.
    if (max_idx >= 0) {
      atomicAdd(&in_grad[in_offset+max_idx], out_grad[index]);
    }
  }
}

template<typename xpu, typename DType>
void PoolingMaskOp<xpu, DType>::pool_mask_forward(mshadow::Stream<gpu>* s, 
                              const DType* in_data, DType* out_data, int* mask,
                              const TShape& ishape, const TShape& oshape,
                              const TShape& kernel, const TShape& stride, const TShape& pad) {
  using namespace mxnet_op;                              
  pool_max_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                   oshape.Size(), in_data, ishape[1], ishape[2], ishape[3],
                                   oshape[2], oshape[3], kernel[0], kernel[1],
                                   stride[0], stride[1], pad[0], pad[1], out_data, mask);
  MSHADOW_CUDA_POST_KERNEL_CHECK(pool_max_2d_gpu_kernel);                                  
}

template<typename xpu, typename DType>
void PoolingMaskOp<xpu, DType>::pool_mask_backward(mshadow::Stream<gpu>* s, 
                              DType* in_grad, const DType* out_grad, const int* mask,
                              const TShape& ishape, const TShape& oshape,
                              const TShape& kernel, const TShape& stride, const TShape& pad) {
  using namespace mxnet_op;                                 
  unpool_max_2d_gpu_kernel<<<cuda_get_num_blocks(oshape.Size()), mshadow::cuda::kBaseThreadNum, 
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
                                     oshape.Size(), out_grad,
                                     ishape[1], ishape[2], ishape[3],
                                     oshape[2], oshape[3], kernel[0], kernel[1],
                                     stride[0], stride[1], pad[0], pad[1], in_grad, mask);
  MSHADOW_CUDA_POST_KERNEL_CHECK(unpool_max_2d_gpu_kernel);
}

template<>
Operator *CreateOp<gpu>(PoolingMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PoolingMaskOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
