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
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling.cu
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
*/
#include "./roi_pooling-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void ROIPoolForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale, const int batch_size,
                                     const int channels, const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois, Dtype* top_data,
                                     index_t* argmax_data) {
  for (index_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = static_cast<int>(bottom_rois[0]);

    if (roi_batch_ind < 0 || roi_batch_ind >= batch_size) {
      top_data[index] = 0;
      argmax_data[index] = -1;
      continue;
    }

    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    index_t maxidx = -1;
    index_t offset_bottom_data = (roi_batch_ind * channels + c) * height * width;
    bottom_data += offset_bottom_data;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        index_t bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = offset_bottom_data + bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template<typename Dtype>
inline void ROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, index_t> &max_idx,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  index_t *argmax_data = max_idx.dptr_;
  const index_t count = out.shape_.Size();
  const int batch_size = data.size(0);
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ROIPoolForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, batch_size, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data, argmax_data);
  MSHADOW_CUDA_POST_KERNEL_CHECK(ROIPoolForwardKernel);
}

template<typename Dtype>
__global__ void ROIPoolBackwardAccKernel(const int count, const Dtype* top_diff,
                                         const index_t* argmax_data, Dtype* bottom_diff) {
  for (index_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    index_t max_idx = argmax_data[index];
    if (max_idx >= 0) {
      atomicAdd(&bottom_diff[max_idx], top_diff[index]);
    }
  }
}

template<typename Dtype>
inline void ROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, index_t> &max_idx,
                               const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  index_t *argmax_data = max_idx.dptr_;
  const index_t count = out_grad.shape_.Size();
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIPooling Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ROIPoolBackwardAccKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, argmax_data, bottom_diff);
  MSHADOW_CUDA_POST_KERNEL_CHECK(ROIPoolBackwardAccKernel);
}

}  // namespace cuda

template<typename Dtype>
inline void ROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, index_t> &max_idx,
                           const float spatial_scale) {
  cuda::ROIPoolForward(out, data, bbox, max_idx, spatial_scale);
}

template<typename Dtype>
inline void ROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, index_t> &max_idx,
                               const float spatial_scale) {
  cuda::ROIPoolBackwardAcc(in_grad, out_grad, bbox, max_idx, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIPoolingParam param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIPoolingOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
