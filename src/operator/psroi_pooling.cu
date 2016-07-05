/*!
 * Copyright (c) 2016 by Contributors
 * \file psroi_pooling.cu
 * \brief psroi pooling operator
 * \author Yi Li, Tairui Chen
*/
#include "./psroi_pooling-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/gpu_util.h"

#define PSROIPOOLING_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {

template <typename Dtype>
__global__ void PSROIPoolForwardKernel(
  const int count,
  const Dtype* bottom_data,
  const Dtype spatial_scale,
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
  const Dtype* bottom_rois,
  const int output_dim,
  const int group_size,
  Dtype* top_data,
  Dtype* mapping_channel) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
    Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
    Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
    Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom 
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                        + roi_start_h);
    int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                        + roi_start_w);
    int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                      + roi_start_h);
    int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                      + roi_start_w);
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0),width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gw = pw;
    int gh = ph;
    int c = (ctop*group_size + gh)*group_size + gw;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    Dtype out_sum = 0;
    for (int h = hstart; h < hend; ++h){
      for (int w = wstart; w < wend; ++w){
        int bottom_index = h*width + w;
        out_sum += bottom_data[bottom_index];
      }
    }

    Dtype bin_area = (hend - hstart)*(wend - wstart);
    top_data[index] = is_empty? 0. : out_sum/bin_area;
    mapping_channel[index] = c;
  }
}

template<typename Dtype>
inline void PSROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &mapping_channel,
                           const float spatial_scale,
                           const int output_dim_, 
                           const int group_size_) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *mapping_channel_ptr = mapping_channel.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "PSROIPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  PSROIPoolForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, output_dim_, group_size_, top_data, mapping_channel_ptr);
  PSROIPOOLING_CUDA_CHECK(cudaPeekAtLastError());
}


template <typename Dtype>
__global__ void PSROIPoolBackwardKernel(
  const int count,
  const Dtype* top_diff,
  const Dtype* mapping_channel,
  const int num_rois,
  const Dtype spatial_scale,
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
  const int output_dim, 
  Dtype* bottom_diff,
  const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
    Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
    Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
    Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

    // Compute w and h at bottom 
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
      + roi_start_h);
    int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
      + roi_start_w);
    int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
      + roi_start_h);
    int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
      + roi_start_w);
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Compute c at bottom
    int c = mapping_channel[index];
    Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    Dtype bin_area = (hend - hstart)*(wend - wstart);
    Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
    for (int h = hstart; h < hend; ++h){
      for (int w = wstart; w < wend; ++w){
        int bottom_index = h*width + w;
        mxnet_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
      }
    }
  }
}


template<typename Dtype>
inline void PSROIPoolBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                            const Tensor<gpu, 4, Dtype> &out_grad,
                            const Tensor<gpu, 2, Dtype> &bbox,
                            const Tensor<gpu, 4, Dtype> &mapping_channel,
                            const float spatial_scale,
                            const int output_dim_) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype* mapping_channel_ptr = mapping_channel.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "PSROIPooling Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  PSROIPoolBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, mapping_channel_ptr, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, output_dim_, bottom_diff, bottom_rois);
  PSROIPOOLING_CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace cuda

template<typename Dtype>
inline void PSROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &mapping_channel,
                           const float spatial_scale,
                           const int output_dim_, 
                           const int group_size_) {
  cuda::PSROIPoolForward(out, data, bbox, mapping_channel, spatial_scale, output_dim_, group_size_);
}

template<typename Dtype>
inline void PSROIPoolBackward(const Tensor<gpu, 4, Dtype> &in_grad,
                            const Tensor<gpu, 4, Dtype> &out_grad,
                            const Tensor<gpu, 2, Dtype> &bbox,
                            const Tensor<gpu, 4, Dtype> &mapping_channel,
                            const float spatial_scale,
                            const int output_dim_) {
  cuda::PSROIPoolBackward(in_grad, out_grad, bbox, mapping_channel, spatial_scale, output_dim_);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(PSROIPoolingParam param) {
  return new PSROIPoolingOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
