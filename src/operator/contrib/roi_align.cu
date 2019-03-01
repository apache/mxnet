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
 * \file roi_align.cu
 * \brief roi align operator
 * \author Hang Zhang, Shesung
 * Adapted from Caffe2
*/
#include "./roi_align-inl.h"


namespace mxnet {
namespace op {

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

using namespace mshadow::cuda;

// The maximum number of blocks to use in the default kernel call.
constexpr int ROI_MAXIMUM_NUM_BLOCKS = 4096;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int ROI_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock,
          ROI_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}


template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void RoIAlignForwardKernel(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const bool position_sensitive,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int c_unpooled = c;
    int channels_unpooled = channels;
    if (position_sensitive) {
      c_unpooled = c * pooled_height * pooled_width + ph * pooled_width + pw;
      channels_unpooled = channels * pooled_height * pooled_width;
    }
    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels_unpooled + c_unpooled)
        * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T* w1,
    T* w2,
    T* w3,
    T* w4,
    int* x_low,
    int* x_high,
    int* y_low,
    int* y_high,
    const int /*index*/ /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    *w1 = *w2 = *w3 = *w4 = 0.;
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  *y_low = static_cast<int>(y);
  *x_low = static_cast<int>(x);

  if (*y_low >= height - 1) {
    *y_high = *y_low = height - 1;
    y = (T)*y_low;
  } else {
    *y_high = *y_low + 1;
  }

  if (*x_low >= width - 1) {
    *x_high = *x_low = width - 1;
    x = (T)*x_low;
  } else {
    *x_high = *x_low + 1;
  }

  T ly = y - *y_low;
  T lx = x - *x_low;
  T hy = 1. - ly, hx = 1. - lx;

  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

  return;
}

template <typename T>
__global__ void RoIAlignBackwardKernel(
    const int nthreads,
    const T* top_diff,
    const int num_rois,
    const T spatial_scale,
    const bool position_sensitive,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[1] * spatial_scale;
    T roi_start_h = offset_bottom_rois[2] * spatial_scale;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale;
    T roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int c_unpooled = c;
    int channels_unpooled = channels;
    if (position_sensitive) {
      c_unpooled = c * pooled_height * pooled_width + ph * pooled_width + pw;
      channels_unpooled = channels * pooled_height * pooled_width;
    }
    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels_unpooled + c_unpooled)
        * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            &w1,
            &w2,
            &w3,
            &w4,
            &x_low,
            &x_high,
            &y_low,
            &y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
        }  // if
      }  // ix
    }  // iy
  }  // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward

template<typename xpu>
void ROIAlignForwardCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  size_t expected_in = 2;
  size_t expected_out = 1;
  CHECK_EQ(in_data.size(), expected_in);
  CHECK_EQ(out_data.size(), expected_out);
  CHECK_EQ(out_data[roialign::kOut].shape_[0], in_data[roialign::kBox].shape_[0]);

  const ROIAlignParam param = nnvm::get<ROIAlignParam>(attrs.parsed);

  const int count = out_data[roialign::kOut].Size();
  const int num_rois = in_data[roialign::kBox].size(0);
  const int channels = out_data[roialign::kOut].size(1);  // channels of pooled output
  const int height = in_data[roialign::kData].size(2);
  const int width = in_data[roialign::kData].size(3);
  const int pooled_height = out_data[roialign::kOut].size(2);
  const int pooled_width = out_data[roialign::kOut].size(3);

  Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *bottom_data = in_data[roialign::kData].dptr<DType>();
    const DType *bottom_rois = in_data[roialign::kBox].dptr<DType>();
    DType *top_data = out_data[roialign::kOut].dptr<DType>();
    RoIAlignForwardKernel<DType>
      <<<ROI_GET_BLOCKS(count),
         kMaxThreadsPerBlock,
         0,
         stream>>>(
          count,
          bottom_data,
          param.spatial_scale,
          param.position_sensitive,
          channels,
          height,
          width,
          pooled_height,
          pooled_width,
          param.sample_ratio,
          bottom_rois,
          top_data);
  })
}


template<typename xpu>
void ROIAlignBackwardCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 2);
  // the order here relates to the order in ROIAlignGrad
  std::vector<TBlob> out_grad(1, inputs[0]);
  std::vector<TBlob> in_data(1, inputs[1]);
  // std::vector<TBlob> out_data(1, inputs[2]);

  CHECK_EQ(out_grad[0].shape_[0], in_data[0].shape_[0]);
  CHECK_NE(req[0], kWriteInplace) <<
    "ROIAlign: Backward doesn't support kWriteInplace.";
  CHECK_NE(req[1], kWriteInplace) <<
    "ROIAlign: Backward doesn't support kWriteInplace.";

  const ROIAlignParam param = nnvm::get<ROIAlignParam>(attrs.parsed);

  const int count = out_grad[0].Size();
  const int num_rois = in_data[0].size(0);
  const int channels = out_grad[0].size(1);  // channels of pooled output
  const int height = outputs[0].size(2);
  const int width = outputs[0].size(3);
  const int pooled_height = out_grad[0].size(2);
  const int pooled_width = out_grad[0].size(3);

  Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);

  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(out_grad[0].type_flag_, DType, {
    const DType *top_diff = out_grad[0].dptr<DType>();
    const DType *bottom_rois = in_data[0].dptr<DType>();
    DType *grad_in = outputs[0].dptr<DType>();

    if (kWriteTo == req[roialign::kBox]) {
      Fill<false>(s, outputs[1], kWriteTo, static_cast<DType>(0));
    }
    if (kNullOp == req[roialign::kData]) return;
    if (kWriteTo == req[roialign::kData]) {
      Fill<false>(s, outputs[0], kWriteTo, static_cast<DType>(0));
    }
    RoIAlignBackwardKernel<DType>
    <<<ROI_GET_BLOCKS(count),
       kMaxThreadsPerBlock,
       0,
       stream>>>(
        count,
        top_diff,
        num_rois,
        param.spatial_scale,
        param.position_sensitive,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        param.sample_ratio,
        grad_in,
        bottom_rois);
  })
}


NNVM_REGISTER_OP(_contrib_ROIAlign)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignForwardCompute<gpu>);

NNVM_REGISTER_OP(_backward_ROIAlign)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignBackwardCompute<gpu>);

}  // namespace op
}  // namespace mxnet
