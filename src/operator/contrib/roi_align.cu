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
 * \author Yuchen Guo, Zehao Shi, Hang Zhang
 * modified from TuSimple/mx-maskrcnn
*/
#include "./roi_align-inl.h"


namespace mxnet {
namespace op {

/*!
 * \brief Kernel for backward pass of ROIAlign.
 */
template<typename xpu>
struct ROIAlignBackwardKernel2 {
  /*!
   * \param index          loop index
   * \param top_diff       gradient of output data
   * \param num_rois       number of rois
   * \param spatial_scale  ratio of input feature map height (or width)
                               to raw image height (or width)
   * \param channels       channels of input data
   * \param height         height of input data
   * \param width          width of input data
   * \param pooled_height  height of fix pooled size
   * \param pooled_width   width of fix pooled size
   * \param bottom_diff    gradient of input 4D feature map
   * \param bottom_rois    gradient of input rois
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int index, const DType* top_diff,
                                  const int num_rois, const float spatial_scale,
                                  const int channels, const int height, const int width,
                                  const int pooled_height, const int pooled_width,
                                  DType* bottom_diff, const DType* bottom_rois) {
    using namespace mxnet::op::mshadow_op;
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // Accumulate gradient over all ROIs that pooled this element
    const DType* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    DType roi_start_w = (offset_bottom_rois[1]) * spatial_scale;
    DType roi_start_h = (offset_bottom_rois[2]) * spatial_scale;
    DType roi_end_w = (offset_bottom_rois[3]) * spatial_scale;
    DType roi_end_h = (offset_bottom_rois[4]) * spatial_scale;

    DType roi_width = maximum::Map(roi_end_w - roi_start_w, static_cast<DType>(1));
    DType roi_height = maximum::Map(roi_end_h - roi_start_h, static_cast<DType>(1));
    DType bin_size_h = static_cast<DType>(roi_height) / static_cast<DType>(pooled_height);
    DType bin_size_w = static_cast<DType>(roi_width) / static_cast<DType>(pooled_width);

    DType h = static_cast<DType>(ph) * bin_size_h + roi_start_h;
    DType w = static_cast<DType>(pw) * bin_size_w + roi_start_w;

    // Bilinear interpolation
    // int img_start = roi_batch_ind * channels * height * width;
    int offset = (roi_batch_ind * channels + c) * height * width;

    // bilinear interpolation
    if (!(h < 0 || h >= height || w < 0 || w >= width)) {
      int hlow = minimum::Map(maximum::Map(static_cast<int>(floor::Map(h)), 0), height-1);
      int hhigh = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(h)), 0), height-1);
      int wleft = minimum::Map(maximum::Map(static_cast<int>(floor::Map(w)), 0), width-1);
      int wright = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(w)), 0), width-1);

      int topleft = offset + hlow * width + wleft;
      int topright = offset + hlow * width + wright;
      int bottomleft = offset + hhigh * width + wleft;
      int bottomright = offset + hhigh * width + wright;

      DType alpha = (hlow == hhigh) ? static_cast<DType>(0.5) : (h - hlow) / (hhigh - hlow);
      DType beta = (wleft == wright) ? static_cast<DType>(0.5) : (w - wleft) / (wright - wleft);

      atomicAdd(bottom_diff + topleft, top_diff[index] * (1. - alpha) * (1 - beta));
      atomicAdd(bottom_diff + topright, top_diff[index] * (1. - alpha) * beta);
      atomicAdd(bottom_diff + bottomleft, top_diff[index] * alpha * (1 - beta));
      atomicAdd(bottom_diff + bottomright, top_diff[index] * alpha * beta);
    }
  }
};

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


template<typename real>
__global__ void ROIAlignForwardKernel(const int nthreads, const real* bottom_data, const float spatial_scale, const int height, const int width,
                const int channels, const int aligned_height, const int aligned_width, const real* bottom_rois, real* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    // int n = index;
    // int pw = n % aligned_width;
    // n /= aligned_width;
    // int ph = n % aligned_height;
    // n /= aligned_height;
    // int c = n % channels;
    // n /= channels;

    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    // bottom_rois += n * 5;
    real roi_batch_ind = bottom_rois[n * 5 + 0];
    real roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
    real roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
    real roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
    real roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    real roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    real roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    real bin_size_h = roi_height / (aligned_height - 1.);
    real bin_size_w = roi_width / (aligned_width - 1.);

    real h = (real)(ph) * bin_size_h + roi_start_h;
    real w = (real)(pw) * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    int img_start = roi_batch_ind * channels * height * width;

    // bilinear interpolation
    if (h < 0 || h >= height || w < 0 || w >= width) {
      top_data[index] = 0.;
    } else {
      real h_ratio = h - (real)(hstart);
      real w_ratio = w - (real)(wstart);
      int upleft = img_start + (c * height + hstart) * width + wstart;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      top_data[index] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
        + bottom_data[upright] * (1. - h_ratio) * w_ratio
        + bottom_data[downleft] * h_ratio * (1. - w_ratio)
        + bottom_data[downright] * h_ratio * w_ratio;
    }
  }
}


template<typename real>
int ROIAlignForwardLaucher(const real* bottom_data, const float spatial_scale, const int num_rois,
      const int height, const int width, const int channels, const int aligned_height, const int aligned_width,
      const real* bottom_rois, real* top_data, cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * aligned_height * aligned_width * channels;
  cudaError_t err;


  ROIAlignForwardKernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    output_size, bottom_data, spatial_scale, height, width, channels,
    aligned_height, aligned_width, bottom_rois, top_data);

  err = cudaGetLastError();
  if(cudaSuccess != err) {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}


template<typename real>
__global__ void ROIAlignBackwardKernel(const int nthreads, const real* top_diff, const float spatial_scale,
      const int height, const int width, const int channels, const int aligned_height, const int aligned_width,
      real* bottom_diff, const real* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {

    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    real roi_batch_ind = bottom_rois[n * 5 + 0];
    real roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
    real roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
    real roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
    real roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;
    /* int roi_start_w = round(bottom_rois[1] * spatial_scale); */
    /* int roi_start_h = round(bottom_rois[2] * spatial_scale); */
    /* int roi_end_w = round(bottom_rois[3] * spatial_scale); */
    /* int roi_end_h = round(bottom_rois[4] * spatial_scale); */

    // Force malformed ROIs to be 1x1
    real roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    real roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    real bin_size_h = roi_height / (aligned_height - 1.);
    real bin_size_w = roi_width / (aligned_width - 1.);

    real h = (real)(ph) * bin_size_h + roi_start_h;
    real w = (real)(pw) * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    int img_start = roi_batch_ind * channels * height * width;

    // bilinear interpolation
    if (!(h < 0 || h >= height || w < 0 || w >= width)) {
      real h_ratio = h - (real)(hstart);
      real w_ratio = w - (real)(wstart);
      int upleft = img_start + (c * height + hstart) * width + wstart;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      atomicAdd(bottom_diff + upleft, top_diff[index] * (1. - h_ratio) * (1 - w_ratio));
      atomicAdd(bottom_diff + upright, top_diff[index] * (1. - h_ratio) * w_ratio);
      atomicAdd(bottom_diff + downleft, top_diff[index] * h_ratio * (1 - w_ratio));
      atomicAdd(bottom_diff + downright, top_diff[index] * h_ratio * w_ratio);
    }
  }
}


template<typename real>
int ROIAlignBackwardLaucher(const real* top_diff, const float spatial_scale,
      const int num_rois, const int height, const int width, const int channels, const int aligned_height,
      const int aligned_width, const real* bottom_rois, real* bottom_diff, cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * aligned_height * aligned_width * channels;
  cudaError_t err;

  ROIAlignBackwardKernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    output_size, top_diff, spatial_scale, height, width, channels,
    aligned_height, aligned_width, bottom_diff, bottom_rois);

  err = cudaGetLastError();
  if(cudaSuccess != err) {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}


template<typename xpu>
void ROIAlignForward(const nnvm::NodeAttrs& attrs,
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

  // const int count = out_data[roialign::kOut].Size();
  const int num_rois = in_data[roialign::kBox].size(0);
  const int channels = in_data[roialign::kData].size(1);
  const int height = in_data[roialign::kData].size(2);
  const int width = in_data[roialign::kData].size(3);
  const int pooled_height = out_data[roialign::kOut].size(2);
  const int pooled_width = out_data[roialign::kOut].size(3);

  Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *bottom_data = in_data[roialign::kData].dptr<DType>();
    const DType *bottom_rois = in_data[roialign::kBox].dptr<DType>();
    DType *top_data = out_data[roialign::kOut].dptr<DType>();

    ROIAlignForwardLaucher<DType>(bottom_data, param.spatial_scale, num_rois,
                            height, width, channels, pooled_height, pooled_width, bottom_rois,
                            top_data, stream);
    /*
    mxnet_op::Kernel<ROIAlignForwardKernel, gpu>::Launch(s,
      count, bottom_data, param.spatial_scale, channels, height, width, pooled_height,
      pooled_width, bottom_rois, top_data);
    */
  })
}


template<typename xpu>
void ROIAlignBackward(const nnvm::NodeAttrs& attrs,
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
  const int channels = outputs[0].size(1);
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

    if (kAddTo == req[roialign::kData] || kWriteTo == req[roialign::kData]) {
      if (kWriteTo == req[roialign::kData]) {
        Fill<false>(s, outputs[0], kWriteTo, static_cast<DType>(0));
      }
      /*
      mxnet_op::Kernel<ROIAlignBackwardKernel2<gpu>, gpu>::Launch(s,
        count, top_diff, num_rois, param.spatial_scale,
        channels, height, width,
        pooled_height, pooled_width, grad_in, bottom_rois);
      */
      ROIAlignBackwardLaucher<DType>(top_diff, param.spatial_scale, num_rois,
                     height, width, channels,
                     pooled_height, pooled_width, bottom_rois, grad_in, stream);
    }
    if (kWriteTo == req[roialign::kBox]) {
      Fill<false>(s, outputs[1], kWriteTo, static_cast<DType>(0));
    }
  })
}


NNVM_REGISTER_OP(_contrib_ROIAlign)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignForward<gpu>);

NNVM_REGISTER_OP(_backward_ROIAlign)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignBackward<gpu>);

}  // namespace op
}  // namespace mxnet
