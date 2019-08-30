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
 * Copyright (c) 2019 by Contributors
 * \file rroi_align.cc
 * \brief rroi align operator
 * \author Yixin Bao
 * Forward pass adapted from Caffe2
 * link: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_align_rotated_op.cc
 */
#include "./rroi_align-inl.h"
#include <mshadow/tensor.h>
#include "math.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mxnet {
namespace op {

template <typename DType>
struct position_for_bilinear_interpolate {
  // 4 positions and corresponding weights for
  // computing bilinear interpolation
  int pos1, pos2, pos3, pos4;
  DType w1, w2, w3, w4;
};

template <typename DType>
void pre_calc_for_bilinear_interpolate(
    const int height, const int width, const int pooled_height, const int pooled_width,
    const int iy_upper, const int ix_upper, DType roi_start_h, DType roi_start_w,
    DType bin_size_h, DType bin_size_w, int roi_bin_grid_h, int roi_bin_grid_w,
    DType roi_center_h, DType roi_center_w, DType theta,
    std::vector<position_for_bilinear_interpolate<DType>> *pre_calc) {
  int pre_calc_index = 0;
  DType cosTheta = cos(theta);
  DType sinTheta = sin(theta);
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      // calc bin grid position (xx,yy)
      for (int iy = 0; iy < iy_upper; iy++) {
        const DType yy = roi_start_h + ph * bin_size_h +
            static_cast<DType>(iy + .5f) * bin_size_h /
                static_cast<DType>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const DType xx = roi_start_w + pw * bin_size_w +
              static_cast<DType>(ix + .5f) * bin_size_w /
                  static_cast<DType>(roi_bin_grid_w);

          // Rotate by theta around the center and translate
          DType x = xx * cosTheta + yy * sinTheta + roi_center_w;
          DType y = yy * cosTheta - xx * sinTheta + roi_center_h;

          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            position_for_bilinear_interpolate<DType> &pc = (*pre_calc)[pre_calc_index];
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc_index += 1;
            continue;
          }
          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          // calc 4 points for interpolation
          int y_low = static_cast<int>(y);
          int x_low = static_cast<int>(x);
          int y_high;
          int x_high;
          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (DType)y_low;
          } else {
            y_high = y_low + 1;
          }
          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (DType)x_low;
          } else {
            x_high = x_low + 1;
          }
          DType ly = y - y_low;
          DType lx = x - x_low;
          DType hy = 1. - ly, hx = 1. - lx;
          DType w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // Save weights and indices
          position_for_bilinear_interpolate<DType> &pc = (*pre_calc)[pre_calc_index];
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename DType>
inline void RROIAlignForward(const OpContext &ctx, const RROIAlignParam &param,
                             const std::vector<TBlob> &in_data, const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &out_data) {
  // data: [batch_size, c, h, w]
  const TBlob &data = in_data[rroialign::kData];
  const TBlob &bbox = in_data[rroialign::kBox];
  const DType *bottom_data = data.dptr<DType>();
  const int channels_ = data.size(1);
  const int height_ = data.size(2);
  const int width_ = data.size(3);
  const index_t data_size_c = height_ * width_;
  const index_t data_size = channels_ * data_size_c;

  // bbox: [num_rois, 6] (6: [batch_index, x, y, w, h, theta])
  const DType *bottom_rois = bbox.dptr<DType>();
  const int num_rois = bbox.size(0);
  const float spatial_scale_ = param.spatial_scale;
  const int sampling_ratio_ = param.sampling_ratio;

  // out: [num_rois, c, pooled_h, pooled_w]
  const TBlob &out = out_data[rroialign::kOut];
  DType *top_data = out.dptr<DType>();
  const int pooled_height_ = out.size(2);
  const int pooled_width_ = out.size(3);
  const index_t out_size_c = pooled_height_ * pooled_width_;
  const index_t out_size = channels_ * out_size_c;

  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int n = 0; n < num_rois; ++n) {
    // Increment ROI data pointer
    const DType *bottom_rois_n = bottom_rois + n * bbox.size(1);
    DType *top_data_n = top_data + n * out_size;
    int roi_batch_ind = static_cast<int>(bottom_rois_n[0]);
    DType roi_center_w = bottom_rois_n[1] * spatial_scale_;
    DType roi_center_h = bottom_rois_n[2] * spatial_scale_;
    DType roi_width = bottom_rois_n[3] * spatial_scale_;
    DType roi_height = bottom_rois_n[4] * spatial_scale_;
    DType roi_theta = bottom_rois_n[5] * M_PI / 180.0;

    // force malformed ROIs to be 1 * 1
    roi_width = max(roi_width, (DType) 1.);
    roi_height = max(roi_height, (DType) 1.);
    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    DType roi_start_h = -roi_height / 2.0;
    DType roi_start_w = -roi_width / 2.0;

    const DType bin_size_h = static_cast<DType>(roi_height) / static_cast<DType>(pooled_height_);
    const DType bin_size_w = static_cast<DType>(roi_width) / static_cast<DType>(pooled_width_);
    // We use roi_bin_grid to sample the grid and mimic integral,
    // e.g. roi_bin_grid = 2, means sample 2*2=4 points in each bin
    int roi_bin_grid_h =
        (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_height / pooled_height_);
    int roi_bin_grid_w = (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_width / pooled_width_);
    const DType bin_points_count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    // We want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization.
    std::vector<position_for_bilinear_interpolate<DType>> pre_calc(roi_bin_grid_h * roi_bin_grid_w *
                                                                   pooled_width_ * pooled_height_);

    pre_calc_for_bilinear_interpolate(height_, width_, pooled_height_, pooled_width_,
                                      roi_bin_grid_h, roi_bin_grid_w, roi_start_h, roi_start_w,
                                      bin_size_h, bin_size_w, roi_bin_grid_h, roi_bin_grid_w,
                                      roi_center_h, roi_center_w, roi_theta, &pre_calc);

    for (int c = 0; c < channels_; ++c) {
      const DType *offset_bottom_data = bottom_data + roi_batch_ind * data_size + c * data_size_c;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height_; ph++) {
        for (int pw = 0; pw < pooled_width_; pw++) {
          DType output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              position_for_bilinear_interpolate<DType> pc = pre_calc[pre_calc_index];
              output_val +=
                  pc.w1 * offset_bottom_data[pc.pos1] + pc.w2 * offset_bottom_data[pc.pos2] +
                  pc.w3 * offset_bottom_data[pc.pos3] + pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= bin_points_count;  // avg pooling for bin grid
          int index = c * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw;
          top_data_n[index] = output_val;
        }   // for pw
      }   // for ph
    }   // for c
  }   // for n
}

template<typename xpu>
void RROIAlignForwardCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& in_data,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& out_data) {
  const RROIAlignParam& param = nnvm::get<RROIAlignParam>(attrs.parsed);
  CHECK_EQ(in_data.size(), 2);
  CHECK_EQ(out_data.size(), 1);
  CHECK_EQ(out_data[rroialign::kOut].shape_[0], in_data[rroialign::kBox].shape_[0]);

  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    RROIAlignForward<DType>(ctx, param, in_data, req, out_data);
  })
}

template<typename xpu>
void RROIAlignBackwardCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& in_data,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& out_data) {
  LOG(FATAL) << "RROIAlign: Backward is not supported.";
}

DMLC_REGISTER_PARAMETER(RROIAlignParam);

NNVM_REGISTER_OP(_contrib_RROIAlign)
.describe(R"code(Performs Rotated ROI Align on the input array.

This operator takes a 4D feature map as an input array and region proposals as `rois`,
then align the feature map over sub-regions of input and produces a fixed-sized output array.

Different from ROI Align, RROI Align uses rotated rois, which is suitable for text detection.
RRoIAlign computes the value of each sampling point by bilinear interpolation from the nearby
grid points on the rotated feature map. No quantization is performed on any coordinates
involved in the RoI, its bins, or the sampling points. Bilinear interpolation is used to
compute the exact values of the input features at four regularly sampled locations in
each RoI bin. Then the feature map can be aggregated by avgpooling.

References
----------

Ma, Jianqi, et al. "Arbitrary-Oriented Scene Text Detection via Rotation Proposals."
IEEE Transactions on Multimedia, 2018.

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "rois"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr_parser(ParamParser<RROIAlignParam>)
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  const RROIAlignParam& param = nnvm::get<RROIAlignParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2U) << "Input:[data, rois]";
  // data: [batch_size, c, h, w]
  mxnet::TShape dshape = in_shape->at(rroialign::kData);
  CHECK_EQ(dshape.ndim(), 4U) << "data should be a 4D tensor";
  // bbox: [num_rois, 6]
  mxnet::TShape bshape = in_shape->at(rroialign::kBox);
  CHECK_EQ(bshape.ndim(), 2U) << "bbox should be a 2D tensor of shape [batch, 6]";
  CHECK_EQ(bshape[1], 6U) << "bbox should be a 2D tensor of shape [batch, 6]";
  // out: [num_rois, c, pooled_h, pooled_w]
  out_shape->clear();
  out_shape->push_back(Shape4(bshape[0], dshape[1], param.pooled_size[0], param.pooled_size[1]));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2U);
  int dtype = (*in_type)[0];
  CHECK_EQ(dtype, (*in_type)[1]);
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", RROIAlignForwardCompute<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array")
.add_arguments(RROIAlignParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_RROIAlign)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<RROIAlignParam>)
.set_attr<FCompute>("FCompute<cpu>", RROIAlignBackwardCompute<cpu>);

}  // namespace op
}  // namespace mxnet
