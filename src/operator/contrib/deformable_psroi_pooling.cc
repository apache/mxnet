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
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_psroi_pooling.cc
 * \brief
 * \author Yi Li, Guodong Zhang, Jifeng Dai
*/
#include "./deformable_psroi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {

  template <typename DType>
  inline DType bilinear_interp_cpu(
    const DType* data,
    const DType x,
    const DType y,
    const int width,
    const int height) {
    int x1 = floor(x);
    int x2 = ceil(x);
    int y1 = floor(y);
    int y2 = ceil(y);
    DType dist_x = static_cast<DType>(x - x1);
    DType dist_y = static_cast<DType>(y - y1);
    DType value11 = data[y1*width + x1];
    DType value12 = data[y2*width + x1];
    DType value21 = data[y1*width + x2];
    DType value22 = data[y2*width + x2];
    DType value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12
      + dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
    return value;
  }

  template <typename DType>
  inline void DeformablePSROIPoolForwardCPU(
    const int count,
    const DType* bottom_data,
    const DType spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const DType* bottom_rois, const DType* bottom_trans,
    const bool no_trans,
    const DType trans_std,
    const int sample_per_part,
    const int output_dim,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class,
    DType* top_data,
    DType* top_count) {
    const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#pragma omp parallel for num_threads(omp_threads)
    for (int index = 0; index < count; index++) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      const DType* offset_bottom_rois = bottom_rois + n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
      DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
      DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

      // Force too small ROIs to be 1x1
      DType roi_width = max(roi_end_w - roi_start_w, static_cast<DType>(0.1));  // avoid 0
      DType roi_height = max(roi_end_h - roi_start_h, static_cast<DType>(0.1));

      // Compute w and h at bottom
      DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
      DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

      DType sub_bin_size_h = bin_size_h / static_cast<DType>(sample_per_part);
      DType sub_bin_size_w = bin_size_w / static_cast<DType>(sample_per_part);

      int part_h = floor(static_cast<DType>(ph) / pooled_height*part_size);
      int part_w = floor(static_cast<DType>(pw) / pooled_width*part_size);
      int class_id = ctop / channels_each_class;
      DType trans_x = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;
      DType trans_y = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2 + 1)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;

      DType wstart = static_cast<DType>(pw)* bin_size_w
        + roi_start_w;
      wstart += trans_x * roi_width;
      DType hstart = static_cast<DType>(ph) * bin_size_h
        + roi_start_h;
      hstart += trans_y * roi_height;

      DType sum = 0;
      int count = 0;
      int gw = floor(static_cast<DType>(pw) * group_size / pooled_width);
      int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
      gw = min(max(gw, 0), group_size - 1);
      gh = min(max(gh, 0), group_size - 1);

      const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
      for (int ih = 0; ih < sample_per_part; ih++) {
        for (int iw = 0; iw < sample_per_part; iw++) {
          DType w = wstart + iw*sub_bin_size_w;
          DType h = hstart + ih*sub_bin_size_h;
          // bilinear interpolation
          if (w<-0.5 || w>width - 0.5 || h<-0.5 || h>height - 0.5) {
            continue;
          }
          w = min(max(w, static_cast<DType>(0)), static_cast<DType>(width - 1));
          h = min(max(h, static_cast<DType>(0)), static_cast<DType>(height - 1));
          int c = (ctop*group_size + gh)*group_size + gw;
          DType val = bilinear_interp_cpu(offset_bottom_data + c*height*width, w, h, width, height);
          sum += val;
          count++;
        }
      }
      top_data[index] = count == 0 ? static_cast<DType>(0) : sum / count;
      top_count[index] = count;
    }
  }

  template<typename DType>
  inline void DeformablePSROIPoolForward(const Tensor<cpu, 4, DType> &out,
    const Tensor<cpu, 4, DType> &data,
    const Tensor<cpu, 2, DType> &bbox,
    const Tensor<cpu, 4, DType> &trans,
    const Tensor<cpu, 4, DType> &top_count,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    // LOG(INFO) << "DeformablePSROIPoolForward";
    const DType *bottom_data = data.dptr_;
    const DType *bottom_rois = bbox.dptr_;
    const DType *bottom_trans = no_trans ? NULL : trans.dptr_;
    DType *top_data = out.dptr_;
    DType *top_count_data = top_count.dptr_;
    const int count = out.shape_.Size();
    const int channels = data.size(1);
    const int height = data.size(2);
    const int width = data.size(3);
    const int pooled_height = pooled_size;
    const int pooled_width = pooled_size;
    const int num_classes = no_trans ? 1 : trans.size(1) / 2;
    const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
    DeformablePSROIPoolForwardCPU <DType> (
        count, bottom_data, spatial_scale, channels, height, width, pooled_height, pooled_width,
        bottom_rois, bottom_trans, no_trans, trans_std, sample_per_part, output_dim,
        group_size, part_size, num_classes, channels_each_class, top_data, top_count_data);

    return;
  }

  template<typename DType>
  inline void DeformablePSROIPoolBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
    const Tensor<cpu, 4, DType> &trans_grad,
    const Tensor<cpu, 4, DType> &out_grad,
    const Tensor<cpu, 4, DType> &data,
    const Tensor<cpu, 2, DType> &bbox,
    const Tensor<cpu, 4, DType> &trans,
    const Tensor<cpu, 4, DType> &top_count,
    const bool no_trans,
    const float spatial_scale,
    const int output_dim,
    const int group_size,
    const int pooled_size,
    const int part_size,
    const int sample_per_part,
    const float trans_std) {
    // NOT_IMPLEMENTED;
    LOG(FATAL) << "only implemented in GPU";
    return;
  }
}  // namespace mshadow

namespace mxnet {
namespace op {

  template<>
  Operator *CreateOp<cpu>(DeformablePSROIPoolingParam param, int dtype) {
    Operator* op = nullptr;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new DeformablePSROIPoolingOp<cpu, DType>(param);
    });
    return op;
  }

  Operator *DeformablePSROIPoolingProp::CreateOperatorEx(
    Context ctx, mxnet::ShapeVector *in_shape,
    std::vector<int> *in_type) const {
    mxnet::ShapeVector out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
  }

  DMLC_REGISTER_PARAMETER(DeformablePSROIPoolingParam);

  MXNET_REGISTER_OP_PROPERTY(_contrib_DeformablePSROIPooling, DeformablePSROIPoolingProp)
    .describe("Performs deformable position-sensitive region-of-interest pooling on inputs.\n"
      "The DeformablePSROIPooling operation is described in https://arxiv.org/abs/1703.06211 ."
      "batch_size will change to the number of region bounding boxes after DeformablePSROIPooling")
    .add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
    .add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
      "[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
      "of designated region of interest. batch_index indicates the index of corresponding image "
      "in the input data")
    .add_argument("trans", "Symbol", "transition parameter")
    .add_arguments(DeformablePSROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
