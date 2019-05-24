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
using std::round;

namespace mshadow {

  template <typename DType>
  inline DType bilinear_interp_cpu(const DType* data,
                                   const DType x, const DType y,
                                   const index_t width, const index_t height) {
    index_t x1 = floor(x);
    index_t x2 = ceil(x);
    index_t y1 = floor(y);
    index_t y2 = ceil(y);
    DType dist_x = static_cast<DType>(x - x1);
    DType dist_y = static_cast<DType>(y - y1);
    DType value11 = data[y1 * width + x1];
    DType value12 = data[y2 * width + x1];
    DType value21 = data[y1 * width + x2];
    DType value22 = data[y2 * width + x2];
    DType value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) * dist_y * value12 +
      dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
    return value;
  }

  template <typename DType>
  inline void DeformablePSROIPoolForwardCPU(const index_t count, const DType* bottom_data,
                                            const DType spatial_scale, const index_t channels,
                                            const index_t height, const index_t width,
                                            const index_t pooled_height, const index_t pooled_width,
                                            const DType* bottom_rois, const DType* bottom_trans,
                                            const bool no_trans, const DType trans_std,
                                            const index_t sample_per_part, const index_t output_dim,
                                            const index_t group_size, const index_t part_size,
                                            const index_t num_classes,
                                            const index_t channels_each_class,
                                            DType* top_data, DType* top_count) {
    const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#pragma omp parallel for num_threads(omp_threads)
    for (index_t index = 0; index < count; index++) {
      // The output is in order (n, ctop, ph, pw)
      index_t pw = index % pooled_width;
      index_t ph = (index / pooled_width) % pooled_height;
      index_t ctop = (index / pooled_width / pooled_height) % output_dim;
      index_t n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      const DType* offset_bottom_rois = bottom_rois + n * 5;
      index_t roi_batch_ind = offset_bottom_rois[0];
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

      index_t part_h = floor(static_cast<DType>(ph) / pooled_height * part_size);
      index_t part_w = floor(static_cast<DType>(pw) / pooled_width * part_size);
      index_t class_id = ctop / channels_each_class;
      DType trans_x = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;
      DType trans_y = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2 + 1)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;

      DType wstart = static_cast<DType>(pw) * bin_size_w + roi_start_w;
      wstart += trans_x * roi_width;
      DType hstart = static_cast<DType>(ph) * bin_size_h + roi_start_h;
      hstart += trans_y * roi_height;

      DType sum = 0;
      index_t count = 0;
      index_t gw = floor(static_cast<DType>(pw) * group_size / pooled_width);
      index_t gh = floor(static_cast<DType>(ph) * group_size / pooled_height);
      gw = min(max(gw, static_cast<index_t>(0)), group_size - 1);
      gh = min(max(gh, static_cast<index_t>(0)), group_size - 1);

      const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
      for (index_t ih = 0; ih < sample_per_part; ih++) {
        for (index_t iw = 0; iw < sample_per_part; iw++) {
          DType w = wstart + iw * sub_bin_size_w;
          DType h = hstart + ih * sub_bin_size_h;
          // bilinear interpolation
          if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
            continue;
          }
          w = min(max(w, static_cast<DType>(0)), static_cast<DType>(width - 1));
          h = min(max(h, static_cast<DType>(0)), static_cast<DType>(height - 1));
          index_t c = (ctop * group_size + gh) * group_size + gw;
          DType val = bilinear_interp_cpu(offset_bottom_data + c * height * width,
                                          w, h, width, height);
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
                                         const bool no_trans, const float spatial_scale,
                                         const index_t output_dim, const index_t group_size,
                                         const index_t pooled_size, const index_t part_size,
                                         const index_t sample_per_part, const float trans_std) {
    const DType *bottom_data = data.dptr_;
    const DType *bottom_rois = bbox.dptr_;
    const DType *bottom_trans = no_trans ? nullptr : trans.dptr_;
    DType *top_data = out.dptr_;
    DType *top_count_data = top_count.dptr_;
    const index_t count = out.shape_.Size();
    const index_t channels = data.size(1);
    const index_t height = data.size(2);
    const index_t width = data.size(3);
    const index_t pooled_height = pooled_size;
    const index_t pooled_width = pooled_size;
    const index_t num_classes = no_trans ? 1 : trans.size(1) / 2;
    const index_t channels_each_class = no_trans ? output_dim : output_dim / num_classes;
    DeformablePSROIPoolForwardCPU<DType>(count, bottom_data, spatial_scale,
                                         channels, height, width,
                                         pooled_height, pooled_width,
                                         bottom_rois, bottom_trans,
                                         no_trans, trans_std, sample_per_part,
                                         output_dim, group_size, part_size, num_classes,
                                         channels_each_class, top_data, top_count_data);

    return;
  }

  template <typename DType>
  inline void DeformablePSROIPoolBackwardAccCPU(const index_t count, const DType* top_diff,
                                                const DType* top_count, const index_t num_rois,
                                                const DType spatial_scale, const index_t channels,
                                                const index_t height, const index_t width,
                                                const index_t pooled_height,
                                                const index_t pooled_width,
                                                const index_t output_dim,
                                                DType* bottom_data_diff,
                                                DType* bottom_trans_diff,
                                                const DType* bottom_data,
                                                const DType* bottom_rois,
                                                const DType* bottom_trans,
                                                const bool no_trans,
                                                const DType trans_std,
                                                const index_t sample_per_part,
                                                const index_t group_size,
                                                const index_t part_size,
                                                const index_t num_classes,
                                                const index_t channels_each_class) {
    for (index_t index = 0; index < count; index++) {
      // The output is in order (n, ctop, ph, pw)
      index_t pw = index % pooled_width;
      index_t ph = (index / pooled_width) % pooled_height;
      index_t ctop = (index / pooled_width / pooled_height) % output_dim;
      index_t n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      const DType* offset_bottom_rois = bottom_rois + n * 5;
      index_t roi_batch_ind = offset_bottom_rois[0];
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

      index_t part_h = floor(static_cast<DType>(ph) / pooled_height * part_size);
      index_t part_w = floor(static_cast<DType>(pw) / pooled_width * part_size);
      index_t class_id = ctop / channels_each_class;
      DType trans_x = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;
      DType trans_y = no_trans ? static_cast<DType>(0) :
        bottom_trans[(((n * num_classes + class_id) * 2 + 1)
                        * part_size + part_h)
                        * part_size + part_w] * trans_std;

      DType wstart = static_cast<DType>(pw) * bin_size_w + roi_start_w;
      wstart += trans_x * roi_width;
      DType hstart = static_cast<DType>(ph) * bin_size_h + roi_start_h;
      hstart += trans_y * roi_height;

      if (top_count[index] <= 0) {
        continue;
      }
      DType diff_val = top_diff[index] / top_count[index];
      const DType* offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
      DType* offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
      index_t gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
      index_t gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
      gw = min(max(gw, static_cast<index_t>(0)), group_size - 1);
      gh = min(max(gh, static_cast<index_t>(0)), group_size - 1);

      for (index_t ih = 0; ih < sample_per_part; ih++) {
        for (index_t iw = 0; iw < sample_per_part; iw++) {
          DType w = wstart + iw * sub_bin_size_w;
          DType h = hstart + ih * sub_bin_size_h;
          // bilinear interpolation
          if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
            continue;
          }
          w = min(max(w, static_cast<DType>(0)), static_cast<DType>(width - 1));
          h = min(max(h, static_cast<DType>(0)), static_cast<DType>(height - 1));
          index_t c = (ctop * group_size + gh) * group_size + gw;
          // backward on feature
          index_t x0 = floor(w);
          index_t x1 = ceil(w);
          index_t y0 = floor(h);
          index_t y1 = ceil(h);
          DType dist_x = w - x0, dist_y = h - y0;
          DType q00 = (1 - dist_x) * (1 - dist_y);
          DType q01 = (1 - dist_x) * dist_y;
          DType q10 = dist_x * (1 - dist_y);
          DType q11 = dist_x * dist_y;
          index_t bottom_index_base = c * height * width;
          offset_bottom_data_diff[bottom_index_base + y0 * width + x0] += q00 * diff_val;
          offset_bottom_data_diff[bottom_index_base + y1 * width + x0] += q01 * diff_val;
          offset_bottom_data_diff[bottom_index_base + y0 * width + x1] += q10 * diff_val;
          offset_bottom_data_diff[bottom_index_base + y1 * width + x1] += q11 * diff_val;

          if (no_trans) {
            continue;
          }
          DType U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
          DType U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
          DType U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
          DType U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
          DType diff_x = U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y - U00 * (1 - dist_y);
          diff_x *= trans_std * diff_val * roi_width;
          DType diff_y = U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x - U00 * (1 - dist_x);
          diff_y *= trans_std * diff_val * roi_height;

          index_t offset_trans_diff = (((n * num_classes + class_id) * 2)
            * part_size + part_h) * part_size + part_w;
          bottom_trans_diff[offset_trans_diff] += diff_x;
          bottom_trans_diff[offset_trans_diff + part_size * part_size] += diff_y;
        }
      }
    }
  }

  template<typename DType>
  inline void DeformablePSROIPoolBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
                                             const Tensor<cpu, 4, DType> &trans_grad,
                                             const Tensor<cpu, 4, DType> &out_grad,
                                             const Tensor<cpu, 4, DType> &data,
                                             const Tensor<cpu, 2, DType> &bbox,
                                             const Tensor<cpu, 4, DType> &trans,
                                             const Tensor<cpu, 4, DType> &top_count,
                                             const bool no_trans, const float spatial_scale,
                                             const index_t output_dim, const index_t group_size,
                                             const index_t pooled_size, const index_t part_size,
                                             const index_t sample_per_part, const float trans_std) {
    const DType *top_diff = out_grad.dptr_;
    const DType *bottom_data = data.dptr_;
    const DType *bottom_rois = bbox.dptr_;
    const DType *bottom_trans = no_trans ? nullptr : trans.dptr_;
    DType *bottom_data_diff = in_grad.dptr_;
    DType *bottom_trans_diff = no_trans ? nullptr : trans_grad.dptr_;
    const DType *top_count_data = top_count.dptr_;
    const index_t count = out_grad.shape_.Size();
    const index_t num_rois = bbox.size(0);
    const index_t channels = in_grad.size(1);
    const index_t height = in_grad.size(2);
    const index_t width = in_grad.size(3);
    const index_t pooled_height = pooled_size;
    const index_t pooled_width = pooled_size;
    const index_t num_classes = no_trans ? 1 : trans_grad.size(1) / 2;
    const index_t channels_each_class = no_trans ? output_dim : output_dim / num_classes;
    DeformablePSROIPoolBackwardAccCPU<DType>(count, top_diff, top_count_data, num_rois,
                                             spatial_scale, channels, height, width,
                                             pooled_height, pooled_width, output_dim,
                                             bottom_data_diff, bottom_trans_diff,
                                             bottom_data, bottom_rois, bottom_trans,
                                             no_trans, trans_std, sample_per_part,
                                             group_size, part_size, num_classes,
                                             channels_each_class);

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

  Operator *DeformablePSROIPoolingProp::CreateOperatorEx(Context ctx,
                                                         mxnet::ShapeVector *in_shape,
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
