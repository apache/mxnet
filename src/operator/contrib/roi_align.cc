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
 * \file roi_align.cc
 * \brief roi align operator
 * \author Hang Zhang, Shesung
 * Adapted from Caffe2
*/
#include "./roi_align-inl.h"


namespace mxnet {
namespace op {

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>* pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc->at(pre_calc_index) = pc;
            pre_calc_index += 1;
            continue;
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
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc->at(pre_calc_index) = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward(
    const int nthreads,
    const T* bottom_data,
    const T& spatial_scale,
    const bool position_sensitive,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    int roi_cols,
    T* top_data) {
  DCHECK(roi_cols == 4 || roi_cols == 5);

  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
#pragma omp parallel for \
num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : std::ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        &pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          int c_unpooled = c;
          int channels_unpooled = channels;
          if (position_sensitive) {
            c_unpooled = c * pooled_height * pooled_width + ph * pooled_width + pw;
            channels_unpooled = channels * pooled_height * pooled_width;
          }
          const T* offset_bottom_data =
              bottom_data + (roi_batch_ind * channels_unpooled + c_unpooled)
              * height * width;
          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                  pc.w2 * offset_bottom_data[pc.pos2] +
                  pc.w3 * offset_bottom_data[pc.pos3] +
                  pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          top_data[index] = output_val;
        }  // for pw
      }  // for ph
    }  // for c
  }  // for n
}


template <typename T>
void bilinear_interpolate_gradient(
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

template <class T>
inline void add(const T& val, T* address) {
  *address += val;
}

template <typename T>
void ROIAlignBackward(
    const int nthreads,
    const T* top_diff,
    const int /*num_rois*/,
    const T& spatial_scale,
    const bool position_sensitive,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois,
    int rois_cols) {
  DCHECK(rois_cols == 4 || rois_cols == 5);

  for (int index = 0; index < nthreads; index++) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * rois_cols;
    int roi_batch_ind = 0;
    if (rois_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
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
        : std::ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
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
          // atomic add is not needed for now since it is single threaded
          add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
          add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
          add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
          add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
        }  // if
      }  // ix
    }  // iy
  }  // for
}  // ROIAlignBackward


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

  const ROIAlignParam& param = nnvm::get<ROIAlignParam>(attrs.parsed);

  const int count = out_data[roialign::kOut].Size();
  // const int num_rois = in_data[roialign::kBox].size(0);
  const int channels = out_data[roialign::kOut].size(1);  // channels of pooled output
  const int height = in_data[roialign::kData].size(2);
  const int width = in_data[roialign::kData].size(3);
  const int pooled_height = out_data[roialign::kOut].size(2);
  const int pooled_width = out_data[roialign::kOut].size(3);
  const int rois_cols = in_data[roialign::kBox].size(1);

  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *bottom_data = in_data[roialign::kData].dptr<DType>();
    const DType *bottom_rois = in_data[roialign::kBox].dptr<DType>();
    DType *top_data = out_data[roialign::kOut].dptr<DType>();

    ROIAlignForward<DType>(count, bottom_data, param.spatial_scale, param.position_sensitive,
                           channels, height, width, pooled_height, pooled_width,
                           param.sample_ratio, bottom_rois, rois_cols, top_data);
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

  const ROIAlignParam& param = nnvm::get<ROIAlignParam>(attrs.parsed);

  const int count = out_grad[0].Size();
  const int num_rois = in_data[0].size(0);
  const int channels = out_grad[0].size(1);  // channels of pooled output
  const int height = outputs[0].size(2);
  const int width = outputs[0].size(3);
  const int pooled_height = out_grad[0].size(2);
  const int pooled_width = out_grad[0].size(3);
  const int rois_cols = in_data[0].size(1);

  Stream<cpu> *s = ctx.get_stream<cpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(out_grad[0].type_flag_, DType, {
    const DType *top_diff = out_grad[0].dptr<DType>();
    const DType *bottom_rois = in_data[0].dptr<DType>();
    DType *grad_in = outputs[0].dptr<DType>();

    if (kAddTo == req[roialign::kData] || kWriteTo == req[roialign::kData]) {
      if (kWriteTo == req[roialign::kData]) {
        Fill<false>(s, outputs[0], kWriteTo, static_cast<DType>(0));
      }
      ROIAlignBackward<DType>(count, top_diff, num_rois, param.spatial_scale,
                     param.position_sensitive, channels, height, width,
                     pooled_height, pooled_width, param.sample_ratio, grad_in,
                     bottom_rois, rois_cols);
    }
    if (kWriteTo == req[roialign::kBox]) {
      Fill<false>(s, outputs[1], kWriteTo, static_cast<DType>(0));
    }
  })
}

DMLC_REGISTER_PARAMETER(ROIAlignParam);

NNVM_REGISTER_OP(_contrib_ROIAlign)
.describe(R"code(
This operator takes a 4D feature map as an input array and region proposals as `rois`,
then align the feature map over sub-regions of input and produces a fixed-sized output array.
This operator is typically used in Faster R-CNN & Mask R-CNN networks.

Different from ROI pooling, ROI Align removes the harsh quantization, properly aligning
the extracted features with the input. RoIAlign computes the value of each sampling point
by bilinear interpolation from the nearby grid points on the feature map. No quantization is
performed on any coordinates involved in the RoI, its bins, or the sampling points.
Bilinear interpolation is used to compute the exact values of the
input features at four regularly sampled locations in each RoI bin.
Then the feature map can be aggregated by avgpooling.


References
----------

He, Kaiming, et al. "Mask R-CNN." ICCV, 2017
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
.set_attr_parser(ParamParser<ROIAlignParam>)
.set_attr<nnvm::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      std::vector<TShape> *in_shape, std::vector<TShape> *out_shape){
  using namespace mshadow;
  const ROIAlignParam& param = nnvm::get<ROIAlignParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";
  // data: [batch_size, c, h, w]
  TShape dshape = in_shape->at(roialign::kData);
  CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";
  // bbox: [num_rois, 5]
  TShape bshape = in_shape->at(roialign::kBox);
  CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
  CHECK_EQ(bshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";
  // out: [num_rois, c, pooled_h, pooled_w]
  out_shape->clear();
  if (param.position_sensitive) {
    CHECK_EQ(dshape[1] % (param.pooled_size[0]*param.pooled_size[1]), 0) <<
      "Input channels should be divided by pooled_size[0]*pooled_size[1]"
      "when position_sensitive is true.";
    out_shape->push_back(
         Shape4(bshape[0], dshape[1]/param.pooled_size[0]/param.pooled_size[1],
                param.pooled_size[0], param.pooled_size[1]));
  } else {
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param.pooled_size[0], param.pooled_size[1]));
  }
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2);
  int dtype = (*in_type)[0];
  CHECK_EQ(dtype, (*in_type)[1]);
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", ROIAlignForwardCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[roialign::kOut]);
    heads.push_back(n->inputs[roialign::kBox]);
    return MakeGradNode("_backward_ROIAlign", n, heads, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array")
.add_arguments(ROIAlignParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_ROIAlign)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<ROIAlignParam>)
.set_attr<FCompute>("FCompute<cpu>", ROIAlignBackwardCompute<cpu>);

}  // namespace op
}  // namespace mxnet

