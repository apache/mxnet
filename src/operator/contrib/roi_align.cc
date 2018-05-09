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
struct ROIAlignBackwardKernel {
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

      // lack of atomicAdd in cpu
      bottom_diff[topleft] += top_diff[index] * (1. - alpha) * (1 - beta);
      bottom_diff[topright] += top_diff[index] * (1. - alpha) * beta;
      bottom_diff[bottomleft] += top_diff[index] * alpha * (1 - beta);
      bottom_diff[bottomright] += top_diff[index] * alpha * beta;
    }
  }
};

template<typename real>
void ROIAlignForwardCpu(const real* bottom_data, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const real * bottom_rois,
                     real* top_data);

template<typename real>
void ROIAlignBackwardCpu(const real* top_diff, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const real * bottom_rois,
                     real* bottom_diff);


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

  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *bottom_data = in_data[roialign::kData].dptr<DType>();
    const DType *bottom_rois = in_data[roialign::kBox].dptr<DType>();
    DType *top_data = out_data[roialign::kOut].dptr<DType>();

    ROIAlignForwardCpu<DType>(bottom_data, param.spatial_scale, num_rois,
                            height, width, channels, pooled_height, pooled_width, bottom_rois,
                            top_data);
    /*
    mxnet_op::Kernel<ROIAlignForwardKernel, cpu>::Launch(s,
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

  // const int count = out_grad[0].Size();
  const int num_rois = in_data[0].size(0);
  const int channels = outputs[0].size(1);
  const int height = outputs[0].size(2);
  const int width = outputs[0].size(3);
  const int pooled_height = out_grad[0].size(2);
  const int pooled_width = out_grad[0].size(3);

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
      /*
      mxnet_op::Kernel<ROIAlignBackwardKernel<cpu>, cpu>::Launch(s,
        count, top_diff, num_rois, param.spatial_scale,
        channels, height, width,
        pooled_height, pooled_width, grad_in, bottom_rois);
      */
      ROIAlignBackwardCpu<DType>(top_diff, param.spatial_scale, num_rois,
                     height, width, channels,
                     pooled_height, pooled_width, bottom_rois, grad_in);
    }
    if (kWriteTo == req[roialign::kBox]) {
      Fill<false>(s, outputs[1], kWriteTo, static_cast<DType>(0));
    }
  })
}



template<typename real>
void ROIAlignForwardCpu(const real* bottom_data, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const real * bottom_rois,
                     real* top_data)
{
    const int output_size = num_rois * aligned_height * aligned_width * channels;

    #pragma omp parallel for 
    for (int idx = 0; idx < output_size; ++idx)
    {
        // (n, c, ph, pw) is an element in the aligned output
        int pw = idx % aligned_width;
        int ph = (idx / aligned_width) % aligned_height;
        int c = (idx / aligned_width / aligned_height) % channels;
        int n = idx / aligned_width / aligned_height / channels;

        real roi_batch_ind = bottom_rois[n * 5 + 0];
        real roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
        real roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
        real roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
        real roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

        // Force malformed ROI to be 1x1
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
        if (h < 0 || h >= height || w < 0 || w >= width)
        {
            top_data[idx] = 0.;
        }
        else
        {
            real h_ratio = h - (real)(hstart);
            real w_ratio = w - (real)(wstart);
            int upleft = img_start + (c * height + hstart) * width + wstart;
            int upright = upleft + 1;
            int downleft = upleft + width;
            int downright = downleft + 1;

            top_data[idx] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
                + bottom_data[upright] * (1. - h_ratio) * w_ratio
                + bottom_data[downleft] * h_ratio * (1. - w_ratio)
                + bottom_data[downright] * h_ratio * w_ratio;
        }
    }
}

template<typename real>
void ROIAlignBackwardCpu(const real* top_diff, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const real * bottom_rois,
                     real* bottom_diff)
{
    const int output_size = num_rois * aligned_height * aligned_width * channels;

    #pragma omp parallel for 
    for (int idx = 0; idx < output_size; ++idx)
    {
        // (n, c, ph, pw) is an element in the aligned output
        int pw = idx % aligned_width;
        int ph = (idx / aligned_width) % aligned_height;
        int c = (idx / aligned_width / aligned_height) % channels;
        int n = idx / aligned_width / aligned_height / channels;

        real roi_batch_ind = bottom_rois[n * 5 + 0];
        real roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
        real roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
        real roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
        real roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

        // Force malformed ROI to be 1x1
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
        if (h < 0 || h >= height || w < 0 || w >= width)
        {
            real h_ratio = h - (real)(hstart);
            real w_ratio = w - (real)(wstart);
            int upleft = img_start + (c * height + hstart) * width + wstart;
            int upright = upleft + 1;
            int downleft = upleft + width;
            int downright = downleft + 1;

            bottom_diff[upleft] += top_diff[idx] * (1. - h_ratio) * (1. - w_ratio);
            bottom_diff[upright] += top_diff[idx] * (1. - h_ratio) *  w_ratio;
            bottom_diff[downleft] += top_diff[idx] * h_ratio * (1. - w_ratio);
            bottom_diff[downright] += top_diff[idx] * h_ratio * w_ratio;
        }
    }
}


DMLC_REGISTER_PARAMETER(ROIAlignParam);


NNVM_REGISTER_OP(_contrib_ROIAlign)
.describe(R"code(
ROI Align Layer

This operator takes a 4D feature map as an input array and region proposals as `rois`,
then align the feature map over sub-regions of input and produces a fixed-sized output array.
This operator is typically used in Faster R-CNN & Mask R-CNN networks.

Different from ROI pooling, ROI Align removes the harsh quantization, properly aligning
the extracted features with the input. RoIAlign computes the value of each sampling point
by bilinear interpolation from the nearby grid points on the feature map. No quantization is
performed on any coordinates involved in the RoI, its bins, or the sampling points.
Bilinear interpolation is used to compute the exact values of the
input features at four regularly sampled locations in each RoI bin.
Then the feature map can be aggregated by avg or max pooling.


Reference
---------

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
  const ROIAlignParam param = nnvm::get<ROIAlignParam>(attrs.parsed);
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
  out_shape->push_back(
       Shape4(bshape[0], dshape[1], param.pooled_size[0], param.pooled_size[1]));
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
.set_attr<FCompute>("FCompute<cpu>", ROIAlignForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ROIAlignGrad{"_backward_ROIAlign"})
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array")
.add_arguments(ROIAlignParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_ROIAlign)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<ROIAlignParam>)
.set_attr<FCompute>("FCompute<cpu>", ROIAlignBackward<cpu>);

}  // namespace op
}  // namespace mxnet

