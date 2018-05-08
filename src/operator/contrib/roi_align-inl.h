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
 * \file roi_align-inl.h
 * \brief roi align operator and symbol
 * \author Yuchen Guo, Zehao Shi
 * adapted from TuSimple/mx-maskrcnn
*/
#ifndef MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_
#define MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_

#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../tensor/init_op.h"


namespace mxnet {
namespace op {


// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace roialign {
enum ROIAlignOpInputs {kData, kBox};
enum ROIAlignOpOutputs {kOut, kMaxIdx_x, kMaxIdx_y};
}  // roialign


struct ROIAlignParam : public dmlc::Parameter<ROIAlignParam> {
  TShape pooled_size;
  float spatial_scale;
  DMLC_DECLARE_PARAMETER(ROIAlignParam) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("fix pooled size: (h, w)");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
  }
};


/*!
 * \brief Kernel for forward pass of ROIAlign.
 */
struct ROIAlignForwardKernel {
  /*!
   * \param index          loop index
   * \param bottom_data    input data which is a 4D feature map
   * \param spatial_scale  ratio of input feature map height (or width)
                               to raw image height (or width)
   * \param channels       channels of input data
   * \param height         height of input data
   * \param width          width of input data
   * \param pooled_height  height of fix pooled size
   * \param pooled_width   width of fix pooled size
   * \param bottom_rois    input rois of shape (batch, 5)
   * \param top_data       output data of shape (num_rois, channels, height, width)
   * \param argmax_x       index of value in pooled feature map on x axis, -1 if nothing is pooled
   * \param argmax_y       index of value in pooled feature map on y axis, -1 if nothing is pooled
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int index, const DType* bottom_data,
                                  const float spatial_scale, const int channels,
                                  const int height, const int width,
                                  const int pooled_height, const int pooled_width,
                                  const DType* bottom_rois, DType* top_data,
                                  DType* argmax_x, DType* argmax_y) {
    using namespace mxnet::op::mshadow_op;
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_x[index] = 0;
      argmax_y[index] = 0;
      return;
    }

    DType roi_start_w = (bottom_rois[1]) * spatial_scale;
    DType roi_start_h = (bottom_rois[2]) * spatial_scale;
    DType roi_end_w = (bottom_rois[3]) * spatial_scale;
    DType roi_end_h = (bottom_rois[4]) * spatial_scale;
    // Force malformed ROIs to be 1x1
    DType roi_width = maximum::Map(roi_end_w - roi_start_w, static_cast<DType>(1));
    DType roi_height = maximum::Map(roi_end_h - roi_start_h, static_cast<DType>(1));
    DType bin_size_h = static_cast<DType>(roi_height)
                       / static_cast<DType>(pooled_height);
    DType bin_size_w = static_cast<DType>(roi_width)
                       / static_cast<DType>(pooled_width);

    DType hstart = static_cast<DType>((ph) * bin_size_h);
    DType wstart = static_cast<DType>((pw) * bin_size_w);
    DType hend = static_cast<DType>((ph + 1) * bin_size_h);
    DType wend = static_cast<DType>((pw + 1) * bin_size_w);
    // Add roi offsets and clip to input boundaries
    hstart = minimum::Map(maximum::Map(hstart + roi_start_h, static_cast<DType>(0)),
                 static_cast<DType>(height));
    hend = minimum::Map(maximum::Map(hend + roi_start_h, static_cast<DType>(0)),
               static_cast<DType>(height));
    wstart = minimum::Map(maximum::Map(wstart + roi_start_w, static_cast<DType>(0)),
                 static_cast<DType>(width));
    wend = minimum::Map(maximum::Map(wend + roi_start_w, static_cast<DType>(0)),
               static_cast<DType>(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    // Define an empty pooling region to be zero
    DType maxval = is_empty ? static_cast<DType>(0) : mshadow::red::limits::MinValue<DType>();
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    DType maxidx_x = -1;
    DType maxidx_y = -1;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    DType h_stride = (hend - hstart)/3.0;
    DType w_stride = (wend - wstart)/3.0;
    for (DType h = hstart+h_stride; h <= hend-h_stride+0.01;
           h += maximum::Map(h_stride, static_cast<DType>(0.01))) {
      for (DType w = wstart+w_stride; w <= wend-w_stride+0.01;
             w += maximum::Map(w_stride, static_cast<DType>(0.01))) {
        int hlow = minimum::Map(maximum::Map(static_cast<int>(floor::Map(h)), 0), height-1);
        int hhigh = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(h)), 0), height-1);
        int wleft = minimum::Map(maximum::Map(static_cast<int>(floor::Map(w)), 0), width-1);
        int wright = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(w)), 0), width-1);
        int topleft = hlow * width + wleft;
        int topright = hlow * width + wright;
        int bottomleft = hhigh * width + wleft;
        int bottomright = hhigh * width + wright;

        DType alpha = (hlow == hhigh) ? static_cast<DType>(0.5) : (h - hlow) / (hhigh - hlow);
        DType beta = (wleft == wright) ? static_cast<DType>(0.5) : (w - wleft) / (wright - wleft);
        DType value = (1 - alpha) * (1 - beta) * bottom_data[topleft]
                        + alpha * (1 - beta) * bottom_data[bottomleft]
                        + (1 - alpha) * beta * bottom_data[topright]
                        + alpha * beta * bottom_data[bottomright];

        if (value > maxval) {
          maxval = value;
          maxidx_x = w;
          maxidx_y = h;
        }
      }
    }
    top_data[index] = maxval;
    argmax_x[index] = (DType)maxidx_x;
    argmax_y[index] = (DType)maxidx_y;
  }
};


template<typename xpu>
void ROIAlignForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& in_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  size_t expected_in = 2;
  size_t expected_out = 3;
  CHECK_EQ(in_data.size(), expected_in);
  CHECK_EQ(out_data.size(), expected_out);
  CHECK_EQ(out_data[roialign::kOut].shape_[0], in_data[roialign::kBox].shape_[0]);
  CHECK_EQ(out_data[roialign::kMaxIdx_x].shape_[0], in_data[roialign::kBox].shape_[0]);
  CHECK_EQ(out_data[roialign::kMaxIdx_y].shape_[0], in_data[roialign::kBox].shape_[0]);

  const ROIAlignParam param = nnvm::get<ROIAlignParam>(attrs.parsed);

  const int count = out_data[roialign::kOut].Size();
  const int channels = in_data[roialign::kData].size(1);
  const int height = in_data[roialign::kData].size(2);
  const int width = in_data[roialign::kData].size(3);
  const int pooled_height = out_data[roialign::kOut].size(2);
  const int pooled_width = out_data[roialign::kOut].size(3);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *bottom_data = in_data[roialign::kData].dptr<DType>();
    const DType *bottom_rois = in_data[roialign::kBox].dptr<DType>();
    DType *top_data = out_data[roialign::kOut].dptr<DType>();
    DType *argmax_x = out_data[roialign::kMaxIdx_x].dptr<DType>();
    DType *argmax_y = out_data[roialign::kMaxIdx_y].dptr<DType>();

    mxnet_op::Kernel<ROIAlignForwardKernel, xpu>::Launch(s,
      count, bottom_data, param.spatial_scale, channels, height, width, pooled_height,
      pooled_width, bottom_rois, top_data, argmax_x, argmax_y);
  })
}


/*!
 * \brief Kernel for backward pass of ROIAlign.
 */
struct ROIAlignBackwardKernel {
  /*!
   * \param index          loop index
   * \param top_diff       gradient of output data
   * \param argmax_x       index of value in pooled feature map on x axis
   * \param argmax_y       index of value in pooled feature map on y axis
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
                                  const DType* argmax_x, const DType* argmax_y,
                                  const int num_rois, const float spatial_scale,
                                  const int channels, const int height, const int width,
                                  const int pooled_height, const int pooled_width,
                                  DType* bottom_diff, const DType* bottom_rois) {
    using namespace mxnet::op::mshadow_op;
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    DType gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const DType* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      DType roi_start_w = (offset_bottom_rois[1]) * spatial_scale;
      DType roi_start_h = (offset_bottom_rois[2]) * spatial_scale;
      DType roi_end_w = (offset_bottom_rois[3]) * spatial_scale;
      DType roi_end_h = (offset_bottom_rois[4]) * spatial_scale;

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w > roi_start_w - 1.0 && w < roi_end_w + 1.0 &&
                           h > roi_start_h - 1.0 && h < roi_end_h + 1.0);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const DType* offset_top_diff = top_diff + offset;
      const DType* offset_argmax_x = argmax_x + offset;
      const DType* offset_argmax_y = argmax_y + offset;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          const int pool_index = ph * pooled_width + pw;
          DType a_x = offset_argmax_x[pool_index];
          DType a_y = offset_argmax_y[pool_index];
          int hlow = minimum::Map(maximum::Map(static_cast<int>(floor::Map(a_y)), 0), height-1);
          int hhigh = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(a_y)), 0), height-1);
          int wleft = minimum::Map(maximum::Map(static_cast<int>(floor::Map(a_x)), 0), width-1);
          int wright = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(a_x)), 0), width-1);
          // (w, h) is not around (a_x, a_y)
          if (h != hlow && h != hhigh && w != wleft && w != wright)
            continue;

          DType alpha = (hlow == hhigh) ? static_cast<DType>(0.5)
                                        : (a_y - hlow) / (hhigh - hlow);
          DType beta = (wleft == wright) ? static_cast<DType>(0.5)
                                         : (a_x - wleft) / (wright - wleft);
          if (h == hlow && w == wleft) {
            gradient += offset_top_diff[pool_index] * (1 - alpha) * (1 - beta);
          } else if (h == hlow && w == wright) {
            gradient += offset_top_diff[pool_index] * (1 - alpha) * beta;
          } else if (h == hhigh && w == wleft) {
            gradient += offset_top_diff[pool_index] * alpha * (1 - beta);
          } else if (h == hhigh && w == wright) {
            gradient += offset_top_diff[pool_index] * alpha * beta;
          }
        }
      }
    }
    bottom_diff[index] += gradient;
  }
};


template<typename xpu>
void ROIAlignBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 4);
  CHECK_EQ(outputs.size(), 2);
  // the order here relates to the order in ROIAlignGrad
  std::vector<TBlob> out_grad(1, inputs[0]);
  std::vector<TBlob> in_data(1, inputs[1]);
  std::vector<TBlob> out_data(inputs.begin() + 2, inputs.begin() + 4);

  CHECK_EQ(out_grad[0].shape_[0], in_data[0].shape_[0]);
  CHECK_EQ(out_data[0].shape_[0], in_data[0].shape_[0]);
  CHECK_EQ(out_data[1].shape_[0], in_data[0].shape_[0]);
  CHECK_NE(req[0], kWriteInplace) <<
    "ROIAlign: Backward doesn't support kWriteInplace.";
  CHECK_NE(req[1], kWriteInplace) <<
    "ROIAlign: Backward doesn't support kWriteInplace.";

  const ROIAlignParam param = nnvm::get<ROIAlignParam>(attrs.parsed);

  const int count = outputs[0].Size();
  const int num_rois = in_data[0].size(0);
  const int channels = outputs[0].size(1);
  const int height = outputs[0].size(2);
  const int width = outputs[0].size(3);
  const int pooled_height = out_grad[0].size(2);
  const int pooled_width = out_grad[0].size(3);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(out_grad[0].type_flag_, DType, {
    const DType *top_diff = out_grad[0].dptr<DType>();
    const DType *bottom_rois = in_data[0].dptr<DType>();
    DType *argmax_x = out_data[0].dptr<DType>();
    DType *argmax_y = out_data[1].dptr<DType>();
    DType *grad_in = outputs[0].dptr<DType>();
    // DType *grad_roi = outputs[1].dptr<DType>();

    if (kAddTo == req[roialign::kData] || kWriteTo == req[roialign::kData]) {
      if (kWriteTo == req[roialign::kData]) {
        Fill<false>(s, outputs[0], kWriteTo, static_cast<DType>(0));
      }
      mxnet_op::Kernel<ROIAlignBackwardKernel, xpu>::Launch(s,
        count, top_diff, argmax_x, argmax_y, num_rois, param.spatial_scale,
        channels, height, width,
        pooled_height, pooled_width, grad_in, bottom_rois);
    }
    if (kWriteTo == req[roialign::kBox]) {
      Fill<false>(s, outputs[1], kWriteTo, static_cast<DType>(0));
    }
  })
}


struct ROIAlignGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[roialign::kOut]);
    heads.push_back(n->inputs[roialign::kBox]);
    heads.emplace_back(nnvm::NodeEntry{n, roialign::kMaxIdx_x, 0});
    heads.emplace_back(nnvm::NodeEntry{n, roialign::kMaxIdx_y, 0});

    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_
