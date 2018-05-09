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
 * \author Yuchen Guo, Zehao Shi, Hang Zhang
 * modified from TuSimple/mx-maskrcnn
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
enum ROIAlignOpOutputs {kOut};
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
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int index, const DType* bottom_data,
                                  const float spatial_scale, const int channels,
                                  const int height, const int width,
                                  const int pooled_height, const int pooled_width,
                                  const DType* bottom_rois, DType* top_data) {
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
      return;
    }

    DType roi_start_w = (bottom_rois[1]) * spatial_scale;
    DType roi_start_h = (bottom_rois[2]) * spatial_scale;
    DType roi_end_w = (bottom_rois[3]) * spatial_scale;
    DType roi_end_h = (bottom_rois[4]) * spatial_scale;
    // Force malformed ROIs to be 1x1
    DType roi_width = maximum::Map(roi_end_w - roi_start_w, static_cast<DType>(1));
    DType roi_height = maximum::Map(roi_end_h - roi_start_h, static_cast<DType>(1));
    DType bin_size_h = static_cast<DType>(roi_height) / static_cast<DType>(pooled_height);
    DType bin_size_w = static_cast<DType>(roi_width) / static_cast<DType>(pooled_width);

    DType h = static_cast<DType>(ph) * bin_size_h + roi_start_h;
    DType w = static_cast<DType>(pw) * bin_size_w + roi_start_w;

    // Add roi offsets and clip to input boundaries
    bottom_data += (roi_batch_ind * channels + c) * height * width;

    if (h < 0 || h >= height || w < 0 || w >= width) {
      top_data[index] = 0.;
    } else {
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
      // Bilinear interpolation
      top_data[index] =  (1 - alpha) * (1 - beta) * bottom_data[topleft]
                        + alpha * (1 - beta) * bottom_data[bottomleft]
                        + (1 - alpha) * beta * bottom_data[topright]
                        + alpha * beta * bottom_data[bottomright];
    }
  }
};


struct ROIAlignGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[roialign::kOut]);
    heads.push_back(n->inputs[roialign::kBox]);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_
