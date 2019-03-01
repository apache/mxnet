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
 * \author Hang Zhang, Shesung
 * modified from Caffe2
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
  mxnet::TShape pooled_size;
  float spatial_scale;
  int sample_ratio;
  bool position_sensitive;
  DMLC_DECLARE_PARAMETER(ROIAlignParam) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("ROI Align output roi feature map height and width: (h, w)");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
    DMLC_DECLARE_FIELD(sample_ratio).set_default(-1)
    .describe("Optional sampling ratio of ROI align, using adaptive size by default.");
    DMLC_DECLARE_FIELD(position_sensitive).set_default(false)
    .describe("Whether to perform position-sensitive RoI pooling. PSRoIPooling is "
    "first proposaled by R-FCN and it can reduce the input channels by ph*pw times, "
    "where (ph, pw) is the pooled_size");
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ROI_ALIGN_INL_H_
