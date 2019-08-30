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
 * \file rroi_align-inl.h
 * \brief rroi align operator and symbol
 * \author Yixin Bao
*/
#ifndef MXNET_OPERATOR_CONTRIB_RROI_ALIGN_INL_H_
#define MXNET_OPERATOR_CONTRIB_RROI_ALIGN_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace rroialign {
enum RROIAlignOpInputs{kData, kBox};
enum RROIAlignOpOutputs {kOut};
}  // rroialign

struct RROIAlignParam : public dmlc::Parameter<RROIAlignParam> {
  mxnet::TShape pooled_size;
  float spatial_scale;
  int sampling_ratio;
  DMLC_DECLARE_PARAMETER(RROIAlignParam) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("RROI align output shape (h,w) ");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or width) to raw image height (or width). "
    "Equals the reciprocal of total stride in convolutional layers");
    DMLC_DECLARE_FIELD(sampling_ratio).set_default(-1)
    .describe("Optional sampling ratio of RROI align, using adaptive size by default.");
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_RROI_ALIGN_INL_H_
