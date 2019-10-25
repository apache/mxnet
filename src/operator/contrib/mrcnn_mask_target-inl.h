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
 *  Copyright (c) 2019 by Contributors
 * \file mrcnn_mask_target-inl.h
 * \brief Mask-RCNN target generator
 * \author Serge Panev
 */


#ifndef MXNET_OPERATOR_CONTRIB_MRCNN_MASK_TARGET_INL_H_
#define MXNET_OPERATOR_CONTRIB_MRCNN_MASK_TARGET_INL_H_

#include <mxnet/operator.h>
#include <vector>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

namespace mrcnn_index {
  enum ROIAlignOpInputs {kRoi, kGtMask, kMatches, kClasses};
  enum ROIAlignOpOutputs {kMask, kMaskClasses};
}  // namespace mrcnn_index

struct MRCNNMaskTargetParam : public dmlc::Parameter<MRCNNMaskTargetParam> {
  int num_rois;
  int num_classes;
  int sample_ratio;
  mxnet::TShape mask_size;

  DMLC_DECLARE_PARAMETER(MRCNNMaskTargetParam) {
    DMLC_DECLARE_FIELD(num_rois)
    .describe("Number of sampled RoIs.");
    DMLC_DECLARE_FIELD(num_classes)
    .describe("Number of classes.");
    DMLC_DECLARE_FIELD(mask_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("Size of the pooled masks height and width: (h, w).");
    DMLC_DECLARE_FIELD(sample_ratio).set_default(2)
    .describe("Sampling ratio of ROI align. Set to -1 to use adaptative size.");
  }
};

inline bool MRCNNMaskTargetShape(const NodeAttrs& attrs,
                                 std::vector<mxnet::TShape>* in_shape,
                                 std::vector<mxnet::TShape>* out_shape) {
  using namespace mshadow;
  const MRCNNMaskTargetParam& param = nnvm::get<MRCNNMaskTargetParam>(attrs.parsed);

  CHECK_EQ(in_shape->size(), 4) << "Input:[rois, gt_masks, matches, cls_targets]";

  // (B, N, 4)
  mxnet::TShape tshape = in_shape->at(mrcnn_index::kRoi);
  CHECK_EQ(tshape.ndim(), 3) << "rois should be a 2D tensor of shape [batch, rois, 4]";
  CHECK_EQ(tshape[2], 4) << "rois should be a 2D tensor of shape [batch, rois, 4]";
  auto batch_size = tshape[0];
  auto num_rois = tshape[1];

  // (B, M, H, W)
  tshape = in_shape->at(mrcnn_index::kGtMask);
  CHECK_EQ(tshape.ndim(), 4) << "gt_masks should be a 4D tensor";
  CHECK_EQ(tshape[0], batch_size) << " batch size should be the same for all the inputs.";

  // (B, N)
  tshape = in_shape->at(mrcnn_index::kMatches);
  CHECK_EQ(tshape.ndim(), 2) << "matches should be a 2D tensor";
  CHECK_EQ(tshape[0], batch_size) << " batch size should be the same for all the inputs.";

  // (B, N)
  tshape = in_shape->at(mrcnn_index::kClasses);
  CHECK_EQ(tshape.ndim(), 2) << "matches should be a 2D tensor";
  CHECK_EQ(tshape[0], batch_size) << " batch size should be the same for all the inputs.";

  // out: 2 * (B, N, C, MS, MS)
  auto oshape = Shape5(batch_size, num_rois, param.num_classes,
                       param.mask_size[0], param.mask_size[1]);
  out_shape->clear();
  out_shape->push_back(oshape);
  out_shape->push_back(oshape);
  return true;
}

inline bool MRCNNMaskTargetType(const NodeAttrs& attrs,
                                std::vector<int>* in_type,
                                std::vector<int>* out_type) {
  CHECK_EQ(in_type->size(), 4);
  int dtype = (*in_type)[1];
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(dtype);
  out_type->push_back(dtype);
  return true;
}

template<typename xpu>
void MRCNNMaskTargetRun(const MRCNNMaskTargetParam& param, const std::vector<TBlob> &inputs,
                        const std::vector<TBlob> &outputs, mshadow::Stream<xpu> *s);

template<typename xpu>
void MRCNNMaskTargetCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<TBlob> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &outputs) {
  auto s = ctx.get_stream<xpu>();
  const auto& p = dmlc::get<MRCNNMaskTargetParam>(attrs.parsed);
  MRCNNMaskTargetRun<xpu>(p, inputs, outputs, s);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_MRCNN_MASK_TARGET_INL_H_
