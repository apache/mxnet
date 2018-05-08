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
 * \author Yuchen Guo, Zehao Shi
*/
#include "./roi_align-inl.h"


namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(ROIAlignParam);


NNVM_REGISTER_OP(_contrib_ROIAlign)
.describe("ROIAlign foward.")
.set_num_inputs(2)
.set_num_outputs(3)
.set_attr_parser(ParamParser<ROIAlignParam>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "rois"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "maxidx_x", "maxidx_y"};
})
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
  // max_idx_x: [num_rois, c, pooled_h, pooled_w]
  // max_idx_y: [num_rois, c, pooled_h, pooled_w]
  out_shape->clear();
  out_shape->push_back(
       Shape4(bshape[0], dshape[1], param.pooled_size[0], param.pooled_size[1]));
  out_shape->push_back(
       Shape4(bshape[0], dshape[1], param.pooled_size[0], param.pooled_size[1]));
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
  out_type->push_back(dtype);
  out_type->push_back(dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", ROIAlignForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ROIAlignGrad{"_backward_ROIAlign"})
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array")
.add_arguments(ROIAlignParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_ROIAlign)
.describe("ROIAlign backward.")
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<ROIAlignParam>)
.set_attr<FCompute>("FCompute<cpu>", ROIAlignBackward<cpu>);

}  // namespace op
}  // namespace mxnet
