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
 * Copyright (c) 2016 by Contributors
 * \file multibox_target-inl.h
 * \brief
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_MULTIBOX_TARGET_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTIBOX_TARGET_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <nnvm/tuple.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <valarray>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace mshadow_op {
struct safe_divide {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (b == DType(0.0f)) return DType(0.0f);
    return DType(a / b);
  }
};  // struct safe_divide
}  // namespace mshadow_op

namespace mboxtarget_enum {
enum MultiBoxTargetOpInputs {kAnchor, kLabel, kClsPred};
enum MultiBoxTargetOpOutputs {kLoc, kLocMask, kCls};
enum MultiBoxTargetOpResource {kTempSpace};
}  // namespace mboxtarget_enum

struct MultiBoxTargetParam : public dmlc::Parameter<MultiBoxTargetParam> {
  float overlap_threshold;
  float ignore_label;
  float negative_mining_ratio;
  float negative_mining_thresh;
  int minimum_negative_samples;
  nnvm::Tuple<float> variances;
  DMLC_DECLARE_PARAMETER(MultiBoxTargetParam) {
    DMLC_DECLARE_FIELD(overlap_threshold).set_default(0.5f)
    .describe("Anchor-GT overlap threshold to be regarded as a positive match.");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("Label for ignored anchors.");
    DMLC_DECLARE_FIELD(negative_mining_ratio).set_default(-1.0f)
    .describe("Max negative to positive samples ratio, use -1 to disable mining");
    DMLC_DECLARE_FIELD(negative_mining_thresh).set_default(0.5f)
    .describe("Threshold used for negative mining.");
    DMLC_DECLARE_FIELD(minimum_negative_samples).set_default(0)
    .describe("Minimum number of negative samples.");
    DMLC_DECLARE_FIELD(variances).set_default({0.1f, 0.1f, 0.2f, 0.2f})
    .describe("Variances to be encoded in box regression target.");
  }
};  // struct MultiBoxTargetParam

template<typename xpu, typename DType>
class MultiBoxTargetOp : public Operator {
 public:
  explicit MultiBoxTargetOp(MultiBoxTargetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow_op;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> anchors = in_data[mboxtarget_enum::kAnchor]
      .get_with_shape<xpu, 2, DType>(
      Shape2(in_data[mboxtarget_enum::kAnchor].size(1), 4), s);
    Tensor<xpu, 3, DType> labels = in_data[mboxtarget_enum::kLabel]
      .get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> cls_preds = in_data[mboxtarget_enum::kClsPred]
      .get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> loc_target = out_data[mboxtarget_enum::kLoc]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> loc_mask = out_data[mboxtarget_enum::kLocMask]
      .get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> cls_target = out_data[mboxtarget_enum::kCls]
      .get<xpu, 2, DType>(s);

    index_t num_batches = labels.size(0);
    index_t num_anchors = anchors.size(0);
    index_t num_labels = labels.size(1);
    // TODO(zhreshold): use maximum valid ground-truth in batch rather than # in dataset
    Shape<4> temp_shape = Shape4(11, num_batches, num_anchors, num_labels);
    Tensor<xpu, 4, DType> temp_space = ctx.requested[mboxtarget_enum::kTempSpace]
      .get_space_typed<xpu, 4, DType>(temp_shape, s);
    loc_target = 0.f;
    loc_mask = 0.0f;
    cls_target = param_.ignore_label;
    temp_space = -1.0f;
    CHECK_EQ(anchors.CheckContiguous(), true);
    CHECK_EQ(labels.CheckContiguous(), true);
    CHECK_EQ(cls_preds.CheckContiguous(), true);
    CHECK_EQ(loc_target.CheckContiguous(), true);
    CHECK_EQ(loc_mask.CheckContiguous(), true);
    CHECK_EQ(cls_target.CheckContiguous(), true);
    CHECK_EQ(temp_space.CheckContiguous(), true);

    // compute overlaps
    // TODO(zhreshold): squeeze temporary memory space
    // temp_space, 0:out, 1:l1, 2:t1, 3:r1, 4:b1, 5:l2, 6:t2, 7:r2, 8:b2
    // 9: intersection, 10:union
    temp_space[1] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 0, 1), -1,
      num_batches), 2, num_labels);
    temp_space[2] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 1, 2), -1,
      num_batches), 2, num_labels);
    temp_space[3] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 2, 3), -1,
      num_batches), 2, num_labels);
    temp_space[4] = broadcast_keepdim(broadcast_with_axis(slice<1>(anchors, 3, 4), -1,
      num_batches), 2, num_labels);
    Shape<3> temp_reshape = Shape3(num_batches, 1, num_labels);
    temp_space[5] = broadcast_keepdim(reshape(slice<2>(labels, 1, 2), temp_reshape), 1,
      num_anchors);
    temp_space[6] = broadcast_keepdim(reshape(slice<2>(labels, 2, 3), temp_reshape), 1,
      num_anchors);
    temp_space[7] = broadcast_keepdim(reshape(slice<2>(labels, 3, 4), temp_reshape), 1,
      num_anchors);
    temp_space[8] = broadcast_keepdim(reshape(slice<2>(labels, 4, 5), temp_reshape), 1,
      num_anchors);
    temp_space[9] = F<maximum>(ScalarExp<DType>(0.0f),
      F<minimum>(temp_space[3], temp_space[7]) - F<maximum>(temp_space[1], temp_space[5]))
        * F<maximum>(ScalarExp<DType>(0.0f),
        F<minimum>(temp_space[4], temp_space[8]) - F<maximum>(temp_space[2], temp_space[6]));
    temp_space[10] = (temp_space[3] - temp_space[1]) * (temp_space[4] - temp_space[2])
     + (temp_space[7] - temp_space[5]) * (temp_space[8] - temp_space[6])
      - temp_space[9];
    temp_space[0] = F<safe_divide>(temp_space[9], temp_space[10]);

    MultiBoxTargetForward(loc_target, loc_mask, cls_target,
                          anchors, labels, cls_preds, temp_space,
                          param_.overlap_threshold,
                          param_.ignore_label,
                          param_.negative_mining_ratio,
                          param_.negative_mining_thresh,
                          param_.minimum_negative_samples,
                          param_.variances);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> grad = in_grad[mboxtarget_enum::kClsPred].FlatTo2D<xpu, DType>(s);
  grad = 0.f;
}

 private:
  MultiBoxTargetParam param_;
};  // class MultiBoxTargetOp

template<typename xpu>
Operator* CreateOp(MultiBoxTargetParam param, int dtype);

#if DMLC_USE_CXX11
class MultiBoxTargetProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"anchor", "label", "cls_pred"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"loc_target", "loc_mask", "cls_target"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input: [anchor, label, clsPred]";
    TShape ashape = in_shape->at(mboxtarget_enum::kAnchor);
    CHECK_EQ(ashape.ndim(), 3) << "Anchor should be batch shared N*4 tensor";
    CHECK_EQ(ashape[0], 1) << "Anchors are shared across batches, first dim=1";
    CHECK_GT(ashape[1], 0) << "Number boxes should > 0";
    CHECK_EQ(ashape[2], 4) << "Box dimension should be 4: [xmin-ymin-xmax-ymax]";
    TShape lshape = in_shape->at(mboxtarget_enum::kLabel);
    CHECK_EQ(lshape.ndim(), 3) << "Label should be [batch-num_labels-(>=5)] tensor";
    CHECK_GT(lshape[1], 0) << "Padded label should > 0";
    CHECK_GE(lshape[2], 5) << "Label width must >=5";
    TShape pshape = in_shape->at(mboxtarget_enum::kClsPred);
    CHECK_EQ(pshape.ndim(), 3) << "Prediction: [nbatch-num_classes-num_anchors]";
    CHECK_EQ(pshape[2], ashape[1]) << "Number of anchors mismatch";
    TShape loc_shape = Shape2(lshape[0], ashape.Size());  // batch - (num_box * 4)
    TShape lm_shape = loc_shape;
    TShape label_shape = Shape2(lshape[0], ashape[1]);  // batch - num_box
    out_shape->clear();
    out_shape->push_back(loc_shape);
    out_shape->push_back(lm_shape);
    out_shape->push_back(label_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    MultiBoxTargetProp* MultiBoxTarget_sym = new MultiBoxTargetProp();
    MultiBoxTarget_sym->param_ = this->param_;
    return MultiBoxTarget_sym;
  }

  std::string TypeString() const override {
    return "_contrib_MultiBoxTarget";
  }

  //  decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  std::vector<ResourceRequest> ForwardResource(
       const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                              std::vector<int> *in_type) const override;

 private:
  MultiBoxTargetParam param_;
};  // class MultiBoxTargetProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_MULTIBOX_TARGET_INL_H_
