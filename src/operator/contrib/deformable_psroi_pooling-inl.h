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
* \file deformable_psroi_pooling-inl.h
* \brief deformable psroi pooling operator and symbol
* \author Yi Li, Guodong Zhang, Jifeng Dai
*/
#ifndef MXNET_OPERATOR_CONTRIB_DEFORMABLE_PSROI_POOLING_INL_H_
#define MXNET_OPERATOR_CONTRIB_DEFORMABLE_PSROI_POOLING_INL_H_

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
namespace deformablepsroipool {
  enum DeformablePSROIPoolingOpInputs { kData, kBox, kTrans };
  enum DeformablePSROIPoolingOpOutputs { kOut, kTopCount };
}  // deformablepsroipool

struct DeformablePSROIPoolingParam : public dmlc::Parameter<DeformablePSROIPoolingParam> {
  // TShape pooled_size;
  float spatial_scale;
  int output_dim;
  int group_size;
  int pooled_size;
  int part_size;
  int sample_per_part;
  float trans_std;
  bool no_trans;
  DMLC_DECLARE_PARAMETER(DeformablePSROIPoolingParam) {
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
      .describe("Ratio of input feature map height (or w) to raw image height (or w). "
        "Equals the reciprocal of total stride in convolutional layers");
    DMLC_DECLARE_FIELD(output_dim).describe("fix output dim");
    DMLC_DECLARE_FIELD(group_size).describe("fix group size");
    DMLC_DECLARE_FIELD(pooled_size).describe("fix pooled size");
    DMLC_DECLARE_FIELD(part_size).set_default(0).describe("fix part size");
    DMLC_DECLARE_FIELD(sample_per_part).set_default(1).describe("fix samples per part");
    DMLC_DECLARE_FIELD(trans_std).set_default(0.0).set_range(0.0, 1.0)
      .describe("fix transition std");
    DMLC_DECLARE_FIELD(no_trans).set_default(false)
      .describe("Whether to disable trans parameter.");
  }
};

template<typename xpu, typename DType>
class DeformablePSROIPoolingOp : public Operator {
 public:
  explicit DeformablePSROIPoolingOp(DeformablePSROIPoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
    const std::vector<TBlob> &in_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &out_data,
    const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.no_trans? 2 : 3;
    size_t out_expected = 2;
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(out_data[deformablepsroipool::kOut].shape_[0],
             in_data[deformablepsroipool::kBox].shape_[0]);
    CHECK_EQ(out_data[deformablepsroipool::kTopCount].shape_[0],
             in_data[deformablepsroipool::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[deformablepsroipool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[deformablepsroipool::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[deformablepsroipool::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> top_count = out_data[deformablepsroipool::kTopCount]
                                        .get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(top_count.CheckContiguous(), true);
    out = -FLT_MAX;
    top_count = 0.0f;

    Tensor<xpu, 4, DType> trans;
    if (!param_.no_trans) {
      trans = in_data[deformablepsroipool::kTrans].get<xpu, 4, DType>(s);
    }
    DeformablePSROIPoolForward(out, data, bbox, trans, top_count, param_.no_trans,
      param_.spatial_scale, param_.output_dim, param_.group_size, param_.pooled_size,
      param_.part_size, param_.sample_per_part, param_.trans_std);
  }

  virtual void Backward(const OpContext &ctx,
    const std::vector<TBlob> &out_grad,
    const std::vector<TBlob> &in_data,
    const std::vector<TBlob> &out_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &in_grad,
    const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.no_trans ? 2 : 3;
    size_t out_expected = 2;
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(out_grad[deformablepsroipool::kOut].shape_[0],
             in_data[deformablepsroipool::kBox].shape_[0]);
    CHECK_EQ(out_data[deformablepsroipool::kTopCount].shape_[0],
             in_data[deformablepsroipool::kBox].shape_[0]);
    CHECK_NE(req[deformablepsroipool::kData], kWriteInplace) <<
      "DeformablePSROIPooling: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[deformablepsroipool::kBox], kWriteInplace) <<
      "DeformablePSROIPooling: Backward doesn't support kWriteInplace.";
    // CHECK_NE(req[deformablepsroipool::kTrans], kWriteInplace) <<
    //  "DeformablePSROIPooling: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[deformablepsroipool::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[deformablepsroipool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[deformablepsroipool::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> top_count = out_data[deformablepsroipool::kTopCount]
                                        .get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[deformablepsroipool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grad_roi = in_grad[deformablepsroipool::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> grad_trans;
    Tensor<xpu, 4, DType> trans;
    if (!param_.no_trans) {
      CHECK_EQ(in_grad.size(), 3);
      trans = in_data[deformablepsroipool::kTrans].get<xpu, 4, DType>(s);
      grad_trans = in_grad[deformablepsroipool::kTrans].get<xpu, 4, DType>(s);
    }

    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(top_count.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);

    Assign(grad_in, req[deformablepsroipool::kData], 0);
    if (!param_.no_trans) {
      Assign(grad_trans, req[deformablepsroipool::kTrans], 0);
    }
    DeformablePSROIPoolBackwardAcc(grad_in, grad_trans, grad_out, data, bbox, trans,
      top_count, param_.no_trans, param_.spatial_scale, param_.output_dim, param_.group_size,
      param_.pooled_size, param_.part_size, param_.sample_per_part, param_.trans_std);
    Assign(grad_roi, req[deformablepsroipool::kBox], 0);
  }

 private:
  DeformablePSROIPoolingParam param_;
};  // class DeformablePSROIPoolingOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(DeformablePSROIPoolingParam param, int dtype);

#if DMLC_USE_CXX11
class DeformablePSROIPoolingProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (param_.no_trans) {
      return{ "data", "rois" };
    } else {
      return{ "data", "rois", "trans" };
    }
  }

  std::vector<std::string> ListOutputs() const override {
    return{ "output", "top_count" };
  }

  int NumOutputs() const override {
    return 2;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
    if (param_.part_size == 0) {
      param_.part_size = param_.pooled_size;
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (param_.no_trans) {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";
    } else {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, rois, trans]";
      // trans: [num_rois, 2, pooled_h, pooled_w]
      TShape tshape = in_shape->at(deformablepsroipool::kTrans);
      CHECK_EQ(tshape.ndim(), 4) << "trans should be a 4D tensor of shape";
    }

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(deformablepsroipool::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(deformablepsroipool::kBox);
    CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
    CHECK_EQ(bshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";

    // out: [num_rois, c, pooled_h, pooled_w]
    // top_count: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(
      Shape4(bshape[0], param_.output_dim, param_.pooled_size, param_.pooled_size));
    out_shape->push_back(
      Shape4(bshape[0], param_.output_dim, param_.pooled_size, param_.pooled_size));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
    std::vector<int> *out_type,
    std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 2);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    DeformablePSROIPoolingProp* deformable_psroi_pooling_sym = new DeformablePSROIPoolingProp();
    deformable_psroi_pooling_sym->param_ = this->param_;
    return deformable_psroi_pooling_sym;
  }

  std::string TypeString() const override {
    return "_contrib_DeformablePSROIPooling";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.no_trans) {
      return{ out_grad[deformablepsroipool::kOut], in_data[deformablepsroipool::kData],
              in_data[deformablepsroipool::kBox], out_data[deformablepsroipool::kTopCount] };
    } else {
      return{ out_grad[deformablepsroipool::kOut], in_data[deformablepsroipool::kData],
              in_data[deformablepsroipool::kBox], in_data[deformablepsroipool::kTrans],
              out_data[deformablepsroipool::kTopCount] };
    }
  }


  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const override;


 private:
  DeformablePSROIPoolingParam param_;
};  // class DeformablePSROIPoolingProp
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_DEFORMABLE_PSROI_POOLING_INL_H_
