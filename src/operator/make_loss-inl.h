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
 * Copyright (c) 2015 by Contributors
 * \file make_loss-inl.h
 * \brief special layer for propagating loss
*/
#ifndef MXNET_OPERATOR_MAKE_LOSS_INL_H_
#define MXNET_OPERATOR_MAKE_LOSS_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace make_loss_enum {
enum MakeLossOpInputs {kData};
enum MakeLossOpOutputs {kOut};
enum MakeLossOpType {kNull, kBatch, kValid};
enum MakeLossOpResource {kTempSpace};
}  // namespace make_loss_enum

struct MakeLossParam : public dmlc::Parameter<MakeLossParam> {
  float grad_scale;
  int normalization;
  float valid_thresh;
  DMLC_DECLARE_PARAMETER(MakeLossParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Gradient scale as a supplement to unary and binary operators");
    DMLC_DECLARE_FIELD(valid_thresh).set_default(0.0f)
    .describe("clip each element in the array to 0 when it is less than ``valid_thresh``."
              " This is used when ``normalization`` is set to ``'valid'``.");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", make_loss_enum::kNull)
    .add_enum("batch", make_loss_enum::kBatch)
    .add_enum("valid", make_loss_enum::kValid)
    .set_default(make_loss_enum::kNull)
    .describe("If this is set to null, the output gradient will not be normalized. "
              "If this is set to batch, the output gradient will be divided by the batch size. "
              "If this is set to valid, the output gradient will be divided by the number of "
              "valid input elements.");
  }
};

template<typename xpu, typename DType>
class MakeLossOp : public Operator {
 public:
  explicit MakeLossOp(MakeLossParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                        const std::vector<TBlob> &in_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &out_data,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U) << "MakeLoss can only be used to one input";
    CHECK_EQ(out_data.size(), 1U);
    if (req[make_loss_enum::kOut] != kWriteInplace) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 2, DType> data = in_data[make_loss_enum::kData].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = out_data[make_loss_enum::kOut].FlatTo2D<xpu, DType>(s);
      Assign(out, req[make_loss_enum::kOut], F<mshadow_op::identity>(data));
    }
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
    Tensor<xpu, 2, DType> grad = in_grad[make_loss_enum::kData].FlatTo2D<xpu, DType>(s);
    if (param_.normalization == make_loss_enum::kValid) {
      Tensor<xpu, 2, DType> data = in_data[make_loss_enum::kData].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 1, DType> temp = ctx.requested[make_loss_enum::kTempSpace]
        .get_space_typed<xpu, 1, DType>(mshadow::Shape1(1), s);
      temp = sumall_except_dim<0>(reduce_keepdim<red::sum, false>(
        F<mshadow_op::threshold>(ScalarExp<DType>(param_.valid_thresh), data), 0));
      temp = F<mshadow_op::maximum>(ScalarExp<DType>(1.f), temp);  // avoid zero
      Assign(grad, req[make_loss_enum::kData],
        ScalarExp<DType>(param_.grad_scale) / broadcast<0>(
        broadcast_keepdim(temp, 0, grad.shape_[0]), grad.shape_));
    } else if (param_.normalization == make_loss_enum::kBatch) {
      Assign(grad, req[make_loss_enum::kData],
        ScalarExp<DType>(param_.grad_scale / grad.shape_[0]));
    } else {
      Assign(grad, req[make_loss_enum::kData], ScalarExp<DType>(param_.grad_scale));
    }
  }

 private:
  MakeLossParam param_;
};  // class MakeLossOp

template <typename xpu>
Operator *CreateOp(MakeLossParam param, int dtype);

#if DMLC_USE_CXX11
class MakeLossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  };

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = in_shape->at(make_loss_enum::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "Input must have specified type";
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MakeLossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MakeLoss";
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    if (param_.normalization == make_loss_enum::kValid) {
      return {in_data[make_loss_enum::kData]};
    }
    return {};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    if (param_.normalization == make_loss_enum::kValid) {
      return {ResourceRequest::kTempSpace};
    }
    return {};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const override {
    return {{in_data[make_loss_enum::kData], out_data[make_loss_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  MakeLossParam param_;
};  // class MakeLossProperty

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MAKE_LOSS_INL_H_
