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
 * \file correlation-inl.h
 * \brief correlation operator and symbol
 * \author Xu Dong
*/
#ifndef MXNET_OPERATOR_CORRELATION_INL_H_
#define MXNET_OPERATOR_CORRELATION_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"
namespace mxnet {
namespace op {
//  Declare enumeration of input order to make code more intuitive.
//  These enums are only visible within this header
namespace Correlation {
enum  CorrelationOpInputs{kData1, kData2};
enum  CorrelationOpOutputs{kOut, kTemp1, kTemp2};
}  //  namespace Correlation
struct CorrelationParam : public dmlc::Parameter<CorrelationParam> {
  uint32_t max_displacement;
  uint32_t kernel_size;
  uint32_t pad_size;
  uint32_t stride1;
  uint32_t stride2;
  bool is_multiply;
  DMLC_DECLARE_PARAMETER(CorrelationParam) {
    DMLC_DECLARE_FIELD(kernel_size).set_default(1)
    .describe("kernel size for Correlation must be an odd number");
    DMLC_DECLARE_FIELD(max_displacement).set_default(1)
    .describe("Max displacement of Correlation ");
    DMLC_DECLARE_FIELD(stride1).set_default(1)
    .describe("stride1 quantize data1 globally");
    DMLC_DECLARE_FIELD(stride2).set_default(1)
    .describe("stride2 quantize data2 within the neighborhood centered around data1");
    DMLC_DECLARE_FIELD(pad_size).set_default(0)
    .describe("pad for Correlation");
    DMLC_DECLARE_FIELD(is_multiply).set_default(true)
    .describe("operation type is either multiplication or subduction");
  }
};
template<typename xpu, typename DType>
class CorrelationOp : public Operator {
 public:
  explicit CorrelationOp(CorrelationParam param) {
    this->param_ = param;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_NE(param_.kernel_size % 2, 0) << "kernel size should be odd number";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data1 = in_data[Correlation::kData1].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data2 = in_data[Correlation::kData2].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out   = out_data[Correlation::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> tmp1  = out_data[Correlation::kTemp1].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> tmp2  = out_data[Correlation::kTemp2].get<xpu, 4, DType>(s);
    tmp1 = DType(0.0f);
    tmp2 = DType(0.0f);
    out = DType(0.0f);
    CHECK_EQ(data1.CheckContiguous(), true);
    CHECK_EQ(data2.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(tmp1.CheckContiguous(), true);
    CHECK_EQ(tmp2.CheckContiguous(), true);
    paddedbottomheight = data1.shape_[2] + 2 * param_.pad_size;
    paddedbottomwidth  = data1.shape_[3] + 2 * param_.pad_size;
    kernel_radius_ = (param_.kernel_size - 1) / 2;
    border_size_ = param_.max_displacement + kernel_radius_;
    stride1 = param_.stride1;
    stride2 = param_.stride2;
    top_width_ = std::ceil(static_cast<float>(paddedbottomwidth - border_size_ * 2)\
     / static_cast<float>(stride1));
    top_height_ = std::ceil(static_cast<float>(paddedbottomheight - border_size_ * 2)\
     / static_cast<float>(stride1));
    neighborhood_grid_radius_ = param_.max_displacement / stride2;
    neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    top_channels_ = neighborhood_grid_width_ * neighborhood_grid_width_;
    num =  data1.shape_[0];
    channels = data1.shape_[1];
    height = data1.shape_[2];
    width = data1.shape_[3];
    CorrelationForward(out, data1, data2, tmp1, tmp2, top_channels_, top_height_, top_width_,
                       param_.pad_size, param_.is_multiply,
                       param_.max_displacement, param_.kernel_size,
                       neighborhood_grid_radius_, neighborhood_grid_width_,
                       kernel_radius_, param_.stride1, param_.stride2);
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad_data1 = in_grad[Correlation::kData1].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_data2 = in_grad[Correlation::kData2].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out_g = out_grad[Correlation::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> tmp1 = out_data[Correlation::kTemp1].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> tmp2 = out_data[Correlation::kTemp2].get<xpu, 4, DType>(s);
    if (req[0] != kAddTo) grad_data1 = DType(0.0f);
    if (req[1] != kAddTo) grad_data2 = DType(0.0f);
    CHECK_EQ(grad_data1.CheckContiguous(), true);
    CHECK_EQ(grad_data2.CheckContiguous(), true);
    CHECK_EQ(out_g.CheckContiguous(), true);
    CHECK_EQ(tmp1.CheckContiguous(), true);
    CHECK_EQ(tmp2.CheckContiguous(), true);
    CorrelationBackward(out_g, grad_data1, grad_data2, tmp1, tmp2, top_channels_,
                        top_height_, top_width_, param_.pad_size, param_.is_multiply,
                        param_.max_displacement, param_.kernel_size, neighborhood_grid_radius_,
                        neighborhood_grid_width_, kernel_radius_, param_.stride1, param_.stride2,
                        num, channels, height, width);
  }

 private:
    CorrelationParam param_;
    int paddedbottomheight;
    int paddedbottomwidth;
    uint32_t kernel_radius_;
    uint32_t border_size_;
    uint32_t stride1;
    uint32_t stride2;
    uint32_t top_width_;
    uint32_t top_height_;
    uint32_t neighborhood_grid_radius_;
    uint32_t neighborhood_grid_width_;
    uint32_t top_channels_;
    int  num;
    int  channels;
    int  height;
    int  width;
};   //  class CorrelationOp
//  Decalre Factory function
template<typename xpu>
Operator* CreateOp(CorrelationParam param, int dtype);
#if DMLC_USE_CXX11
class CorrelationProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data1", "data2"};
  }
  std::vector<std::string> ListOutputs() const override {
    return {"output", "tmp1", "tmp2"};
  }
  int NumOutputs() const override {
    return 3;
  }
  int NumVisibleOutputs() const override {
    return 1;
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
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data1, data2]";
    TShape dshape1 = in_shape->at(Correlation::kData1);
    TShape dshape2 = in_shape->at(Correlation::kData2);
    CHECK_EQ(dshape1.ndim(), 4U) << "data should be a 4D tensor";
    CHECK_EQ(dshape2.ndim(), 4U) << "data should be a 4D tensor";
    int paddedbottomheight;
    int paddedbottomwidth;
    uint32_t kernel_radius_;
    uint32_t stride1;
    uint32_t stride2;
    uint32_t top_width_;
    uint32_t top_height_;
    uint32_t neighborhood_grid_radius_;
    uint32_t neighborhood_grid_width_;
    uint32_t top_channels_;
    uint32_t border_size_;
    paddedbottomheight = dshape1[2] + 2*param_.pad_size;
    paddedbottomwidth  = dshape1[3] + 2*param_.pad_size;
    kernel_radius_ = (param_.kernel_size -1)/2;
    border_size_ = param_.max_displacement + kernel_radius_;
    stride1 = param_.stride1;
    stride2 = param_.stride2;
    top_width_ = std::ceil(static_cast<float>(paddedbottomwidth - border_size_ * 2)\
     / static_cast<float>(stride1));
    top_height_ = std::ceil(static_cast<float>(paddedbottomheight - border_size_ * 2)\
     / static_cast<float>(stride1));
    neighborhood_grid_radius_ = param_.max_displacement / stride2;
    neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    top_channels_ = neighborhood_grid_width_ * neighborhood_grid_width_;
    CHECK_GE(top_width_, 1U) <<
    "Correlation cannot be done with current settings.Neighborhood and kernel don't fit in blob";
    CHECK_GE(top_height_, 1U) <<
    "Correlation cannot be done with current settings.Neighborhood and kernel don't fit in blob";
    out_shape->clear();
    out_shape->push_back(Shape4(dshape1[0], top_channels_, top_height_, top_width_));
    out_shape->push_back(Shape4(dshape1[0], paddedbottomheight, paddedbottomwidth, dshape1[1]));
    out_shape->push_back(Shape4(dshape1[0], paddedbottomheight, paddedbottomwidth, dshape1[1]));
    return true;
  }
  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = (*in_type)[0];
    type_assign(&dtype, (*in_type)[1]);
    type_assign(&dtype, (*out_type)[0]);
    type_assign(&dtype, (*out_type)[1]);
    type_assign(&dtype, (*out_type)[2]);

    TYPE_ASSIGN_CHECK(*in_type, 0, dtype);
    TYPE_ASSIGN_CHECK(*in_type, 1, dtype);
    TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
    TYPE_ASSIGN_CHECK(*out_type, 1, dtype);
    TYPE_ASSIGN_CHECK(*out_type, 2, dtype);
    return dtype != -1;
  }
  OperatorProperty* Copy() const override {
    CorrelationProp* Correlation_sym = new CorrelationProp();
    Correlation_sym->param_ = this->param_;
    return Correlation_sym;
  }
  std::string TypeString() const override {
    return "Correlation";
  }
  //  decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
     return {out_grad[Correlation::kOut],
     out_data[Correlation::kTemp1], out_data[Correlation::kTemp2]};
}
  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  CorrelationParam param_;
};  //  class CorrelationProp
#endif
}  //  namespace op
}  //  namespace mxnet
#endif  //  MXNET_OPERATOR_CORRELATION_INL_H_
