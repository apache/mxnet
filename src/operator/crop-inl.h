/*!
 * Copyright (c) 2015 by Contributors
 * \file crop-inl.h
 * \brief
 * \author Wei Wu
*/
#ifndef MXNET_OPERATOR_CROP_INL_H_
#define MXNET_OPERATOR_CROP_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace crop_enum {
enum CropOpInputs {kData, kCropLike};
enum CropOpOutputs {kOut};
}  // namespace crop_enum

struct CropParam : public dmlc::Parameter<CropParam> {
  int num_args;
  TShape offset;
  TShape h_w;
  bool center_crop;
  DMLC_DECLARE_PARAMETER(CropParam) {
    DMLC_DECLARE_FIELD(num_args).set_range(1, 3)
    .describe("Number of inputs for crop, if equals one, then we will use the h_w"
      "for crop height and width, else if equals two, then we will use the height"
      "and width of the second input symbol, we name crop_like here");
    int shape[] = {0, 0};
    DMLC_DECLARE_FIELD(offset).set_default(TShape(shape, shape + 2))
    .describe("crop offset coordinate: (y, x)");
    DMLC_DECLARE_FIELD(h_w).set_default(TShape(shape, shape + 2))
    .describe("crop height and weight: (h, w)");
    DMLC_DECLARE_FIELD(center_crop).set_default(false)
    .describe("If set to true, then it will use be the center_crop,"
      "or it will crop using the shape of crop_like");
  }
};  // struct CropParam

template<typename xpu>
class CropOp : public Operator {
 public:
  explicit CropOp(CropParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), param_.num_args);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req[crop_enum::kOut], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[crop_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[crop_enum::kOut].get<xpu, 4, real_t>(s);
    offset_hw_ = InferCropOfferset(data.shape_, out.shape_);
    out = crop(data, Shape2(out.size(2), out.size(3)), offset_hw_[0], offset_hw_[1]);
  }

  // because the crop_like input is only used with it's shape, so we should be
  // careful setting its backwrd grad value to zeros, so that it will not hurt
  // the connection of crop_like.
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), param_.num_args) << in_grad.size();
    CHECK_EQ(out_grad.size(), 1) << out_grad.size();
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[crop_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = in_grad[crop_enum::kData].get<xpu, 4, real_t>(s);
    if (param_.num_args > 1) {
      // here backward grad is set to zero for crop_like
      // however, this should only be done when num_args > 1, i.e., crop_like exists
      Tensor<xpu, 4> gcrop_like = in_grad[crop_enum::kCropLike].get<xpu, 4, real_t>(s);
      gcrop_like = (real_t)0.0f;
    }
    offset_hw_ = InferCropOfferset(gdata.shape_, grad.shape_);
    gdata = (real_t)0.0f;
    slice<3>(slice<2>(gdata, offset_hw_[0], offset_hw_[0]+grad.size(2)),
             offset_hw_[1], offset_hw_[1]+grad.size(3)) = grad;
  }

 private:
  CropParam param_;
  std::vector<int> offset_hw_;
  std::vector<int> InferCropOfferset(const mshadow::Shape<4> &data_shape,
                                 const mshadow::Shape<4> &out_shape) {
      std::vector<int> offset_hw;
      CHECK_GE(data_shape[2], out_shape[2]) <<
          "data_shape'height should be larger than that of out_shape";
      CHECK_GE(data_shape[3], out_shape[3]) <<
          "data_shape'weight should be larger than that of out_shape";
      if (param_.center_crop) {
        offset_hw.push_back(static_cast<int>((data_shape[2]-out_shape[2])/2));
        offset_hw.push_back(static_cast<int>((data_shape[3]-out_shape[3])/2));
      } else {
        CHECK_GE(static_cast<int>(param_.offset[0]), 0) <<
            "offset[0] should be larger than 0";
        CHECK_LE(static_cast<int>(param_.offset[0]), data_shape[2]-out_shape[2]) <<
            "offset[0] should be less than the residual space of height";
        CHECK_GE(static_cast<int>(param_.offset[1]), 0) <<
            "offset[1] should be larger than 0";
        CHECK_LE(static_cast<int>(param_.offset[1]), data_shape[3]-out_shape[3]) <<
            "offset[1] should be less than the residual space of width";
        offset_hw.push_back(static_cast<int>(param_.offset[0]));
        offset_hw.push_back(static_cast<int>(param_.offset[1]));
      }
      return offset_hw;
  }
};  // class CropOp

template<typename xpu>
Operator *CreateOp(CropParam param);

#if DMLC_USE_CXX11
class CropProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    // return {"data", "crop_like"};
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_args; ++i) {
      ret.push_back(std::string("arg") + static_cast<char>('0' + i));
    }
    return ret;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args));
    TShape data_shape = in_shape->at(crop_enum::kData);
    if (data_shape.ndim() == 0) return false;
    CHECK_EQ(data_shape.ndim(), 4) << \
        "Input data should be 4D in batch-num_filter-y-x";
    std::vector<int> crop_shape;
    if (param_.num_args == 1) {
      CHECK_GE(static_cast<int>(param_.h_w[0]), 1) <<
          "the crop height(h_w[0]) should be larger than 1";
      CHECK_LE(static_cast<int>(param_.h_w[0]), static_cast<int>(data_shape[2])) <<
          "the crop height(h_w[0]) should be less than the input data's height";
      CHECK_GE(static_cast<int>(param_.h_w[1]), 1) <<
          "the crop width(h_w[1]) should be larger than 1";
      CHECK_LE(static_cast<int>(param_.h_w[1]), static_cast<int>(data_shape[3])) <<
          "the crop width(h_w[1]) should be less than the input data's width";
      crop_shape.push_back(param_.h_w[0]);
      crop_shape.push_back(param_.h_w[1]);
    } else if (param_.num_args == 2) {
      TShape crop_like_shape = in_shape->at(crop_enum::kCropLike);
      crop_shape.push_back(crop_like_shape[2]);
      crop_shape.push_back(crop_like_shape[3]);
    }
    if (crop_shape.size() == 0) return false;
    CHECK_EQ(crop_shape.size(), 2) << \
        "Input crop_like should be 2D in height-width";
    out_shape->clear();
    data_shape[2] = crop_shape[0];
    data_shape[3] = crop_shape[1];
    out_shape->push_back(data_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new CropProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Crop";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return out_grad;
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  CropParam param_;
};  // class CropProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CROP_INL_H_
