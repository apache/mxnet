/*!
 * Copyright (c) 2015 by Contributors
 * \file upsampling-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_UPSAMPLING_NEAREST_INL_H_
#define MXNET_OPERATOR_UPSAMPLING_NEAREST_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace up_enum {
enum UpSamplingNearestOpInputs {kData};
enum UpSamplingNearestOpOutputs {kOut};
}  // namespace up_enum

struct UpSamplingNearestParam : public dmlc::Parameter<UpSamplingNearestParam> {
  index_t scale;
  DMLC_DECLARE_PARAMETER(UpSamplingNearestParam) {
    DMLC_DECLARE_FIELD(scale)
    .set_range(1, 1000)
    .describe("Up sampling scale");
  }
};  // struct UpSamplingNearestParam

template<typename xpu>
class UpSamplingNearestOp : public Operator {
 public:
  explicit UpSamplingNearestOp(UpSamplingNearestParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[up_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[up_enum::kOut].get<xpu, 4, real_t>(s);
    Assign(out, req[up_enum::kOut], upsampling_nearest(data, param_.scale));
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[up_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> input_grad = in_grad[up_enum::kData].get<xpu, 4, real_t>(s);
    mshadow::Shape<2> in_shape = Shape2(input_grad.shape_[2], input_grad.shape_[3]);
    Assign(input_grad, req[up_enum::kData],
           static_cast<float>(1.0f / param_.scale / param_.scale) * \
           pool<mshadow::red::sum>(grad,
                                   in_shape,
                                   param_.scale,
                                   param_.scale,
                                   param_.scale));
  }

 private:
  UpSamplingNearestParam param_;
};  // class UpSamplingNearestOp

template<typename xpu>
Operator *CreateOp(UpSamplingNearestParam param);


#if DMLC_USE_CXX11
class UpSamplingNearestProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = (*in_shape)[0];
    CHECK_EQ(dshape.ndim(), 4) << \
      "UpSamplingNearest: Input data should be 4D in (batch, channel, y, x)";
    if (dshape.ndim() ==  0) return false;
    TShape oshape = dshape;
    oshape[2] = dshape[2] * param_.scale;
    oshape[3] = dshape[3] * param_.scale;
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new UpSamplingNearestProp();
    ptr->param_ = this->param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "UpSamplingNearest";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[up_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[up_enum::kData], in_grad[up_enum::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  UpSamplingNearestParam param_;
};  // class UpSamplingNearestProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_UPSAMPLING_NEAREST_INL_H_

