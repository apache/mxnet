/*!
 * Copyright (c) 2015, 2016 by Contributors
 * \file dropout_allchannels-inl.h
 * \brief
 * \author Bing Xu, Kai Londenberg
*/

#ifndef MXNET_OPERATOR_DROPOUT_ALLCHANNELS_INL_H_
#define MXNET_OPERATOR_DROPOUT_ALLCHANNELS_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "mshadow/extension/broadcast_with_axis.h"


namespace dropout_allchannels {
enum DropoutAllChannelsOpInputs {kData};
enum DropoutAllChannelsOpOutputs {kOut, kMask};
enum DropoutAllChannelsOpForwardResource {kRandom};
}  // namespace dropout_allchannels

namespace mxnet {
namespace op {

struct DropoutAllChannelsParam : public dmlc::Parameter<DropoutAllChannelsParam> {
  float p;
  DMLC_DECLARE_PARAMETER(DropoutAllChannelsParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out at training time");
  }
};  // struct DropoutParam

template<typename xpu>
class DropoutAllChannelsOp : public Operator {
 public:
  explicit DropoutAllChannelsOp(DropoutAllChannelsParam param) {
    this->pkeep_ = 1.0f - param.p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 2);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[dropout_allchannels::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[dropout_allchannels::kOut].get<xpu, 4, real_t>(s);
    if (ctx.is_train) {
      Tensor<xpu, 4> mask = out_data[dropout_allchannels::kMask].get<xpu, 4, real_t>(s);
      Random<xpu> *prnd = ctx.requested[dropout_allchannels::kRandom].get_random<xpu, real_t>(s);
      Shape<3> subshp = Shape3(mask.shape_[0], mask.shape_[2], mask.shape_[3]);
      mask = broadcast_with_axis(F<mshadow_op::threshold>(prnd->uniform(subshp), pkeep_)
                                                       * (1.0f / pkeep_), 0, mask.shape_[1]);
      Assign(out, req[dropout_allchannels::kOut], data * mask);
    } else {
      Assign(out, req[dropout_allchannels::kOut], F<mshadow_op::identity>(data));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[dropout_allchannels::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> mask = out_data[dropout_allchannels::kMask].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gdata = in_grad[dropout_allchannels::kData].get<xpu, 4, real_t>(s);
    Assign(gdata, req[dropout_allchannels::kData], grad * mask);
  }

 private:
  real_t pkeep_;
};  // class DropoutOp


template<typename xpu>
Operator *CreateOp(DropoutAllChannelsParam param);

#if DMLC_USE_CXX11
class DropoutAllChannelsProp : public OperatorProperty {
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
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DropoutAllChannelsProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "DropoutAllChannels";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[dropout_allchannels::kOut], out_data[dropout_allchannels::kMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[dropout_allchannels::kOut], in_grad[dropout_allchannels::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[dropout_allchannels::kData], out_data[dropout_allchannels::kOut]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kRandom};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  DropoutAllChannelsParam param_;
};  // class DropoutAllChannelsProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DROPOUT_ALLCHANNELS_INL_H_

