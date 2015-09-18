/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SOFTMAX_INL_H_
#define MXNET_OPERATOR_SOFTMAX_INL_H_

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

enum SoftmaxOpInputs {kData, kLabel};
enum SoftmaxOpOutputs {kOut};

struct SoftmaxParam : public dmlc::Parameter<SoftmaxParam> {
  float grad_scale;
  DMLC_DECLARE_PARAMETER(SoftmaxParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
  };
};

template<typename xpu>
class SoftmaxOp : public Operator {
 public:
  explicit SoftmaxOp(SoftmaxParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "Softmax Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "Softmax Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    Softmax(out, data);
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
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> label = in_data[kLabel].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = in_grad[kData].FlatTo2D<xpu, real_t>(s);
    SoftmaxGrad(grad, out, label);
    if (param_.grad_scale < 1.0) {
      grad *= param_.grad_scale;
    }
  }

 private:
  SoftmaxParam param_;
};  // class SoftmaxOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxParam param);

#if DMLC_USE_CXX11
class SoftmaxProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    SHAPE_ASSIGN_CHECK(*in_shape, kLabel, Shape1(dshape[0]));
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SoftmaxProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Softmax";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[kLabel], out_data[kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[kOut], in_grad[kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[kData], out_data[kOut]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  SoftmaxParam param_;
};  // class SoftmaxProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTMAX_INL_H_
