/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
#define MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_

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

namespace softmaxout_enum {
enum SoftmaxOutputOpInputs {kData, kLabel};
enum SoftmaxOutputOpOutputs {kOut};
}  // namespace softmaxout_enum

struct SoftmaxOutputParam : public dmlc::Parameter<SoftmaxOutputParam> {
  float grad_scale;
  float ignore_label;
  bool multi_output;
  bool use_ignore;
  DMLC_DECLARE_PARAMETER(SoftmaxOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("the ignore_label will not work in backward, and this only "
      "be used when multi_output=true");
    DMLC_DECLARE_FIELD(multi_output).set_default(false)
    .describe("If set to true, for a (n,k,x_1,..,x_n) dimensional "
      "input tensor, softmax will generate n*x_1*...*x_n output, each "
      "has k classes");
    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
    .describe("If set to true, the ignore_label value will not contribute "
      "to the backward gradient");
  };
};

template<typename xpu>
class SoftmaxOutputOp : public Operator {
 public:
  explicit SoftmaxOutputOp(SoftmaxOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "SoftmaxOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "SoftmaxOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.multi_output) {
      int n = in_data[softmaxout_enum::kData].size(0);
      int k = in_data[softmaxout_enum::kData].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(in_data[softmaxout_enum::kData].Size()/n/k));
      Tensor<xpu, 3> data = in_data[softmaxout_enum::kData].get_with_shape<xpu, 3, real_t>(s3, s);
      Tensor<xpu, 3> out = out_data[softmaxout_enum::kOut].get_with_shape<xpu, 3, real_t>(s3, s);
      Softmax(out, data);
    } else {
      Tensor<xpu, 2> data = in_data[softmaxout_enum::kData].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> out = out_data[softmaxout_enum::kOut].FlatTo2D<xpu, real_t>(s);
      Softmax(out, data);
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
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.multi_output) {
      int n = out_data[softmaxout_enum::kOut].size(0);
      int k = out_data[softmaxout_enum::kOut].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<int>(out_data[softmaxout_enum::kOut].Size()/n/k));
      Tensor<xpu, 2> label = in_data[softmaxout_enum::kLabel].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 3> out = out_data[softmaxout_enum::kOut].get_with_shape<xpu, 3, real_t>(s3, s);
      Tensor<xpu, 3> grad = in_grad[softmaxout_enum::kData].get_with_shape<xpu, 3, real_t>(s3, s);
      if (param_.use_ignore) {
          SoftmaxGrad(grad, out, label, static_cast<real_t>(param_.ignore_label));
      } else {
          SoftmaxGrad(grad, out, label);
      }
      grad *= param_.grad_scale/s3[2];
    } else {
      Tensor<xpu, 1> label = in_data[softmaxout_enum::kLabel].get<xpu, 1, real_t>(s);
      Tensor<xpu, 2> out = out_data[softmaxout_enum::kOut].FlatTo2D<xpu, real_t>(s);
      Tensor<xpu, 2> grad = in_grad[softmaxout_enum::kData].FlatTo2D<xpu, real_t>(s);
      SoftmaxGrad(grad, out, label);
      grad *= param_.grad_scale;
    }
  }

 private:
  SoftmaxOutputParam param_;
};  // class SoftmaxOutputOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxOutputParam param);

#if DMLC_USE_CXX11
class SoftmaxOutputProp : public OperatorProperty {
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
    if (param_.multi_output) {
      SHAPE_ASSIGN_CHECK(*in_shape, softmaxout_enum::kLabel,
                         Shape2(dshape[0], dshape.Size()/dshape[0]/dshape[1]));
    } else {
      SHAPE_ASSIGN_CHECK(*in_shape, softmaxout_enum::kLabel, Shape1(dshape[0]));
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SoftmaxOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SoftmaxOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[softmaxout_enum::kLabel], out_data[softmaxout_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[softmaxout_enum::kOut], in_grad[softmaxout_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softmaxout_enum::kData], out_data[softmaxout_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 protected:
  SoftmaxOutputParam param_;
};  // class SoftmaxOutputProp

class DeprecatedSoftmaxProp : public SoftmaxOutputProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    LOG(INFO) << "Softmax symbol is renamed to SoftmaxOutput. "
      << "This API will be deprecated in Dec, 2015";
    SoftmaxOutputProp::param_.Init(kwargs);
  }

  std::string TypeString() const override {
    return "Softmax";
  }
};
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
