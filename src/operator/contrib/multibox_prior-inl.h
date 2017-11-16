/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_prior-inl.h
 * \brief generate multibox prior boxes
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_MULTIBOX_PRIOR_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTIBOX_PRIOR_INL_H_
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


namespace mxnet {
namespace op {

namespace mshadow_op {
struct clip_zero_one {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    if (a < 0.f) return DType(0.f);
    if (a > 1.f) return DType(1.f);
    return DType(a);
  }
};  // struct clip_zero_one
}  // namespace mshadow_op

namespace mboxprior_enum {
enum MultiBoxPriorOpInputs {kData};
enum MultiBoxPriorOpOutputs {kOut};
}  // namespace mboxprior_enum

struct MultiBoxPriorParam : public dmlc::Parameter<MultiBoxPriorParam> {
  nnvm::Tuple<float> sizes;
  nnvm::Tuple<float> ratios;
  bool clip;
  nnvm::Tuple<float> steps;
  nnvm::Tuple<float> offsets;
  DMLC_DECLARE_PARAMETER(MultiBoxPriorParam) {
    DMLC_DECLARE_FIELD(sizes).set_default({1.0f})
    .describe("List of sizes of generated MultiBoxPriores.");
    DMLC_DECLARE_FIELD(ratios).set_default({1.0f})
    .describe("List of aspect ratios of generated MultiBoxPriores.");
    DMLC_DECLARE_FIELD(clip).set_default(false)
    .describe("Whether to clip out-of-boundary boxes.");
    DMLC_DECLARE_FIELD(steps).set_default({-1.f, -1.f})
    .describe("Priorbox step across y and x, -1 for auto calculation.");
    DMLC_DECLARE_FIELD(offsets).set_default({0.5f, 0.5f})
    .describe("Priorbox center offsets, y and x respectively");
  }
};  // struct MultiBoxPriorParam

template<typename xpu, typename DType>
class MultiBoxPriorOp : public Operator {
 public:
  explicit MultiBoxPriorOp(MultiBoxPriorParam param)
    : clip_(param.clip), sizes_(param.sizes.begin(), param.sizes.end()),
    ratios_(param.ratios.begin(), param.ratios.end()),
    steps_(param.steps.begin(), param.steps.end()),
    offsets_(param.offsets.begin(), param.offsets.end()) {
      CHECK_GT(sizes_.size(), 0);
      CHECK_GT(ratios_.size(), 0);
      CHECK_EQ(steps_.size(), 2);
      CHECK_EQ(offsets_.size(), 2);
      CHECK_GE(offsets_[0], 0.f);
      CHECK_LE(offsets_[0], 1.f);
      CHECK_GE(offsets_[1], 0.f);
      CHECK_LE(offsets_[1], 1.f);
    }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> out;
    // TODO(zhreshold): this implementation is to be compliant to original ssd in caffe
    // The prior boxes could be implemented in more versatile ways
    // since input sizes are same in each batch, we could share MultiBoxPrior
    const int num_sizes = static_cast<int>(sizes_.size());
    const int num_ratios = static_cast<int>(ratios_.size());
    const int num_anchors = num_sizes - 1 + num_ratios;  // anchors per location
    int in_height = in_data[mboxprior_enum::kData].size(2);
    int in_width = in_data[mboxprior_enum::kData].size(3);
    Shape<2> oshape = Shape2(num_anchors * in_width * in_height, 4);
    out = out_data[mboxprior_enum::kOut].get_with_shape<xpu, 2, DType>(oshape, s);
    CHECK_GE(steps_[0] * steps_[1], 0) << "Must specify both step_y and step_x";
    if (steps_[0] <= 0 || steps_[1] <= 0) {
      // estimate using layer shape
      steps_[0] = 1.f / in_height;
      steps_[1] = 1.f / in_width;
    }
    MultiBoxPriorForward(out, sizes_, ratios_, in_width, in_height, steps_, offsets_);

    if (clip_) {
      Assign(out, req[mboxprior_enum::kOut], F<mshadow_op::clip_zero_one>(out));
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad = in_grad[mboxprior_enum::kData].FlatTo2D<xpu, DType>(s);
    grad = 0.f;
  }

 private:
  bool clip_;
  std::vector<float> sizes_;
  std::vector<float> ratios_;
  std::vector<float> steps_;
  std::vector<float> offsets_;
};  // class MultiBoxPriorOp

template<typename xpu>
Operator *CreateOp(MultiBoxPriorParam, int dtype);

#if DMLC_USE_CXX11
class MultiBoxPriorProp: public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Inputs: [data]" << in_shape->size();
    TShape dshape = in_shape->at(mboxprior_enum::kData);
    CHECK_GE(dshape.ndim(), 4) << "Input data should be 4D: batch-channel-y-x";
    int in_height = dshape[2];
    CHECK_GT(in_height, 0) << "Input height should > 0";
    int in_width = dshape[3];
    CHECK_GT(in_width, 0) << "Input width should > 0";
    // since input sizes are same in each batch, we could share MultiBoxPrior
    TShape oshape = TShape(3);
    int num_sizes = param_.sizes.ndim();
    int num_ratios = param_.ratios.ndim();
    oshape[0] = 1;
    oshape[1] = in_height * in_width * (num_sizes + num_ratios - 1);
    oshape[2] = 4;
    out_shape->clear();
    out_shape->push_back(oshape);
    CHECK_EQ(param_.steps.ndim(), 2) << "Step ndim must be 2: (step_y, step_x)";
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiBoxPriorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_MultiBoxPrior";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  MultiBoxPriorParam param_;
};  // class MultiBoxPriorProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_MULTIBOX_PRIOR_INL_H_
