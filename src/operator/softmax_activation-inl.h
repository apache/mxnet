/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_activation-inl.h
 * \brief SoftmaxActivation operator
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_SOFTMAX_ACTIVATION_INL_H_
#define MXNET_OPERATOR_SOFTMAX_ACTIVATION_INL_H_

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
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace softmax_activation {
enum SoftmaxActivationOpInputs {kData};
enum SoftmaxActivationOpOutputs {kOut};
enum SoftmaxActivationOpType {kInstance, kChannel};
}  // softmax_activation

struct SoftmaxActivationParam : public dmlc::Parameter<SoftmaxActivationParam> {
  // use int for enumeration
  int type;
  DMLC_DECLARE_PARAMETER(SoftmaxActivationParam) {
    DMLC_DECLARE_FIELD(type)
    .add_enum("instance", softmax_activation::kInstance)
    .add_enum("channel", softmax_activation::kChannel)
    .set_default(softmax_activation::kInstance)
    .describe("Softmax Mode. If set to instance, this operator will compute a "
    "softmax for each instance in the batch; this is the default mode. "
    "If set to channel, this operator will compute a num_channel-class softmax at "
    "each position of each instance; this can be used for fully convolutional network, "
    "image segmentation, etc.");
  }
};

/**
 * \brief This is the implementation of softmax_activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class SoftmaxActivationOp : public Operator {
 public:
  explicit SoftmaxActivationOp(SoftmaxActivationParam p) {
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
    // Stream<xpu> *s = ctx.get_stream<xpu>();
    // Tensor<xpu, 2> data = in_data[softmax_activation::kData].FlatTo2D<xpu, real_t>(s);
    // Tensor<xpu, 2> out = out_data[softmax_activation::kOut].FlatTo2D<xpu, real_t>(s);
    LOG(FATAL) << "non-cuDNN version not implemented yet.";
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
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    // Stream<xpu> *s = ctx.get_stream<xpu>();
    // Tensor<xpu, 2> m_out_grad = out_grad[softmax_activation::kOut].FlatTo2D<xpu, real_t>(s);
    // Tensor<xpu, 2> m_out_data = out_data[softmax_activation::kOut].FlatTo2D<xpu, real_t>(s);
    // Tensor<xpu, 2> m_in_grad = in_grad[softmax_activation::kData].FlatTo2D<xpu, real_t>(s);
    LOG(FATAL) << "non-cuDNN version not implemented yet.";
  }

 private:
  SoftmaxActivationParam param_;
};  // class SoftmaxActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxActivationParam type);

#if DMLC_USE_CXX11
class SoftmaxActivationProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(softmax_activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SoftmaxActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SoftmaxActivation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[softmax_activation::kOut], out_data[softmax_activation::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[softmax_activation::kOut], in_grad[softmax_activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softmax_activation::kData], out_data[softmax_activation::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  SoftmaxActivationParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTMAX_ACTIVATION_INL_H_
