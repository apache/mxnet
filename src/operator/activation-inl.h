/*!
 * Copyright (c) 2015 by Contributors
 * \file activation-inl.h
 * \brief Activation operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ACTIVATION_INL_H_
#define MXNET_OPERATOR_ACTIVATION_INL_H_

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
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpType {kReLU, kSigmoid, kTanh};

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", kReLU)
    .add_enum("sigmoid", kSigmoid)
    .add_enum("tanh", kTanh)
    .describe("Activation function to be applied.");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp>
class ActivationOp : public Operator {
 public:
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
    Tensor<xpu, 2> data = in_data[kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[kOut], F<ForwardOp>(data));
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> m_out_grad = out_grad[kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> m_out_data = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> m_in_grad = in_grad[kData].FlatTo2D<xpu, real_t>(s);
    Assign(m_in_grad, req[kData], F<BackwardOp>(m_out_data) * m_out_grad);
  }
};  // class ActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ActivationParam type);

#if DMLC_USE_CXX11
class ActivationProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Activation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[kOut], out_data[kOut], in_data[kData]};
#else
    return {out_grad[kOut], out_data[kOut]};
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[kOut], in_grad[kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[kData], out_data[kOut]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  ActivationParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ACTIVATION_INL_H_
