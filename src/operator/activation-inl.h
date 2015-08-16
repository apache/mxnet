/*!
 * Copyright (c) 2015 by Contributors
 * \file activation-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ACTIVATION_INL_H_
#define MXNET_OPERATOR_ACTIVATION_INL_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <cstring>
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
enum ActivationOpType {kUnknown, kReLU, kSigmoid, kTanh};
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
                       const std::vector<TBlob> &out_data) {
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
                        const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out_gradient = out_grad[kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> output = out_data[kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = out_grad[kOut].FlatTo2D<xpu, real_t>(s);
    Assign(grad, req[kData], F<BackwardOp>(out_gradient * output));
  }
};  // class ActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateActivationOp(ActivationOpType type);

#if DMLC_USE_CXX11
class ActivationProp : public OperatorProperty {
 public:
  explicit ActivationProp() : type_(kUnknown) {}

  explicit ActivationProp(ActivationOpType type) : type_(type) {}

  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "type")) {
      if (!strcmp(val, "relu")) type_ = kReLU;
      if (!strcmp(val, "sigmoid")) type_ = kSigmoid;
      if (!strcmp(val, "tanh")) type_ = kTanh;
    }
    CHECK(type_ >= kReLU && type_ <= kTanh) << "Invalid activation type";
  }
  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape) const {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  virtual OperatorProperty* Copy() const {
    auto ptr = new ActivationProp(type_);
    return ptr;
  }

  virtual std::string TypeString() const {
    return "Activation";
  }

  // decalre dependency and inplace optimization options
  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {out_grad[kOut], out_data[kData]};
  }

  virtual std::vector<std::pair<int, int> > BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<int> &in_grad) const {
    return {{out_grad[kData], in_grad[kData]}};
  }

  virtual std::vector<std::pair<int, int> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {{in_data[kData], out_data[kData]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  ActivationOpType type_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ACTIVATION_INL_H_

