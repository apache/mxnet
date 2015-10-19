/*!
 * Copyright (c) 2015 by Contributors
 * \file regression_ouput-inl.h
 * \brief Regression output operator.
 */
#ifndef MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
#define MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
enum RegressionOutputOpInputs {kData, kLabel};
enum RegressionOutputOutputs {kOut};
enum RegressionOutputType {kLinear, kLogistic};

// Special Operator to output regression value in forward
// And get gradient in calculation.
template<typename xpu, typename ForwardOp, typename BackwardOp>
class RegressionOutputOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "RegressionOutputOp Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "RegressionOutputOp Output: [output]";
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
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> label = in_data[kLabel].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = in_grad[kData].FlatTo2D<xpu, real_t>(s);
    Assign(grad, req[kData], F<BackwardOp>(out, reshape(label, grad.shape_)));
  }
};

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateRegressionOutputOp(RegressionOutputType type);

#if DMLC_USE_CXX11
template<RegressionOutputType type>
class RegressionOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
  }

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    CHECK_EQ(dshape[1], 1) << TypeString() << " requires input's num_hidden=1.";
    SHAPE_ASSIGN_CHECK(*in_shape, 1, Shape1(dshape[0]));
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    return new RegressionOutputProp<type>();
  }

  std::string TypeString() const override {
    switch (type) {
      case kLinear: return "LinearRegressionOutput";
      case kLogistic: return "LogisticRegressionOutput";
      default: LOG(FATAL) << "unknown type"; return "";
    }
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
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
