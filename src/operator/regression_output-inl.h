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

namespace reg_enum {
enum RegressionOutputOpInputs {kData, kLabel};
enum RegressionOutputOutputs {kOut};
enum RegressionOutputType {kLinear, kLogistic, kMAE};
}  // reg_enum

struct RegressionOutputParam : public dmlc::Parameter<RegressionOutputParam> {
  float grad_scale;
  DMLC_DECLARE_PARAMETER(RegressionOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
  };
};

// Special Operator to output regression value in forward
// And get gradient in calculation.
template<typename xpu, typename ForwardOp, typename BackwardOp>
class RegressionOutputOp : public Operator {
 public:
  explicit RegressionOutputOp(RegressionOutputParam param) : param_(param) {}

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
    Tensor<xpu, 2> data = in_data[reg_enum::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[reg_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[reg_enum::kOut], F<ForwardOp>(data));
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
    real_t num_output =
      in_data[reg_enum::kLabel].Size()/in_data[reg_enum::kLabel].shape_[0];
    Tensor<xpu, 2> out = out_data[reg_enum::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = in_grad[reg_enum::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> label = in_data[reg_enum::kLabel]
      .get_with_shape<xpu, 2, real_t>(out.shape_, s);
    Assign(grad, req[reg_enum::kData], param_.grad_scale/num_output*
      F<BackwardOp>(out, reshape(label, grad.shape_)));
  }

 private:
  RegressionOutputParam param_;
};

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateRegressionOutputOp(reg_enum::RegressionOutputType type,
                                   RegressionOutputParam param);

#if DMLC_USE_CXX11
template<reg_enum::RegressionOutputType type>
class RegressionOutputProp : public OperatorProperty {
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
    auto &lshape = (*in_shape)[1];
    if (lshape.ndim() == 0) {
      // special treatment for 1D output, to allow 1D label by default.
      // Think about change convention later
      if (dshape.ndim() == 2 && dshape[1] == 1) {
        lshape = Shape1(dshape[0]);
      } else {
        lshape = dshape;
      }
    } else if (lshape[0] != dshape[0] || lshape.Size() != dshape.Size()) {
      std::ostringstream os;
      os << "Shape inconsistent, Provided " <<  '='<< lshape << ','
         << " inferred shape=" << dshape;
      throw ::mxnet::op::InferShapeError(os.str(), 1);
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new RegressionOutputProp<type>();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    switch (type) {
      case reg_enum::kLinear: return "LinearRegressionOutput";
      case reg_enum::kLogistic: return "LogisticRegressionOutput";
      case reg_enum::kMAE: return "MAERegressionOutput";
      default: LOG(FATAL) << "unknown type"; return "";
    }
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[reg_enum::kLabel], out_data[reg_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[reg_enum::kOut], in_grad[reg_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[reg_enum::kData], out_data[reg_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 protected:
  RegressionOutputParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_REGRESSION_OUTPUT_INL_H_
