/*!
 * Copyright (c) 2015 by Contributors
 * \file make_loss-inl.h
 * \brief special layer for propagating loss
*/
#ifndef MXNET_OPERATOR_MAKE_LOSS_INL_H_
#define MXNET_OPERATOR_MAKE_LOSS_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace make_loss_enum {
enum MakeLossOpInputs {kData};
enum MakeLossOpOutputs {kOut};
}  // namespace make_loss_enum

struct MakeLossParam : public dmlc::Parameter<MakeLossParam> {
  float grad_scale;
  DMLC_DECLARE_PARAMETER(MakeLossParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("gradient scale as a supplement to unary and binary operators");
  }
};

template<typename xpu>
class MakeLossOp : public Operator {
 public:
  explicit MakeLossOp(MakeLossParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                        const std::vector<TBlob> &in_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &out_data,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1) << "MakeLoss can only be used to one input";
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(in_data[make_loss_enum::kData].ndim(), 2)
    << "MakeLoss applies to all unary and binary operator with 2 dimension input";
    if (req[make_loss_enum::kOut] != kWriteInplace) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Tensor<xpu, 2> data = in_data[make_loss_enum::kData].get<xpu, 2, real_t>(s);
      Tensor<xpu, 2> out = out_data[make_loss_enum::kOut].get<xpu, 2, real_t>(s);
      Assign(out, req[make_loss_enum::kOut], data);
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
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> grad = in_grad[make_loss_enum::kData].get<xpu, 2, real_t>(s);
    Assign(grad, req[make_loss_enum::kData], ScalarExp<real_t>(param_.grad_scale));
  }

 private:
  MakeLossParam param_;
};  // class MakeLossOp

template <typename xpu>
Operator *CreateOp(MakeLossParam param);

#if DMLC_USE_CXX11
class MakeLossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  };

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1);
    const TShape &dshape = in_shape->at(make_loss_enum::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MakeLossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MakeLoss";
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const override {
    return {{in_data[make_loss_enum::kData], out_data[make_loss_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  MakeLossParam param_;
};  // class MakeLossProperty

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MAKE_LOSS_INL_H_
