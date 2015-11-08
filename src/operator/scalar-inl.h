/*!
 * Copyright (c) 2015 by Contributors
 * \file Scalar-inl.h
 * \brief Scalar operator
 * \author yajiedesign(ShiWen Hu)
*/
#ifndef MXNET_OPERATOR_SCALAR_INL_H_
#define MXNET_OPERATOR_SCALAR_INL_H_

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
namespace Scalar {
enum ScalarOpInputs {kData};
enum ScalarOpOutputs {kOut};
}  // Scalar

struct ScalarParam : public dmlc::Parameter<ScalarParam> {
  // use int for enumeration
  float value;
  DMLC_DECLARE_PARAMETER(ScalarParam) {
    DMLC_DECLARE_FIELD(value)
    .set_default(0)
    .describe("Scalar value.");
  }
};

/**
 * \brief This is the implementation of Scalar operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class ScalarOp : public Operator {
 public:
    explicit ScalarOp(ScalarParam param)
        : value_(param.value) {}
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> out = out_data[Scalar::kOut].FlatTo2D<xpu, real_t>(s);
    out = value_;
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
  }

 private:
     int value_;
};  // class ScalarOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ScalarParam type);

#if DMLC_USE_CXX11
class ScalarProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(Scalar::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ScalarProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Scalar";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[Scalar::kOut], out_data[Scalar::kOut], in_data[Scalar::kData]};
#else
    return {out_grad[Scalar::kOut], out_data[Scalar::kOut]};
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[Scalar::kOut], in_grad[Scalar::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[Scalar::kData], out_data[Scalar::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ScalarParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SCALAR_INL_H_
