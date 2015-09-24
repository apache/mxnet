/*!
 * Copyright (c) 2015 by Contributors
 * \file reshape-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_RESHAPE_INL_H_
#define MXNET_OPERATOR_RESHAPE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

enum ReshapeOpInputs {kData};
enum ReshapeOpOutputs {kOut};

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  TShape target_shape;
  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    DMLC_DECLARE_FIELD(target_shape).describe("Target new shape");
  }
};

template<typename xpu>
class ReshapeOp : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    if (req[kOut] == kNullOp) return;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // TODO(bing): potentail bug here for non-4D input
    Tensor<xpu, 4> data = in_data[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[kOut].get<xpu, 4, real_t>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    if (data.dptr_ == out.dptr_) return;
    CHECK_EQ(data.shape_.Size(), out.shape_.Size());
    Assign(out, req[kOut], reshape(data, out.shape_));
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
    CHECK_EQ(req.size(), 1);
    if (req[kData] == kNullOp) return;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad_out = out_grad[kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_in = in_grad[kOut].get<xpu, 4, real_t>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    if (grad_out.dptr_ == grad_in.dptr_) return;
    CHECK_EQ(grad_out.shape_.Size(), grad_in.shape_.Size());
    Assign(grad_in, req[kData], reshape(grad_out, grad_in.shape_));
  }
};  // class ReshapeOp

template<typename xpu>
Operator* CreateOp();

#if DMLC_USE_CXX11
class ReshapeProp : public OperatorProperty {
 public:
  ReshapeProp() {}

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    const TShape &dshape = in_shape->at(kData);
    if (dshape.ndim() == 0) return false;
    CHECK(param_.target_shape.Size() == dshape.Size())
        << "Target shape size is different to source. "
        << "Target: " << param_.target_shape.Size()
        << "\nSource: " << dshape.Size();
    out_shape->clear();
    out_shape->push_back(param_.target_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ReshapeProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Reshape";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[kOut]};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[kData], out_data[kOut]}};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[kOut], in_grad[kData]}};
  }

  Operator* CreateOperator(Context ctx) const;

 protected:
  ReshapeParam param_;
};  // class ReshapeProp

class FlattenProp : public ReshapeProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {}

  std::map<std::string, std::string> GetParams() const override {
    // need to use this on osx
    return std::map<std::string, std::string>();
  }

  std::string TypeString() const override {
    return "Flatten";
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    const TShape &dshape = in_shape->at(kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    uint32_t target_dim = 1;
    for (uint32_t i = 1; i < dshape.ndim(); ++i) {
      target_dim *= dshape[i];
    }
    out_shape->push_back(mshadow::Shape4(dshape[0], 1, 1, target_dim));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new FlattenProp();
    return ptr;
  }
};  // class FlattenProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_RESHAPE_INL_H_
