/*!
 * Copyright (c) 2015 by Contributors
 * \file lrn-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LRN_INL_H_
#define MXNET_OPERATOR_LRN_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace lrn_enum {
enum LRNInputs {kData};
enum LRNOutputs {kOut, kTmpNorm};
}  // namespace lrn_enum

struct LRNParam : public dmlc::Parameter<LRNParam> {
  float alpha;
  float beta;
  float knorm;
  uint32_t nsize;
  DMLC_DECLARE_PARAMETER(LRNParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1e-4f)
    .describe("The variance scaling parameter :math:`\alpha` in the LRN expression.");
    DMLC_DECLARE_FIELD(beta).set_default(0.75f)
    .describe("The power parameter :math:`\beta` in the LRN expression.");
    DMLC_DECLARE_FIELD(knorm).set_default(2.0f)
    .describe("The parameter :math:`k` in the LRN expression.");
    DMLC_DECLARE_FIELD(nsize)
    .describe("normalization window width in elements.");
  }
};  // struct LRNParam

template<typename xpu>
class LocalResponseNormOp : public Operator {
 public:
  explicit LocalResponseNormOp(LRNParam param) {
    param_ = param;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(xxx): Test with gradient chceker
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    // CHECK_EQ(req.size(), 2);
    CHECK_EQ(param_.nsize % 2, 1U) << "LRN only supports odd values for local_size";
    const real_t salpha = param_.alpha / param_.nsize;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data = in_data[lrn_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out = out_data[lrn_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp_norm = out_data[lrn_enum::kTmpNorm].get<xpu, 4, real_t>(s);
    tmp_norm = chpool<red::sum>(F<mshadow_op::square>(data) , param_.nsize) * salpha + param_.knorm;
    Assign(out, req[lrn_enum::kOut], data *  F<mshadow_op::power>(tmp_norm, -param_.beta));
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
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    const real_t salpha = param_.alpha / param_.nsize;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad = out_grad[lrn_enum::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp_norm = out_data[lrn_enum::kTmpNorm].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data = in_data[lrn_enum::kData].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_in = in_grad[lrn_enum::kData].get<xpu, 4, real_t>(s);
    grad_in = grad * F<mshadow_op::power>(tmp_norm, -param_.beta);
    grad_in += (- 2.0f * param_.beta * salpha) *
               chpool<red::sum>(grad * data *
                                F<mshadow_op::power>(tmp_norm, -param_.beta - 1.0f),
                                param_.nsize)  * data;
  }

 private:
  LRNParam param_;
};  // class LocalResponseNormOp

template<typename xpu>
Operator *CreateOp(LRNParam param, int dtype);

#if DMLC_USE_CXX11
class LocalResponseNormProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    int n_out = this->ListOutputs().size();
    out_type->clear();
    for (int i = 0; i < n_out; ++i ) out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LocalResponseNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "LRN";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {
      out_grad[lrn_enum::kOut], in_data[lrn_enum::kData],
      out_data[lrn_enum::kTmpNorm], out_data[lrn_enum::kOut]
    };
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "tmp_norm"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  LRNParam param_;
};  // LocalResponseNormProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_LRN_INL_H_
