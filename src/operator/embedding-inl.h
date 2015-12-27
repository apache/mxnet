/*!
 * Copyright (c) 2015 by Contributors
 * \file embedding-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_EMBEDDING_INL_H_
#define MXNET_OPERATOR_EMBEDDING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace embedding {
enum EmbeddingOpInputs {kData, kWeight};
enum EmbeddingOpOutputs {kOut};
}  // namespace embedding

struct EmbeddingParam: public dmlc::Parameter<EmbeddingParam> {
  int input_dim;
  int output_dim;
  DMLC_DECLARE_PARAMETER(EmbeddingParam) {
    DMLC_DECLARE_FIELD(input_dim).set_lower_bound(1)
    .describe("input dim of one-hot encoding");
    DMLC_DECLARE_FIELD(output_dim).set_lower_bound(1)
    .describe("output dim of embedding");
  }
};


template<typename xpu>
class EmbeddingOp : public Operator {
 public:
  explicit EmbeddingOp(EmbeddingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[embedding::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(in_data[embedding::kData].ndim(), 1)
        << "Embedding layer expects its input to be one-dimensional. "
        << in_data[embedding::kData].ndim()
        << " dimensional input is given instead";
    CHECK_EQ(in_data[embedding::kWeight].ndim(), 2)
        << "Embedding layer expects its weight to be two-dimensional. "
        << in_data[embedding::kWeight].ndim()
        << " dimensional input is given instead";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> data = in_data[embedding::kData].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[embedding::kWeight].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out = out_data[embedding::kOut].get<xpu, 2, real_t>(s);
    out = take(data, wmat);
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
    CHECK_GE(in_data.size(), 1);
    CHECK_EQ(in_grad.size(), 2);
    CHECK_EQ(req[embedding::kData], kNullOp)
      << "Embedding layer doesn't support calculate data gradient";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> data = in_data[embedding::kData].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> grad_out = out_grad[embedding::kOut].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> grad_in = in_grad[embedding::kWeight].get<xpu, 2, real_t>(s);
    grad_in = take_grad(data, grad_out, param_.input_dim);
  }

 private:
  EmbeddingParam param_;
};  // class EmbeddingOp

template<typename xpu>
Operator* CreateOp(EmbeddingParam param);

#if DMLC_USE_CXX11
class EmbeddingProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "weight"};
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
    const TShape &dshape = (*in_shape)[embedding::kData];
    if (dshape.ndim() ==  0) return false;
    SHAPE_ASSIGN_CHECK(*in_shape, embedding::kWeight, Shape2(param_.input_dim,
                                                          param_.output_dim));
    out_shape->clear();
    out_shape->push_back(Shape2(dshape[0], param_.output_dim));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto sym = new EmbeddingProp();
    sym->param_ = this->param_;
    return sym;
  }

  std::string TypeString() const override {
    return "Embedding";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[embedding::kOut], in_data[embedding::kData]};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  EmbeddingParam param_;
};  // class EmbeddingProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_EMBEDDING_INL_H_
