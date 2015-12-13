/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu-inl.h
 * \brief leaky relu family operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_PROD_SUM_INL_H_
#define MXNET_OPERATOR_PROD_SUM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace prodsum {
enum ProdSumOpInputs {kLhs, kRhs};
enum ProdSumOpOutputs {kOut};
}  // namespace prodsum

struct ProdSumParam : public dmlc::Parameter<ProdSumParam> {
  index_t dot_dim;
  DMLC_DECLARE_PARAMETER(ProdSumParam) {
    DMLC_DECLARE_FIELD(dot_dim)
    .describe("The dimension along with to do dot product.");
  }
};

template<typename xpu>
class ProdSumOp : public Operator {
 public:
  explicit ProdSumOp(ProdSumParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    TShape lshape = in_data[prodsum::kLhs].shape_;
    Shape<3> ishape = ShapeCheck(in_data);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3> lhs = in_data[prodsum::kLhs]
      .get_with_shape<xpu, 3, real_t>(ishape, s);
    Tensor<xpu, 3> rhs = in_data[prodsum::kRhs]
      .get_with_shape<xpu, 3, real_t>(ishape, s);
    Tensor<xpu, 2> out = out_data[prodsum::kOut]
      .get_with_shape<xpu, 2, real_t>(Shape2(ishape[0], ishape[2]), s);
    Assign(out, req[prodsum::kOut], (reduce_with_axis<red::sum, 1>(lhs*rhs)));
  }

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    TShape lshape = in_data[prodsum::kLhs].shape_;
    Shape<3> ishape = ShapeCheck(in_data);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3> lhs = in_data[prodsum::kLhs]
      .get_with_shape<xpu, 3, real_t>(ishape, s);
    Tensor<xpu, 3> lhs_grad = in_grad[prodsum::kLhs]
      .get_with_shape<xpu, 3, real_t>(ishape, s);
    Tensor<xpu, 3> rhs = in_data[prodsum::kRhs]
      .get_with_shape<xpu, 3, real_t>(ishape, s);
    Tensor<xpu, 3> rhs_grad = in_grad[prodsum::kRhs]
      .get_with_shape<xpu, 3, real_t>(ishape, s);
    Tensor<xpu, 2> top = out_grad[prodsum::kOut]
      .get_with_shape<xpu, 2, real_t>(Shape2(ishape[0], ishape[2]), s);
    Assign(lhs_grad, req[prodsum::kLhs], (broadcast_with_axis<0>(top, ishape[1])*rhs));
    Assign(rhs_grad, req[prodsum::kRhs], (broadcast_with_axis<0>(top, ishape[1])*lhs));
  }

 private:
  ProdSumParam param_;

  mshadow::Shape<3> ShapeCheck(const std::vector<TBlob> &in_data) {
    index_t leading = 1, trailing = 1;
    TShape lshape = in_data[prodsum::kLhs].shape_;
    TShape rshape = in_data[prodsum::kRhs].shape_;
    CHECK_EQ(lshape, rshape) << "Shape of two inputs must match";
    CHECK(lshape.ndim() > param_.dot_dim)
      << "Inputs must have more dimensions than dot_dim";
    for (index_t i = 0; i < param_.dot_dim; ++i) {
      leading *= lshape[i];
    }
    for (index_t i = param_.dot_dim+1; i < lshape.ndim(); ++i) {
      trailing *= lshape[i];
    }
    return mshadow::Shape3(leading, lshape[param_.dot_dim], trailing);
  }
};  // class ProdSumOp

template<typename xpu>
Operator* CreateOp(ProdSumParam type);

#if DMLC_USE_CXX11
class ProdSumProp : public OperatorProperty {
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
    TShape lshape = in_shape->at(0);
    TShape rshape = in_shape->at(1);
    CHECK_EQ(lshape, rshape) << "Shape of two inputs must match";
    CHECK(lshape.ndim() > param_.dot_dim)
      << "Inputs must have more dimensions than dot_dim";
    std::vector<index_t> s;
    for (index_t i = 0; i < lshape.ndim(); ++i) {
      if (i != param_.dot_dim) {
        s.push_back(lshape[i]);
      }
    }
    TShape oshape(s.begin(), s.end());
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ProdSumProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "ProdSum";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[prodsum::kLhs], in_data[prodsum::kRhs], out_grad[prodsum::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"lhs", "rhs"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ProdSumParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_PROD_SUM_INL_H_

