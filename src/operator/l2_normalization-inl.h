/*!
 * Copyright (c) 2016 by Contributors
 * \file l2_normalization_op-inl.h
 * \brief instance l2 Normalization op
*/
#ifndef MXNET_OPERATOR_L2_NORMALIZATION_INL_H_
#define MXNET_OPERATOR_L2_NORMALIZATION_INL_H_

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

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace l2_normalization {
enum L2NormalizationOpInputs {kData};
enum L2NormalizationOpOutputs {kOut, kNorm};
enum L2NormalizationBackResource {kTempSpace};
}  // l2_normalization

struct L2NormalizationParam : public dmlc::Parameter<L2NormalizationParam> {
  float eps;
  DMLC_DECLARE_PARAMETER(L2NormalizationParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-10f)
    .describe("Epsilon to prevent div 0");
  }
};

/**
 * \brief This is the implementation of l2 normalization operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class L2NormalizationOp : public Operator {
 public:
  explicit L2NormalizationOp(L2NormalizationParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[l2_normalization::kOut] == kNullOp) return;
    CHECK_EQ(req[l2_normalization::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[l2_normalization::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[l2_normalization::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 1> norm = out_data[l2_normalization::kNorm].get<xpu, 1, real_t>(s);
    norm = sumall_except_dim<0>(F<mxnet::op::mshadow_op::square>(data));
    norm = F<mxnet::op::mshadow_op::square_root>(norm);
    out = data / broadcast<0>(norm + param_.eps, out.shape_);
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
    Tensor<xpu, 2> data = out_data[l2_normalization::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad_in = in_grad[l2_normalization::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad_out = out_grad[l2_normalization::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 1> norm = out_data[l2_normalization::kNorm].get<xpu, 1, real_t>(s);

    Tensor<xpu, 1> temp = ctx.requested[l2_normalization::kTempSpace].get_space<xpu>(
        mshadow::Shape1(data.shape_[0]), s);
    temp = sumall_except_dim<0>(grad_out * data);

    Assign(grad_in, req[l2_normalization::kData],
      (grad_out - data * broadcast<0>(temp, data.shape_)) /
      broadcast<0>(norm + param_.eps, data.shape_));
  }

 private:
  L2NormalizationParam param_;
};  // class L2NormalizationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(L2NormalizationParam param);

#if DMLC_USE_CXX11
class L2NormalizationProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "norm"};
  }

  int NumVisibleOutputs() const override {
    return 1;
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
    CHECK_EQ(in_shape->size(), 1) << "L2Normalization layer only accepts data as input";
    const TShape &dshape = (*in_shape)[l2_normalization::kData];
    // require data to be known
    if ((*in_shape)[l2_normalization::kData].ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[0]));
    return true;
  }

  OperatorProperty* Copy() const override {
    L2NormalizationProp* norm_sym = new L2NormalizationProp();
    norm_sym->param_ = this->param_;
    return norm_sym;
  }

  std::string TypeString() const override {
    return "L2Normalization";
  }

  // declare dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[l2_normalization::kOut],
      out_data[l2_normalization::kOut],
      out_data[l2_normalization::kNorm]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[l2_normalization::kOut], in_grad[l2_normalization::kData]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  L2NormalizationParam param_;
};  // class L2NormalizationSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_L2_NORMALIZATION_INL_H_
