/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_INL_H_

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
enum BatchNormOpInputs {kData, kGamma, kBeta};
enum BatchNormOpOutputs {kOut, kOutNoAffine, kMean, kVar};
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};
enum BatchNormBackResource {kTempSpace};

struct BatchNormParam : public dmlc::Parameter<BatchNormParam> {
  float eps;
  float momentum;
  DMLC_DECLARE_PARAMETER(BatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-10f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.1f)
    .describe("Momentum for moving average");
  }
};

template<typename xpu>
class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(BatchNormParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(aux_states.size(), 2);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 4);
      CHECK_EQ(req.size(), 4);
    } else {
      CHECK_GE(out_data.size(), 1);
      CHECK_GE(req.size(), 1);
      CHECK_EQ(req[kOut], kWriteTo);
    }

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[kData].shape_[1]) /
                         static_cast<real_t>(in_data[kData].shape_.Size());
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out, out_no_affine;
    if (in_data[kData].ndim() == 2) {
      uint32_t ds[] = {in_data[kData].shape_[0], in_data[kData].shape_[1], 1, 1};
      TShape dshape(ds, ds + 4);
      data = in_data[kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      if (ctx.is_train) {
        out_no_affine = out_data[kOutNoAffine].get_with_shape<xpu, 4, real_t>(dshape, s);
      }
    } else {
      data = in_data[kData].get<xpu, 4, real_t>(s);
      out = out_data[kOut].get<xpu, 4, real_t>(s);
      if (ctx.is_train) {
        out_no_affine = out_data[kOutNoAffine].get<xpu, 4, real_t>(s);
      }
    }
    Tensor<xpu, 1> slope = in_data[kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_mean = aux_states[kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[kMovingVar].get<xpu, 1, real_t>(s);
    // cal
    if (ctx.is_train) {
      Tensor<xpu, 1> mean = out_data[kMean].get<xpu, 1, real_t>(s);
      Tensor<xpu, 1> var = out_data[kVar].get<xpu, 1, real_t>(s);
      Assign(mean, req[kMean], scale * sumall_except_dim<1>(data));
      Assign(var, req[kVar], scale * sumall_except_dim<1>(
               F<mshadow_op::square>(data - broadcast<1>(mean, data.shape_))));
      Assign(out_no_affine, req[kOutNoAffine], (data - broadcast<1>(mean, data.shape_)) /
             F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_)));
      Assign(out, req[kOut], out_no_affine * broadcast<1>(slope, out.shape_) +
             broadcast<1>(bias, out.shape_));
      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
    } else {
      Assign(out, req[kOut], broadcast<1>(slope /
                                          F<mshadow_op::square_root>(moving_var + param_.eps),
                                          data.shape_) * data +
             broadcast<1>(bias - (slope * moving_mean) /
                          F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
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
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 4);
    CHECK_EQ(in_grad.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    Tensor<xpu, 4> out, out_no_affine;
    const real_t scale = static_cast<real_t>(out_data[kOut].shape_[1]) /
                         static_cast<real_t>(out_data[kOut].shape_.Size());
    if (in_data[kData].ndim() == 2) {
      uint32_t ds[] = {out_data[kOut].shape_[0], out_data[kOut].shape_[1], 1, 1};
      TShape dshape(ds, ds + 4);
      data = in_data[kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      out_no_affine = out_data[kOutNoAffine].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[kData].get<xpu, 4, real_t>(s);
      grad = out_grad[kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[kData].get<xpu, 4, real_t>(s);
      out = out_data[kOut].get<xpu, 4, real_t>(s);
      out_no_affine = out_data[kOutNoAffine].get<xpu, 4, real_t>(s);
    }

    Tensor<xpu, 1> mean = out_data[kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> slope = in_data[kGamma].get<xpu, 1, real_t>(s);
    // Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[kBeta].get<xpu, 1, real_t>(s);
    // get requested temp space
    Tensor<xpu, 2> workspace = ctx.requested[kTempSpace].get_space<xpu>(
        mshadow::Shape2(3, out.shape_[1]), s);
    Tensor<xpu, 1> gmean = workspace[0];
    Tensor<xpu, 1> gvar = workspace[1];
    Tensor<xpu, 1> tmp = workspace[2];
    // cal
    gvar = sumall_except_dim<1>((grad * broadcast<1>(slope, data.shape_)) *
                                (data - broadcast<1>(mean, data.shape_)) *
                                -0.5f *
                                F<mshadow_op::power>(broadcast<1>(var + param_.eps, data.shape_),
                                                     -1.5f));
    gmean = sumall_except_dim<1>(grad * broadcast<1>(slope, data.shape_));
    gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
    tmp = scale * sumall_except_dim<1>(-2.0f * (data - broadcast<1>(mean, data.shape_)));
    tmp *= gvar;
    gmean += tmp;
    // assign
    Assign(gslope, req[kGamma], sumall_except_dim<1>(grad * out_no_affine));
    Assign(gbias, req[kBeta], sumall_except_dim<1>(grad));
    Assign(grad_in, req[kData], (grad * broadcast<1>(slope, data.shape_)) *
           broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
           broadcast<1>(gvar, data.shape_) * scale * 2.0f * (data - broadcast<1>(mean,
                                                                                 data.shape_)) +
           broadcast<1>(gmean, data.shape_) * scale);
  }

 private:
  BatchNormParam param_;
};  // class BatchNormOp

template<typename xpu>
Operator *CreateOp(BatchNormParam param);


#if DMLC_USE_CXX11
class BatchNormProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(dshape);
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));
    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[kOut],
            out_data[kOut], out_data[kOutNoAffine], out_data[kMean], out_data[kVar],
            in_data[kData], in_data[kGamma], in_data[kBeta]
           };
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[kOut], in_grad[kData]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 4;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "output_no_affine", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  BatchNormParam param_;
};  // class BatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_INL_H_

