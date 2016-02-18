/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_batch_norm-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_CUDNN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_CUDNN_BATCH_NORM_INL_H_
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "./batch_norm-inl.h"

namespace mxnet {
namespace op {

namespace cudnnbatchnorm {
enum CuDNNBatchNormOpInputs {kData, kGamma, kBeta};
enum CuDNNBatchNormOpOutputs {kOut, kMean, kInvVar};
enum CuDNNBatchNormOpAuxiliary {kMovingMean, kMovingInvVar};
}  // namespace cudnnbatchnorm

struct CuDNNBatchNormParam : public dmlc::Parameter<CuDNNBatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  DMLC_DECLARE_PARAMETER(CuDNNBatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(false)
    .describe("Fix gamma while training");
  }
};

template<typename xpu>
Operator *CreateOp(CuDNNBatchNormParam param);


#if DMLC_USE_CXX11
class CuDNNBatchNormProp : public OperatorProperty {
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
    out_shape->push_back(Shape1(dshape[1]));
    out_shape->push_back(Shape1(dshape[1]));

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));
    aux_shape->push_back(Shape1(dshape[1]));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new CuDNNBatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "CuDNNBatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[cudnnbatchnorm::kOut],
            out_data[cudnnbatchnorm::kMean],
            out_data[cudnnbatchnorm::kInvVar],
            in_data[cudnnbatchnorm::kData],
            in_data[cudnnbatchnorm::kGamma],
            in_data[cudnnbatchnorm::kBeta]
           };
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "inv_var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_inv_var"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  CuDNNBatchNormParam param_;
};  // class BatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUDNN_BATCH_NORM_INL_H_
