/*!
 * Copyright (c) 2015 by Contributors
 * \file svm_output-inl.h
 * \brief
 * \author Jonas Amaro
*/
#ifndef MXNET_OPERATOR_SVM_OUTPUT_INL_H_
#define MXNET_OPERATOR_SVM_OUTPUT_INL_H_

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

namespace svm_enum {
enum SVMOutputOpInputs {kData, kLabel};
enum SVMOutputOpOutputs {kOut};
enum SVMOutputNormType {kNull, kBatch, kValid};
enum SVMOutputOpResource {kTempSpace};
}  // namespace svm_enum


struct SVMOutputParam : public dmlc::Parameter<SVMOutputParam> {
  float margin;
  float regularization_coefficient;
  bool use_linear;
  DMLC_DECLARE_PARAMETER(SVMOutputParam) {
    DMLC_DECLARE_FIELD(margin).set_default(1.0f)
    .describe("The loss function penalizes outputs that lie outside this margin. "
        "Default margin is 1.");
    DMLC_DECLARE_FIELD(regularization_coefficient).set_default(1.0f)
    .describe("Regularization parameter for the SVM. "
        "This balances the tradeoff between coefficient size and error.");
    DMLC_DECLARE_FIELD(use_linear).set_default(false)
    .describe("Whether to use L1-SVM objective. L2-SVM objective is used by default.");
  };
};

template<typename xpu, typename DType>
class SVMOutputOp : public Operator {
 public:
  explicit SVMOutputOp(SVMOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U) << "Expecting [data, label]";
    CHECK_EQ(out_data.size(), 1U) << "Expecting [output]";
    CHECK_EQ(req.size(), 1U) << "Expecting output.size() == req.size()";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[svm_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[svm_enum::kOut].FlatTo2D<xpu, DType>(s);
    Assign(out, req[svm_enum::kOut], F<mshadow_op::identity>(data));
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
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_GE(in_grad.size(), 1U);
    CHECK_GE(req.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& label_shape = in_data[svm_enum::kLabel].shape_;

    Tensor<xpu, 1, DType> label = in_data[svm_enum::kLabel].get_with_shape<xpu, 1, DType>(
        Shape1(label_shape.ProdShape(0, label_shape.ndim())), s);
    Tensor<xpu, 2, DType> out = out_data[svm_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = in_grad[svm_enum::kData].FlatTo2D<xpu, DType>(s);
    CHECK_EQ(grad.shape_, out.shape_) << "SVMOutputs: shape mismatch";

    if (param_.use_linear) {
      L1_SVM(DType(param_.margin), DType(param_.regularization_coefficient), grad, label, out);
    } else {
      L2_SVM(DType(param_.margin), DType(param_.regularization_coefficient), grad, label, out);
    }
  }

 private:
  SVMOutputParam param_;
};  // class SVMOutputOp

// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SVMOutputParam param, int dtype);

#if DMLC_USE_CXX11
class SVMOutputProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    TShape label_shape(dshape.ndim() - 1);
    for (index_t i = 0; i + 1 < dshape.ndim(); ++i)
      label_shape[i] = dshape[i];
    SHAPE_ASSIGN_CHECK(*in_shape, svm_enum::kLabel, label_shape);
    out_shape->clear();
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
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SVMOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SVMOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[svm_enum::kLabel], out_data[svm_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[svm_enum::kOut], in_grad[svm_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[svm_enum::kData], out_data[svm_enum::kOut]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  SVMOutputParam param_;
};  // class SVMOutputProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SVM_OUTPUT_INL_H_
