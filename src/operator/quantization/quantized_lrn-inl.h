/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_lrn-inl.h
 * \brief
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZED_LRN_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZED_LRN_INL_H_
#include <mxnet/operator.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

struct QuantizedLRNParam : public dmlc::Parameter<QuantizedLRNParam> {
  float alpha;
  float beta;
  float knorm;
  uint32_t nsize;
  DMLC_DECLARE_PARAMETER(QuantizedLRNParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(1e-4f)
    .describe("The variance scaling parameter :math:`\alpha` in the LRN expression.");
    DMLC_DECLARE_FIELD(beta).set_default(0.75f)
    .describe("The power parameter :math:`\beta` in the LRN expression.");
    DMLC_DECLARE_FIELD(knorm).set_default(2.0f)
    .describe("The parameter :math:`k` in the LRN expression.");
    DMLC_DECLARE_FIELD(nsize)
    .describe("normalization window width in elements.");
  }
};  // struct QuantizedLRNParam

template<typename xpu>
Operator *CreateOp(QuantizedLRNParam param, int dtype);

class QuantizedLRNProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    CHECK(!shape_is_none(dshape));
    for (int i = 1; i < 3; ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
    }

    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(TShape{1});
    out_shape->push_back(TShape{1});
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 3U);
    CHECK_EQ((*in_type)[0], mshadow::kInt8)
      << "`dequantized_lrn` only supports int8 input for now";
    for (int i = 1; i < 3; ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
    }

    out_type->clear();
    out_type->push_back(mshadow::kInt8);
    out_type->push_back(mshadow::kFloat32);
    out_type->push_back(mshadow::kFloat32);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QuantizedLRNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "quantized_lrn";
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "min_data", "max_data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "min_out", "max_out"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedLRNParam param_;
};  // QuantizedLRNProp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZED_LRN_INL_H_
