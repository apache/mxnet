/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_fully_connected_op-inl.h
 * \brief quantized fully_connected operator and symbol
 * \author Ziheng Jiang
*/
#ifndef MXNET_OPERATOR_QUANTIZED_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_QUANTIZED_FULLY_CONNECTED_INL_H_

#include "../operator_common.h"
#include "../elemwise_op_common.h"


namespace mxnet {
namespace op {

struct QuantizedFullyConnectedParam :
  public dmlc::Parameter<QuantizedFullyConnectedParam> {
  uint32_t num_hidden;
  bool no_bias;
  DMLC_DECLARE_PARAMETER(QuantizedFullyConnectedParam) {
    DMLC_DECLARE_FIELD(num_hidden);
    DMLC_DECLARE_FIELD(no_bias).set_default(false);
  }
};

template<typename xpu>
Operator* CreateOp(int dtype,
                   const Context& ctx,
                   const std::vector<TShape>& in_shape,
                   const std::vector<TShape>& out_shape,
                   const QuantizedFullyConnectedParam& param);

class QuantizedFullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (param_.no_bias) {
      return {"data", "weight", "min_data", "max_data",
          "min_weight", "max_weight"};
    } else {
      return {"data", "weight", "bias", "min_data", "max_data",
          "min_weight", "max_weight", "min_bias", "max_bias"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    return {"out", "min_out", "max_out"};
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
    uint32_t num_inputs = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape->size(), num_inputs * 3);

    CHECK(!shape_is_none(in_shape->at(0)));
    const TShape& dshape = in_shape->at(0);
    CHECK_EQ(dshape[1] % 4, 0)
      << "for 8bit fully connected, dshape[1] must be multiple of 4";

    TShape wshape = Shape2(param_.num_hidden, dshape[1]);
    SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
    if (!param_.no_bias) {
      TShape bshape = Shape1(param_.num_hidden);
      SHAPE_ASSIGN_CHECK(*in_shape, 2, bshape);
    }

    for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
    }

    out_shape->clear();
    out_shape->push_back(TShape{dshape[0], wshape[0]});
    out_shape->push_back(TShape{1});
    out_shape->push_back(TShape{1});
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    uint32_t num_inputs = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_type->size(), num_inputs * 3);

    for (size_t i = 0; i < num_inputs; ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kInt8);
    }
    for (size_t i = num_inputs; i < 3 * num_inputs; ++i) {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
    }

    out_type->clear();
    out_type->push_back(mshadow::kInt32);
    out_type->push_back(mshadow::kFloat32);
    out_type->push_back(mshadow::kFloat32);
    return true;
  }

  OperatorProperty* Copy() const override {
    QuantizedFullyConnectedProp* prop = new QuantizedFullyConnectedProp();
    prop->param_ = this->param_;
    return prop;
  }

  std::string TypeString() const override {
    return "quantized_fully_connected";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  QuantizedFullyConnectedParam param_;
};  // class QuantizedFullyConnectedSymbol

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZED_FULLY_CONNECTED_INL_H_
