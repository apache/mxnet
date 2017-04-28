/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_relu-inl.h
 * \brief
 * \author Ziheng Jiang
*/

#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZED_RELU_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZED_RELU_INL_H_
#include <mxnet/operator_util.h>
#include "../operator_common.h"
#include "./quantization_utils.h"

namespace mxnet {
namespace op {


// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(int dtype);

class QuantizedReluProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {}

  std::map<std::string, std::string> GetParams() const override {
    return std::map<std::string, std::string>();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "min_range", "max_range"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "min_range", "max_range"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 3U);

    CHECK(!shape_is_none(in_shape->at(0)));
    for (int i = 1; i < 3; ++i) {
      SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
    }

    out_shape->clear();
    out_shape->push_back(in_shape->at(0));
    out_shape->push_back(TShape{1});
    out_shape->push_back(TShape{1});
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 3U);
    CHECK_EQ((*in_type)[0], mshadow::kInt8)
      << "`quantized_relu` only supports int8 input for now";
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
    auto ptr = new QuantizedReluProp();
    return ptr;
  }

  std::string TypeString() const override {
    return "quantized_relu";
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[0], out_data[0]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
};


}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZED_RELU_H_
