/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu.cc
 * \brief
 * \author Bing Xu
*/

#include "./leaky_relu-inl.h"

#include <nnvm/op_attr_types.h>
namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(LeakyReLUParam param) {
  return new LeakyReLUOp<cpu>(param);
}

Operator *LeakyReLUProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(LeakyReLUParam);

MXNET_REGISTER_OP_PROPERTY(LeakyReLU, LeakyReLUProp)
.describe("Apply activation function to input.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(LeakyReLUParam::__FIELDS__());

NNVM_REGISTER_OP(LeakyReLU)
.set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
    [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
      if (index == 1 && var->attrs.dict.find("__init__") == var->attrs.dict.end()) {
        var->attrs.dict["__init__"] = "[\"Constant\", {\"value\": 0.25}]";
      }
    });

}  // namespace op
}  // namespace mxnet

