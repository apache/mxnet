/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu.cc
 * \brief
 * \author Bing Xu
*/

#include "./leaky_relu-inl.h"

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

}  // namespace op
}  // namespace mxnet

