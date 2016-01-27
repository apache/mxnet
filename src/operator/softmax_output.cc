/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu
*/
#include "./softmax_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxOutputParam param) {
  return new SoftmaxOutputOp<cpu>(param);
}

Operator *SoftmaxOutputProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SoftmaxOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxOutput, SoftmaxOutputProp)
.describe("Perform a softmax transformation on input, backprop with logloss.")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_argument("label", "Symbol", "Label data.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Softmax, DeprecatedSoftmaxProp)
.describe("DEPRECATED: Perform a softmax transformation on input. Please use SoftmaxOutput")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

