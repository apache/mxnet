/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax.cc
 * \brief
 * \author Bing Xu
*/
#include "./softmax-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxParam param) {
  return new SoftmaxOp<cpu>(param);
}

Operator *SoftmaxProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SoftmaxParam);

MXNET_REGISTER_OP_PROPERTY(Softmax, SoftmaxProp)
.describe("Perform a softmax transformation on input.")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_arguments(SoftmaxParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

