/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_sum.cc
 * \brief elementwise sum operator
*/
#include "./elementwise_sum-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ElementWiseSumParam param) {
  return new ElementWiseSumOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* ElementWiseSumProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

MXNET_REGISTER_OP_PROPERTY(ElementWiseSum, ElementWiseSumProp)
.describe("Perform an elementwise sum over all the inputs.")
.add_arguments(ElementWiseSumParam::__FIELDS__())
.set_key_var_num_args("num_args");

}  // namespace op
}  // namespace mxnet
