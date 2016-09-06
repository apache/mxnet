/*!
 * Copyright (c) 2015 by Contributors
 * \file elementwise_sum.cc
 * \brief elementwise sum operator
*/
#include "./elementwise_sum-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ElementWiseSumParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ElementWiseSumOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from static_operator_common.h
Operator* ElementWiseSumProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

MXNET_REGISTER_OP_PROPERTY(ElementWiseSum, ElementWiseSumProp)
.describe("Perform an elementwise sum over all the inputs.")
.add_arguments(ElementWiseSumParam::__FIELDS__())
.set_key_var_num_args("num_args");

}  // namespace op
}  // namespace mxnet
