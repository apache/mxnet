/*!
 * Copyright (c) 2015 by Contributors
 * \file flatten.cc
 * \brief
 * \author Bing Xu
*/

#include "./reshape-inl.h"


namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ReshapeParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ReshapeOp<cpu, DType>(param);
  });
  return op;
}

Operator* ReshapeProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                        std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ReshapeParam);

MXNET_REGISTER_OP_PROPERTY(Reshape, ReshapeProp)
.describe("Reshape input to target shape")
.add_argument("data", "Symbol", "Input data to reshape.")
.add_arguments(ReshapeParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(Flatten, FlattenProp)
.describe("Flatten input")
.add_argument("data", "Symbol", "Input data to flatten.");
}  // namespace op
}  // namespace mxnet
