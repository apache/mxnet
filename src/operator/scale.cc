/*!
 * Copyright (c) 2015 by Contributors
 * \file scale.cc
 * \brief scale operator
*/
#include "./scale-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ScaleParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ScaleOp<cpu, DType>(param);
  });
  return op;
}

Operator* ScaleProp::CreateOperatorEx(Context ctx,
                                      std::vector<TShape> *in_shape,
                                      std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ScaleParam);

MXNET_REGISTER_OP_PROPERTY(Scale, ScaleProp)
.describe("Scale the input initialized by user and learned through backpropogation.")
.add_argument("data", "Symbol", "Input data to the ScaleOp.")
.add_arguments(ScaleParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
