/*!
 * Copyright (c) 2015 by Contributors
 * \file swapaxis.cc
 * \brief
 * \author Ming Zhang
*/

#include "./swapaxis-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(SwapAxisParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SwapAxisOp<cpu, DType>(param);
  });
  return op;
}

Operator* SwapAxisProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}


DMLC_REGISTER_PARAMETER(SwapAxisParam);

MXNET_REGISTER_OP_PROPERTY(SwapAxis, SwapAxisProp)
.add_argument("data", "Symbol", "Input data to the SwapAxisOp.")
.add_arguments(SwapAxisParam::__FIELDS__())
.describe("Apply swapaxis to input.");
}  // namespace op
}  // namespace mxnet
