/*!
 * Copyright (c) 2015 by Contributors
 * \file make_loss.cc
 * \brief special layer for propagating loss
*/
#include "./make_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MakeLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MakeLossOp<cpu, DType>(param);
  });
  return op;
}

Operator *MakeLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MakeLossParam);

MXNET_REGISTER_OP_PROPERTY(MakeLoss, MakeLossProp)
.describe("Get output from a symbol and pass 1 gradient back. "
"This is used as a terminal loss if unary and binary operator "
"are used to composite a loss with no declaration of backward "
"dependency")
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_arguments(MakeLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
