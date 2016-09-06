/*!
 * Copyright (c) 2015 by Contributors
 * \file block_grad.cc
 * \brief
 * \author Bing Xu
*/
#include "./block_grad-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BlockGradientOp<cpu, DType>();
  });
  return op;
}

Operator *BlockGradientProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, in_type->at(0));
}

MXNET_REGISTER_OP_PROPERTY(BlockGrad, BlockGradientProp)
.describe("Get output from a symbol and pass 0 gradient back")
.add_argument("data", "Symbol", "Input data.");

}  // namespace op
}  // namespace mxnet

