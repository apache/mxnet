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
Operator *CreateOp<cpu>() {
  return new BlockGradientOp<cpu>();
}

Operator *BlockGradientProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp);
}

MXNET_REGISTER_OP_PROPERTY(BlockGrad, BlockGradientProp)
.describe("Get output from a symbol and pass 0 gradient back")
.add_argument("data", "Symbol", "Input data.");

}  // namespace op
}  // namespace mxnet

