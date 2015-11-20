/*!
 * Copyright (c) 2015 by Contributors
 * \file onehot.cc
 * \brief
 * \author Bing Xu
*/

#include "./onehot-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(OnehotParam param) {
  return new OnehotEmbeddingOp<cpu>(param);
}

Operator* OnehotEmbeddingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(OnehotParam);

MXNET_REGISTER_OP_PROPERTY(OnehotEmbedding, OnehotEmbeddingProp)
.describe("Get embedding for one-hot input")
.add_argument("data", "Symbol", "Input data to the OnehotEmbeddingOp.")
.add_argument("weight", "Symbol", "Enbedding weight matrix.")
.add_arguments(OnehotParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
