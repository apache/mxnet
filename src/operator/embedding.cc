/*!
 * Copyright (c) 2015 by Contributors
 * \file embedding.cc
 * \brief
 * \author Bing Xu
*/

#include "./embedding-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(EmbeddingParam param) {
  return new EmbeddingOp<cpu>(param);
}

Operator* EmbeddingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(EmbeddingParam);

MXNET_REGISTER_OP_PROPERTY(Embedding, EmbeddingProp)
.describe("Get embedding for one-hot input. A n-dimensional input tensor will "
"be trainsformed into a (n+1)-dimensional tensor, where a new dimension is "
"added for the embedding results.")
.add_argument("data", "Symbol", "Input data to the EmbeddingOp.")
.add_argument("weight", "Symbol", "Enbedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
