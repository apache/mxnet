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
Operator* CreateOp<cpu>(EmbeddingParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new EmbeddingOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *EmbeddingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
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
