/*!
 * Copyright (c) 2016 by Contributors
 * \file indexing_op.cu
 * \brief
 * \author Siyi Li
*/

#include "./indexing_op.h"
namespace mxnet {
namespace op {
NNVM_REGISTER_OP(Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpForward<gpu>);

NNVM_REGISTER_OP(_backward_Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpBackward<gpu>);
}  // namespace op
}  // namespace mxnet

