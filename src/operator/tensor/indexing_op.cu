/*!
 * Copyright (c) 2017 by Contributors
 * \file indexing_op.cu
 * \brief
 * \author Siyi Li, Chi Zhang
*/

#include "./indexing_op.h"
namespace mxnet {
namespace op {
NNVM_REGISTER_OP(Embedding)
.set_attr<FCompute>("FCompute<gpu>", TakeOpForward<gpu>);

NNVM_REGISTER_OP(Take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpForward<gpu>);

NNVM_REGISTER_OP(_backward_Take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpBackward<gpu>);
}  // namespace op
}  // namespace mxnet

