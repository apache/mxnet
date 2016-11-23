/*!
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
// this will be invoked by gcc and compile GPU version
#include "./ordering_op-inl.h"


namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_topk)
.set_attr<FCompute>("FCompute<gpu>", TopK<gpu>);

NNVM_REGISTER_OP(_backward_topk)
.set_attr<FCompute>("FCompute<gpu>", TopKBackward_<gpu>);
}  // namespace op
}  // namespace mxnet
