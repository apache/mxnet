/*!
 * Copyright (c) 2017 by Contributors
 * \file sparse_retain.cu
 * \brief
*/

#include "./sparse_retain-inl.h"
namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sparse_retain)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseRetainOpForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_sparse_retain)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseRetainOpBackwardEx<gpu>);

}  // namespace op
}  // namespace mxnet
