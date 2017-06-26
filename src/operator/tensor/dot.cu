/*!
 *  Copyright (c) 2017 by Contributors
 * \file dot.cu
 * \brief GPU Implementation of matrix dot
 */

#include "./dot-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(dot)
.set_attr<FCompute>("FCompute<gpu>", DotForward_<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", DotForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_dot)
.set_attr<FCompute>("FCompute<gpu>", DotBackward_<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", DotBackwardEx<gpu>);

NNVM_REGISTER_OP(batch_dot)
.set_attr<FCompute>("FCompute<gpu>", BatchDotForward_<gpu>);

NNVM_REGISTER_OP(_backward_batch_dot)
.set_attr<FCompute>("FCompute<gpu>", BatchDotBackward_<gpu>);

}  // namespace op
}  // namespace mxnet
