/*!
 * Copyright (c) 2017 by Contributors
 * \file softmax.cc
 * \brief CPU Implementation of softmax
 */
#include "./softmax-inl.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(softmax)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxCompute<gpu, mxnet_op::softmax_fwd>);

NNVM_REGISTER_OP(_backward_softmax)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxGradCompute<gpu, mshadow::op::mul,
                                                        mxnet_op::softmax_bwd>);

NNVM_REGISTER_OP(log_softmax)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxCompute<gpu, mxnet_op::log_softmax_fwd>);

NNVM_REGISTER_OP(_backward_log_softmax)
.set_attr<FCompute>("FCompute<gpu>", SoftmaxGradCompute<gpu, mshadow_op::left,
                                                        mxnet_op::log_softmax_bwd>);

}  // namespace op
}  // namespace mxnet
