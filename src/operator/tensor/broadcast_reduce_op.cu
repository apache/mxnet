/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cu
 * \brief GPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(sum)
.attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::sum>);

NNVM_REGISTER_OP(_backward_sum)
.attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseNone<gpu>);

NNVM_REGISTER_OP(max)
.attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::maximum>);

NNVM_REGISTER_OP(_backward_max)
.attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(min)
.attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::minimum>);

NNVM_REGISTER_OP(_backward_min)
.attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(broadcast_axis)
.attr<FCompute>("FCompute<gpu>", BroadcastCompute<gpu>);

NNVM_REGISTER_OP(broadcast_to)
.attr<FCompute>("FCompute<gpu>", BroadcastCompute<gpu>);

}  // namespace op
}  // namespace mxnet
