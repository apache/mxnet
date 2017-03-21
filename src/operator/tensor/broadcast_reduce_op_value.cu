/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cu
 * \brief GPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(sum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::sum>);

NNVM_REGISTER_OP(_backward_sum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseNone<gpu>);

NNVM_REGISTER_OP(mean)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::sum, true>);

NNVM_REGISTER_OP(_backward_mean)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseNone<gpu, true>);

NNVM_REGISTER_OP(prod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow_op::product>);

NNVM_REGISTER_OP(_backward_prod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::rdiv>);

NNVM_REGISTER_OP(nansum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow_op::nansum>);

NNVM_REGISTER_OP(_backward_nansum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::nansum_grad>);

NNVM_REGISTER_OP(nanprod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow_op::nanprod>);

NNVM_REGISTER_OP(_backward_nanprod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::nanprod_grad>);

NNVM_REGISTER_OP(max)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::maximum>);

NNVM_REGISTER_OP(_backward_max)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(min)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::minimum>);

NNVM_REGISTER_OP(_backward_min)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(broadcast_axis)
.set_attr<FCompute>("FCompute<gpu>", BroadcastCompute<gpu>);

NNVM_REGISTER_OP(broadcast_to)
.set_attr<FCompute>("FCompute<gpu>", BroadcastCompute<gpu>);

NNVM_REGISTER_OP(_broadcast_backward)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesCompute<gpu, mshadow::red::sum>);

NNVM_REGISTER_OP(norm)
.set_attr<FCompute>("FCompute<gpu>", L2NormCompute<gpu>);

}  // namespace op
}  // namespace mxnet
