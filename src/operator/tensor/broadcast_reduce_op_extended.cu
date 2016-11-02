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
