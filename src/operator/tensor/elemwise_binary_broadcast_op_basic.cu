/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(broadcast_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_backward_broadcast_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseNone<gpu, mshadow_op::identity,
                                                                mshadow_op::identity>);

NNVM_REGISTER_OP(broadcast_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::minus>);

NNVM_REGISTER_OP(_backward_broadcast_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseNone<gpu, mshadow_op::identity,
                                                                mshadow_op::negation>);

NNVM_REGISTER_OP(broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_backward_broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::right,
                                                                mshadow_op::left>);

NNVM_REGISTER_OP(broadcast_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_backward_broadcast_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::div_grad,
                                                                mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(broadcast_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_broadcast_mod)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::mod_grad,
                                                                  mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
