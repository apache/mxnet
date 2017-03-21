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
NNVM_REGISTER_OP(broadcast_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::power>);

NNVM_REGISTER_OP(_backward_broadcast_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::power_grad,
                                                              mshadow_op::power_rgrad>);

NNVM_REGISTER_OP(broadcast_maximum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::maximum>);

NNVM_REGISTER_OP(_backward_broadcast_maximum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::ge,
                                                              mshadow_op::lt>);

NNVM_REGISTER_OP(broadcast_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::minimum>);

NNVM_REGISTER_OP(_backward_broadcast_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::le,
                                                              mshadow_op::gt>);

NNVM_REGISTER_OP(broadcast_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::hypot>);

NNVM_REGISTER_OP(_backward_broadcast_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::hypot_grad_left,
                                                              mshadow_op::hypot_grad_right>);

}  // namespace op
}  // namespace mxnet
