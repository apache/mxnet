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
NNVM_REGISTER_OP(_plus)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_backward_plus)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseNone<gpu, mshadow_op::identity,
                                                                mshadow_op::identity>);

NNVM_REGISTER_OP(_minus)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::minus>);

NNVM_REGISTER_OP(_backward_minus)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseNone<gpu, mshadow_op::identity,
                                                                mshadow_op::negation>);

NNVM_REGISTER_OP(_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_backward_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::right,
                                                              mshadow_op::left>);

NNVM_REGISTER_OP(_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_backward_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::div_grad,
                                                              mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::power>);

NNVM_REGISTER_OP(_backward_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::power_grad,
                                                              mshadow_op::power_rgrad>);

NNVM_REGISTER_OP(_maximum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::maximum>);

NNVM_REGISTER_OP(_backward_maximum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::ge,
                                                              mshadow_op::lt>);

NNVM_REGISTER_OP(_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::minimum>);

NNVM_REGISTER_OP(_backward_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastBackwardUseIn<gpu, mshadow_op::le,
                                                              mshadow_op::gt>);

}  // namespace op
}  // namespace mxnet
