/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_equal)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::eq>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_not_equal)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ne>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_greater)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::gt>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_greater_equal)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::ge>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_lesser)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::lt>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_lesser_equal)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::le>);

}  // namespace op
}  // namespace mxnet
