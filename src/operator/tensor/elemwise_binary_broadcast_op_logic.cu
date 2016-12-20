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

NNVM_REGISTER_OP(broadcast_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(broadcast_not_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::ne>);

NNVM_REGISTER_OP(broadcast_greater)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::gt>);

NNVM_REGISTER_OP(broadcast_greater_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::ge>);

NNVM_REGISTER_OP(broadcast_lesser)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::lt>);

NNVM_REGISTER_OP(broadcast_lesser_equal)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu, mshadow_op::le>);

}  // namespace op
}  // namespace mxnet
